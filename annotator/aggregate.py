# -*- coding: utf-8 -*-
"""全局聚合 M3(Res 4)—— 扩充“不限于三方面”的内容维度。

新增产物:
- themes[]: 2~5 个主题,每个挂 ≥2 个支持事件的 span 证据;
- turning_points[]: 故事转折点 3~7 个,{span, desc, importance};
- motifs/clues[]: 伏笔/呼应线索(服务 L5_foreshadow 自动出题);
- timeline 事件-场景对齐度字段。

因果边增强可信度(Res 4.2):
- 每条边做时序核验: causes 事件 end <= 被引用事件 start,违例不写回并统计;
- 因果边只允许指向 boundary_state ∈ {verified, refined} 的事件(REVIEW D6 补偿)。

人物表(Res 4.3):first_seen_ts / total_screen_time / appearance_count 由场景
数据确定性计算(非 LLM 臆造),支撑 L5_relation_graph、L2_counting 出题。
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from . import prompts
from .config import RunConfig
from .llm_client import LLMClient
from .utils import interval_overlap_ratio

log = logging.getLogger("annotator.aggregate")

_VERIFIED_STATES = {"verified", "refined"}


def _char_stats_from_scenes(scenes: List[Dict]) -> Dict[str, Dict]:
    """人物统计:首次出现 / 总出镜时长 / 出现次数(确定性计算,Res 4.3)。

    键同时收录原始称谓与其归一化形式,便于与 LLM 人物表的姓名做匹配(B15)。
    """
    stats: Dict[str, Dict] = {}
    for sc in scenes:
        try:
            s, e = float(sc["start"]), float(sc["end"])
        except (KeyError, TypeError, ValueError):
            continue                      # 修复 D1:场景来自 LLM,缺字段不应崩溃
        for name in sc.get("characters") or []:
            if not isinstance(name, str) or not name.strip():
                continue
            st = stats.setdefault(name, {"first_seen_ts": None, "total_screen_time": 0.0,
                                         "appearance_count": 0})
            st["first_seen_ts"] = s if st["first_seen_ts"] is None else min(st["first_seen_ts"], s)
            st["total_screen_time"] += max(0.0, e - s)
            st["appearance_count"] += 1
    return {k: {"first_seen_ts": round(v["first_seen_ts"], 3) if v["first_seen_ts"] is not None else None,
                "total_screen_time": round(v["total_screen_time"], 3),
                "appearance_count": v["appearance_count"]}
            for k, v in stats.items()}


def _norm_name(s: Optional[str]) -> str:
    """人物名归一化:小写、去空白与下划线/连字符,便于跨命名体系匹配。"""
    return re.sub(r"[\s_\-·]+", "", str(s or "")).lower()


def _attach_char_stats(characters: List[Dict], char_stats: Dict[str, Dict],
                       report: Dict) -> None:
    """把确定性统计挂到 LLM 人物表上(修复 B15)。

    场景里的 characters 是 prompt 要求的临时称谓(如 man_in_red),而全局人物表给的
    是真实姓名/角色名 —— 此前直接用姓名去查临时称谓表,命中率接近 0,
    first_seen_ts / total_screen_time / appearance_count 基本永远是空的。
    这里改为:精确名 → 归一化名 → 别名(aliases/tmp_ids)三级匹配,并把未匹配上的
    临时称谓作为人工合并清单报出。
    """
    norm_index: Dict[str, str] = {}
    for raw in char_stats:
        norm_index.setdefault(_norm_name(raw), raw)
    matched = set()

    for ch in characters:
        if not isinstance(ch, dict):
            continue
        name = ch.get("name")
        if isinstance(name, str):
            name = {"zh": name, "en": None}
            ch["name"] = name
        name = name or {}
        candidates = [name.get("en"), name.get("zh"), ch.get("id")]
        candidates += [a for a in (ch.get("aliases") or ch.get("tmp_ids") or [])
                       if isinstance(a, str)]
        hit = None
        for cand in candidates:
            if not cand:
                continue
            if cand in char_stats:
                hit = cand
                break
            norm = norm_index.get(_norm_name(cand))
            if norm:
                hit = norm
                break
        if hit:
            ch.update(char_stats[hit])
            ch.setdefault("scene_alias", hit)
            matched.add(hit)

    unmatched = [k for k in char_stats if k not in matched]
    if unmatched:
        log.warning("%d 个场景临时称谓未能并入人物表(需人工合并):%s",
                    len(unmatched), unmatched[:10])
        report.setdefault("character_unmerged", []).extend(
            {"kind": "character_unmerged", "alias": k, **char_stats[k]}
            for k in unmatched)


def _validate_causal(events: List[Dict], causal: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """因果边时序核验(Res 4.2):cause.end <= effect.start 且两端边界已验证。"""
    by_id = {e["id"]: e for e in events}
    valid, invalid = [], []
    for c in causal:
        eff_id, cause_ids = c.get("event"), c.get("causes") or []
        eff = by_id.get(eff_id)
        if not eff:
            continue
        ok_causes = []
        for cid in cause_ids:
            cause = by_id.get(cid)
            if not cause or cause is eff:
                invalid.append({"kind": "causal_missing", "event": eff_id, "cause": cid,
                                "reason": "cause 不存在或自环"})
                continue
            if not (cause.get("boundary_state") in _VERIFIED_STATES
                    and eff.get("boundary_state") in _VERIFIED_STATES):
                invalid.append({"kind": "causal_unverified_boundary",
                                "event": eff_id, "cause": cid,
                                "reason": "因果边只允许指向已验证边界的事件"})
                continue
            if cause["span"][1] > eff["span"][0] + 1e-6:
                invalid.append({"kind": "causal_time_violation",
                                "event": eff_id, "cause": cid,
                                "reason": f"cause.end {cause['span'][1]:.1f}s > effect.start {eff['span'][0]:.1f}s"})
                continue
            ok_causes.append(cid)
        if ok_causes:
            valid.append({"event": eff_id, "causes": ok_causes})
    return valid, invalid


def _timeline_events(events: List[Dict], llm_timeline: List[str]) -> List[str]:
    """时间线:LLM 输出经 id 校验后使用;缺失/非法则按 span 起点排序兜底。"""
    valid_ids = {e["id"] for e in events}
    tl = [t for t in (llm_timeline or []) if t in valid_ids]
    if len(tl) != len(valid_ids):
        tl = [e["id"] for e in sorted(events, key=lambda e: (e["span"][0], e["id"]))]
    return tl


def aggregate_global(client: LLMClient, cfg: RunConfig, scenes: List[Dict],
                     events: List[Dict], meta: Dict) -> Tuple[Dict, List[Dict], Dict]:
    """M3 主入口:纯文本聚合 + 因果核验回填 + 人物统计。返回 (global_block, events, report)。"""
    report: Dict = {"causal_invalid": [], "scene_alignment": []}
    if not scenes and not events:
        return {"synopsis": {"zh": None, "en": None}, "characters": [], "relations": [],
                "timeline": [], "themes": [], "turning_points": [], "motifs": []}, events, report

    try:
        out = client.complete(
            [{"role": "system", "content": prompts.GLOBAL_SYS},
             {"role": "user", "content": prompts.global_user(scenes, events, meta)}],
            vision=False,
        )
    except Exception as e:  # noqa: BLE001
        log.error("全局聚合调用失败:%s(将产出最小块,请重跑)", e)
        out = None
    if not isinstance(out, dict):
        # 修复 D2:此前 out.get("causal") 在 try 之外,模型返回 list 时
        # AttributeError 不被捕获,整个 S4 崩掉
        log.error("全局聚合返回非对象(%s),产出最小块", type(out).__name__)
        return {"synopsis": {"zh": None, "en": None}, "characters": [], "relations": [],
                "timeline": _timeline_events(events, []), "themes": [],
                "turning_points": [], "motifs": []}, events, report

    # 因果边:时序核验 + 已验证边界约束,只回填合法边(Res 4.2)
    causal, invalid = _validate_causal(events, out.get("causal") or [])
    report["causal_invalid"] = invalid
    cause_map = {c["event"]: c["causes"] for c in causal}
    for ev in events:
        ev["causes"] = [c for c in cause_map.get(ev["id"], [])]
    if invalid:
        log.warning("%d 条因果边未通过时序/边界核验,已丢弃并入人工清单", len(invalid))

    timeline = _timeline_events(events, out.get("timeline") or [])

    # 人物统计(确定性,Res 4.3):场景里的临时称谓按名称并入 LLM 人物表
    characters = [c for c in (out.get("characters") or []) if isinstance(c, dict)]
    _attach_char_stats(characters, _char_stats_from_scenes(scenes), report)

    # 事件-场景对齐度(Res 4.1):事件是否被场景边界错位
    scene_span = {}
    for s in scenes:
        try:
            scene_span[s.get("id")] = (float(s["start"]), float(s["end"]))
        except (KeyError, TypeError, ValueError):
            continue
    for ev in events:
        sp = scene_span.get(ev.get("scene_id"))
        if sp:
            over = interval_overlap_ratio(ev["span"], sp)
            ev["scene_alignment"] = round(over, 3)
            if over < 0.5:
                report["scene_alignment"].append({"event": ev["id"],
                                                  "scene": ev.get("scene_id"),
                                                  "overlap": over})

    global_block = {
        "synopsis": out.get("synopsis", {"zh": None, "en": None}),
        "characters": characters,
        "relations": out.get("relations", []),
        "timeline": timeline,
        "themes": out.get("themes", []),
        "turning_points": out.get("turning_points", []),
        "motifs": out.get("motifs", []),
    }
    log.info("全局聚合完成:%d 人物 / %d 关系 / %d 主题 / %d 转折 / %d 线索 / timeline %d 事件",
             len(characters), len(global_block["relations"]), len(global_block["themes"]),
             len(global_block["turning_points"]), len(global_block["motifs"]), len(timeline))
    return global_block, events, report
