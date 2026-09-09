# -*- coding: utf-8 -*-
"""QA 引擎 M4(Res 5)—— 核心需求 2:详细且可信的 QA。

- 证据锚定出题(Res 5.1):锚点池 = 关键事件 ∪ 普通事件 ∪ 显著镜头 ∪ 音频事件;
  每题强制绑定 evidence_spans(取自锚点 span),题干内置 referring 上下文:
    含时间描述 / 序数消歧词 / 事件名·地点·人物锁定短语 → 不满足重写 ≤2 次;
    仍不过 → 剔除并计数 reused_anchor。
- 能力矩阵硬约束(Res 5.2):L1~L5 每层 ≥2 题;cross_scene/whole_video ≥50%;
  task_type: mcq 50~70%、temporal 15~25%;时长 4 桶每桶 ≥15%;refresh_qa_coverage
  输出矩阵报告,失败即发布拦截。
- 题目数量按时长(Res 5.3):qa_quota = ceil(duration_hour × 24),下限 10,分批出。
- 非 MCQ 规范(Res 5.5):open/summary 参考答案 zh≥80/en≥220 词 + ≥2 细节;
  temporal_target 宽度 ≥5s、落在 anchor_id 对应的锚点 span 内、不越出视频时长。
- span 几何校验(Res 5.1):evidence_spans 每段必须落在锚点池中某个锚点的 span 内,
  且至少一段与 anchor_id 对应的锚点相交;越界/倒置/引用未知锚点一律打回重写。
  区间本身的画面级核验在 S6(qa_verify 的 mIoU 通道)完成。
- 防梗概泄漏(Res 5.4):出题输入不含 global.synopsis / detailed_description。
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from . import prompts
from .config import RunConfig
from .llm_client import LLMClient
from .utils import en_len, has_referring_ctx, interval_overlap_ratio, zh_len

log = logging.getLogger("annotator.qa")

_VALID_CAP = set(prompts.CAPABILITY_VOCAB)

#: schema 里 task_type 是必填枚举。此前 _valid 只对已知类型做分支校验,缺失或拼错
#: 的 task_type 会一路通过、直到落盘校验才暴露 —— 那时整题的开销已经花掉了。
_VALID_TASK_TYPES = frozenset({"mcq", "open", "summary", "temporal_grounding"})


# =========================================================================
# 锚点池(Res 5.1)
# =========================================================================
def build_anchor_pool(scenes: List[Dict], events: List[Dict],
                      structure: Dict) -> List[Dict]:
    """锚点池:关键事件(优先)∪ 普通事件 ∪ 显著镜头 ∪ 音频事件。"""
    pool: List[Dict] = []
    for ev in events:
        if not ev.get("span"):
            continue
        pool.append({
            "anchor_id": ev["id"], "span": ev["span"],
            "desc": ev.get("desc", {}),
            "modality": ev.get("modality", ["visual"]),
            "importance": ev.get("importance", "minor"),
            "type": ev.get("type", "event"),
        })
    pool.sort(key=lambda a: (0 if a["importance"] == "key" else 1, a["span"][0]))
    for i, sh in enumerate(structure.get("shots") or []):
        if sh.get("desc", {}).get("zh"):
            pool.append({"anchor_id": f"shot_{sh.get('id', i)}",
                         "span": [sh["start"], sh["end"]],
                         "desc": sh["desc"], "modality": ["visual"],
                         "importance": "minor", "type": "shot"})
    for i, ae in enumerate(structure.get("audio_events") or []):
        pool.append({"anchor_id": f"audio_{i + 1}",
                     "span": [ae["start"], ae["end"]],
                     "desc": ae.get("desc") or {"zh": "", "en": ""},
                     "modality": ["audio"], "importance": "minor", "type": "audio"})
    pool.sort(key=lambda a: a["span"][0])
    return pool


# =========================================================================
# 校验与规范化
# =========================================================================
def _option_len(opt: Dict) -> int:
    """选项长度:取中英两侧的最大值。

    此前只统计 text.zh,纯英文选项一律得到 0,于是 `max(lens)==lens[answer]`
    恒成立、所有英文 MCQ 都被误判为"正确项是唯一最长选项"(B3)。
    """
    text = opt.get("text") if isinstance(opt, dict) else None
    if isinstance(text, str):
        return zh_len(text)
    if not isinstance(text, dict):
        return 0
    return max(zh_len(text.get("zh") or ""), en_len(text.get("en") or ""))


#: 锚点边界与模型自述 span 的容差(秒)。事件 span 本身按 1fps 帧索引复核,
#: 逐帧对齐要求过严会把合理题目误杀。
_SPAN_TOLERANCE = 1.0


def _as_pair(span) -> Optional[Tuple[float, float]]:
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return None
    try:
        s, e = float(span[0]), float(span[1])
    except (TypeError, ValueError):
        return None
    return (s, e) if e > s else None


def _within(inner: Tuple[float, float], outer: Tuple[float, float],
            tol: float = _SPAN_TOLERANCE) -> bool:
    return inner[0] >= outer[0] - tol and inner[1] <= outer[1] + tol


def _anchor_spans(anchors_by_id: Optional[Dict[str, Dict]]) -> List[Tuple[float, float]]:
    spans = []
    for a in (anchors_by_id or {}).values():
        p = _as_pair(a.get("span"))
        if p:
            spans.append(p)
    return spans


def _check_spans(q: Dict, anchors_by_id: Optional[Dict[str, Dict]],
                 duration: float) -> str:
    """span 几何校验:证据区间必须取自锚点池且不越界(Res 5.1 硬性要求 5/6)。"""
    pool = _anchor_spans(anchors_by_id)
    hi = duration if duration > 0 else None
    spans: List[Tuple[float, float]] = []
    for raw in q.get("evidence_spans") or []:
        p = _as_pair(raw)
        if p is None:
            return f"evidence_spans 含非法区间: {raw}"
        if p[0] < 0 or (hi is not None and p[1] > hi + _SPAN_TOLERANCE):
            return f"evidence_span {list(p)} 越出视频时长 [0, {hi}]"
        if pool and not any(_within(p, a) for a in pool):
            return f"evidence_span {list(p)} 不在任何锚点 span 内"
        spans.append(p)
    if not spans:
        return "evidence_spans 为空"

    anchor = (anchors_by_id or {}).get(q.get("anchor_id"))
    anchor_span = _as_pair((anchor or {}).get("span"))
    if anchor is not None and anchor_span is None:
        return f"anchor_id={q.get('anchor_id')} 的 span 非法"
    if anchors_by_id and anchor is None:
        return f"anchor_id={q.get('anchor_id')} 不在锚点池中"
    if anchor_span and not any(
            interval_overlap_ratio(sp, anchor_span) > 0 for sp in spans):
        return "evidence_spans 与 anchor_id 对应的锚点区间没有交集"

    if q.get("task_type") == "temporal_grounding":
        tgt = _as_pair(q.get("temporal_target"))
        if tgt is None:
            return "temporal_target 非法"
        if tgt[0] < 0 or (hi is not None and tgt[1] > hi + _SPAN_TOLERANCE):
            return f"temporal_target {list(tgt)} 越出视频时长 [0, {hi}]"
        if anchor_span and not _within(tgt, anchor_span):
            return (f"temporal_target {list(tgt)} 未落在锚点 span "
                    f"{list(anchor_span)} 内")
    return ""


def _valid(q: Dict, anchors_by_id: Optional[Dict[str, Dict]] = None,
           duration: float = 0.0) -> str:
    """结构 + span 几何校验;返回 '' 表示合法,否则返回原因。"""
    if not q.get("question") or not q.get("evidence_spans"):
        return "缺 question 或 evidence_spans"
    if q.get("task_type") not in _VALID_TASK_TYPES:
        return f"task_type 非法或缺失: {q.get('task_type')!r}"
    span_problem = _check_spans(q, anchors_by_id, duration)
    if span_problem:
        return span_problem
    if q.get("task_type") == "mcq":
        opts = q.get("options", [])
        if len(opts) != 4 or q.get("answer") not in {"A", "B", "C", "D"}:
            return "MCQ 选项/答案不完整"
        lens = [_option_len(o) for o in opts]
        if max(lens) <= 0:
            return "MCQ 选项文本为空"
        import statistics
        mean, std = statistics.fmean(lens), statistics.pstdev(lens)
        if mean > 0 and std / mean >= 0.35:
            return f"选项长度方差过大({std / mean:.2f} ≥ 0.35)"
        # 只有当正确项【严格唯一】最长时才构成长度捷径;并列最长不算(B3)
        ans_len = lens[ord(q["answer"]) - ord("A")]
        if ans_len == max(lens) and lens.count(ans_len) == 1:
            return "正确项是唯一最长选项"
    if q.get("task_type") == "temporal_grounding":
        if not q.get("temporal_target") or len(q["temporal_target"]) != 2:
            return "temporal_grounding 缺 temporal_target"
        s, e = q["temporal_target"]
        if e - s < 5.0:
            return "temporal_target 宽度 < 5s"
    if q.get("task_type") in ("open", "summary"):
        ref = q.get("reference_answer") or {}
        zh, en = ref.get("zh") or "", ref.get("en") or ""
        if zh_len(zh) < 80 and en_len(en) < 220:
            return f"参考答案过短(中文需≥80字/英文≥220词,当前 zh={zh_len(zh)} en={en_len(en)})"
        from .utils import count_time_anchors
        if count_time_anchors(zh) + count_time_anchors(en) < 2 and '"' not in (zh + en):
            return "参考答案缺 ≥2 个视频内具体细节(时间锚点或字幕引用)"
    return ""


def _normalize(q: Dict, video_id: str, idx: int, video_lang: str) -> Dict:
    q["qid"] = f"{video_id}_q{idx:02d}"
    q.setdefault("provenance", "llm_assisted")
    q.setdefault("split", "test")
    q.setdefault("video_lang", video_lang)
    q.setdefault("evidence_modality", [])
    q.setdefault("difficulty", "medium")
    q.setdefault("min_watch", "single_scene")
    q.setdefault("lang_mode", "parallel")
    caps = [c for c in q.get("capability", []) if c in _VALID_CAP]
    q["capability"] = caps or ["L3_causal"]
    if q.get("task_type") == "mcq":
        for o in q.get("options", []):
            o.setdefault("distractor_trap", "none")
    q.setdefault("quality_flags", [])
    return q


def _referring_check(q: Dict) -> str:
    """题干 referring 校验(Res 5.1):时间/序数/锁定短语;返回 '' 或缺失说明。"""
    zh = q.get("question", {}).get("zh") or ""
    en = q.get("question", {}).get("en") or ""
    if zh_len(zh) < 15 and en_len(en) < 10:
        return "题干过短"
    if not (has_referring_ctx(zh) or has_referring_ctx(en)):
        return "题干缺指代上下文(时间描述/序数消歧/锁定短语)"
    return ""


def _rewrite_with_feedback(client: LLMClient, q: Dict, feedback: str) -> Optional[Dict]:
    """带反馈重写(Res 5.1:≤2 次;失败剔除并计 reused_anchor)。"""
    try:
        out = client.complete(
            [{"role": "system", "content": prompts.QA_SYS},
             {"role": "user", "content": prompts.qa_rewrite_user(q, feedback)}],
            vision=False)
        raw = out.get("qa") or [] if isinstance(out, dict) else []
        return raw[0] if raw else None
    except Exception as e:  # noqa: BLE001
        log.warning("题目重写失败: %s", e)
        return None


def _fix_question(client: LLMClient, cfg: RunConfig, q: Dict, video_id: str,
                  video_lang: str, anchors_by_id: Optional[Dict[str, Dict]] = None,
                  duration: float = 0.0) -> Tuple[Optional[Dict], bool]:
    """校验 + 重写循环(≤2 次重写);返回 (题目, 是否被丢弃)。"""
    for attempt in range(3):
        reason = _valid(q, anchors_by_id, duration)
        if not reason:
            ref = _referring_check(q)
            if not ref:
                return _normalize(q, video_id, 0, video_lang), False
            reason = ref
        if attempt >= 2:
            return None, True
        anchor_id = q.get("anchor_id")
        q = _rewrite_with_feedback(client, q, f"不满足:{reason}")
        if q is None:
            return None, True
        if anchor_id and not q.get("anchor_id"):
            q["anchor_id"] = anchor_id
    return None, True


# =========================================================================
# 覆盖矩阵(Res 5.2)
# =========================================================================
def _bucket_of(t: float, duration: float) -> int:
    if duration <= 0:
        return 0
    return min(3, int(t / duration * 4))


def refresh_qa_coverage(qa: List[Dict], duration: float,
                        cfg: RunConfig) -> Dict:
    """能力矩阵报告:层覆盖 / min_watch / task_type / 时长桶;返回 pass 布尔。"""
    layers = {f"L{i}": 0 for i in range(1, 6)}
    cross = 0
    task_type = {"mcq": 0, "temporal_grounding": 0, "open": 0, "summary": 0}
    buckets = {i: 0 for i in range(4)}
    for q in qa:
        for c in q.get("capability", []):
            layers[c[:2]] = layers.get(c[:2], 0) + 1
        if q.get("min_watch") in ("cross_scene", "whole_video"):
            cross += 1
        tt = q.get("task_type")
        if tt in task_type:
            task_type[tt] += 1
        for sp in q.get("evidence_spans", []):
            buckets[_bucket_of(sp[0], duration)] += 1
    n = len(qa)
    total_bucket_marks = sum(buckets.values())
    ok_layers = all(v >= cfg.min_qa_per_capability_layer for v in layers.values())
    ok_cross = (cross / n >= cfg.cross_scene_min_ratio) if n else False
    ok_mcq = cfg.mcq_min_ratio <= (task_type["mcq"] / n) <= cfg.mcq_max_ratio if n else False
    ok_temporal = (cfg.temporal_min_ratio <= (task_type["temporal_grounding"] / n)
                   <= cfg.temporal_max_ratio) if n else False
    ok_buckets = all((b / max(1, total_bucket_marks)) >= 0.15 for b in buckets.values()) \
        if total_bucket_marks else False
    return {
        "capability_layer": layers, "min_watch_cross": cross, "n": n,
        "task_type": task_type, "duration_bucket": buckets,
        "checks": {"layers": ok_layers, "cross_scene": ok_cross, "task_type_mcq": ok_mcq,
                   "task_type_temporal": ok_temporal, "duration_bucket": ok_buckets},
        "pass": bool(n and ok_layers and ok_cross and ok_mcq and ok_temporal and ok_buckets),
    }


# =========================================================================
# 生成
# =========================================================================
def _question_key(q: Dict) -> str:
    """题目去重键:题干(去标点空白)+ 答案。用于跨批次查重(B11)。"""
    zh = re.sub(r"[\s\W_]+", "", (q.get("question") or {}).get("zh") or "")
    en = re.sub(r"[\s\W_]+", "", ((q.get("question") or {}).get("en") or "")).lower()
    return f"{zh or en}|{q.get('answer') or ''}"


def generate_qa(client: LLMClient, cfg: RunConfig, scenes: List[Dict],
                events: List[Dict], global_block: Dict, meta: Dict,
                structure: Optional[Dict] = None, duration: float = 0.0,
                existing_qa: Optional[List[Dict]] = None) -> Tuple[List[Dict], Dict]:
    """锚定出题主入口:按时长配额分批出题,能力矩阵缺口注入提示词。

    返回 (qa, coverage_report)。
    """
    video_id = meta["video_id"]
    video_lang = meta["meta"]["primary_lang"]
    duration = duration or float(meta["meta"].get("duration_sec", 0) or 0)
    quota = cfg.qa_quota(duration)
    structure = structure or {}
    anchors = build_anchor_pool(scenes, events, structure)
    anchors_by_id = {a["anchor_id"]: a for a in anchors}

    qa: List[Dict] = list(existing_qa or [])
    # 跨批次去重(B11):此前每轮把完整锚点池重新喂一遍,既不标记已用锚点也不查重,
    # 2h 视频要出 3 轮,极易产生重复或近重复题。
    seen_keys = {_question_key(q) for q in qa}
    used_anchors = {q.get("anchor_id") for q in qa if q.get("anchor_id")}
    reused_anchor = 0
    duplicates = 0
    rounds = 0
    max_rounds = 2 + quota // cfg.qa_batch_size + (1 if quota % cfg.qa_batch_size else 0)

    while len(qa) < quota and rounds < max_rounds:
        rounds += 1
        need = min(cfg.qa_batch_size, quota - len(qa))
        coverage = refresh_qa_coverage(qa, duration, cfg)
        matrix_req = {
            "capability_deficit": {f"L{i}": max(0, cfg.min_qa_per_capability_layer - coverage["capability_layer"][f"L{i}"])
                                   for i in range(1, 6)},
            "task_type_deficit": {"mcq": max(0, int(quota * cfg.mcq_min_ratio) - coverage["task_type"]["mcq"]),
                                  "temporal_grounding": max(0, int(quota * cfg.temporal_min_ratio) - coverage["task_type"]["temporal_grounding"])},
            "cross_scene_min_ratio": cfg.cross_scene_min_ratio,
            "used_anchor_ids": sorted(a for a in used_anchors if a),
        }
        # 未用过的锚点优先排前面,让模型自然覆盖到新证据
        fresh = [a for a in anchors if a["anchor_id"] not in used_anchors]
        pool = (fresh + [a for a in anchors if a["anchor_id"] in used_anchors]) if fresh else anchors
        try:
            out = client.complete(
                [{"role": "system", "content": prompts.QA_SYS},
                 {"role": "user", "content": prompts.qa_user(pool, global_block, meta,
                                                             need, matrix_req)}],
                vision=False)
        except Exception as e:  # noqa: BLE001
            log.error("出题调用失败(第 %d 轮): %s", rounds, e)
            break
        raw = out.get("qa") or [] if isinstance(out, dict) else []
        added = 0
        for q in raw:
            if not isinstance(q, dict) or not q.get("anchor_id"):
                continue
            fixed, dropped = _fix_question(client, cfg, q, video_id, video_lang,
                                           anchors_by_id, duration)
            if dropped:
                reused_anchor += 1
                continue
            key = _question_key(fixed)
            if key in seen_keys:
                duplicates += 1
                log.debug("跳过重复题:%s", key[:40])
                continue
            seen_keys.add(key)
            used_anchors.add(fixed.get("anchor_id"))
            fixed["qid"] = f"{video_id}_q{len(qa) + 1:02d}"
            qa.append(fixed)
            added += 1
        log.info("出题第 %d 轮:%d 道原始 -> 新增 %d 道(重复 %d)-> 累计 %d/%d",
                 rounds, len(raw), added, duplicates, len(qa), quota)
        if not added and not raw:
            log.warning("本轮未产出任何题目,提前结束出题")
            break

    coverage = refresh_qa_coverage(qa, duration, cfg)
    coverage["reused_anchor"] = reused_anchor
    coverage["duplicates_dropped"] = duplicates
    coverage["anchors_used"] = len(used_anchors)
    coverage["anchors_total"] = len(anchors)
    coverage["target"] = quota
    if not coverage["pass"]:
        log.warning("能力矩阵未达标:%s(发布门禁将拦截)", coverage["checks"])
    log.info("出题完成:%d/%d 道;用到 %d/%d 个锚点;矩阵通过=%s",
             len(qa), quota, len(used_anchors), len(anchors), coverage["pass"])
    return qa, coverage


def qa_quota(cfg: RunConfig, duration: float) -> int:
    """兼容助手:按时长配额(Res 5.3)。"""
    return cfg.qa_quota(duration)
