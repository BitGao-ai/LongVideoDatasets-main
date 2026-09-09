# -*- coding: utf-8 -*-
"""事件引擎 M2(Res 3)—— 核心需求 3:关键事件 100% 定位。

流水线:事件挖掘(滑窗文本轮廓,不再受 180s 窗口硬切)
       → 8×8 帧马赛克边界复核(Verified/Refined/Split/Rejected,≤2 轮)
       → 跨窗/跨摘要去重(三要素对齐 + 相似度 ≥0.85 + 时间重叠 >50%)
       → 关键性分级(key/minor + 依据)
       → span 合法性状态机(越界/倒置/重叠报警,非法置 rejected 进人工清单)

统计:refined_ratio / rejected_ratio / event_density 进 manifest.QC(Res 3.2/3.4)。
"""

import copy
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from . import prompts
from .config import RunConfig
from .frames_store import FrameStore
from .llm_client import LLMClient, image_part, text_part
from .utils import (asym_pad, clamp, interval_overlap_ratio, ngram_similarity,
                    text_similarity)

log = logging.getLogger("annotator.events")

_MIN_EVENT_SEC = 2.0
_MODALITIES = ("visual", "audio", "speech")


# =========================================================================
# 1) 事件挖掘(Res 3.1)
# =========================================================================
def _merge_caption_segments(captions: List[Dict], min_sim: float = 0.30,
                            max_gap: float = 25.0) -> List[Dict]:
    """把 1fps 帧描述流按“相邻主题一致”合并为候选段(纯规则草稿)。

    相邻帧描述 n-gram 相似度 ≥ min_sim 且间隔 ≤ max_gap 视为同一候选事件。
    """
    if not captions:
        return []
    segs: List[Dict] = []
    cur = {"start": captions[0]["t"], "end": captions[0]["t"],
           "texts": [captions[0]["text"]]}
    for i, c in enumerate(captions[1:], start=1):
        prev = captions[i - 1]
        sim = ngram_similarity(cur["texts"][-1], c["text"], n=3)
        if sim >= min_sim and (c["t"] - prev["t"]) <= max_gap:
            cur["end"] = c["t"]
            cur["texts"].append(c["text"])
        else:
            segs.append(cur)
            cur = {"start": c["t"], "end": c["t"], "texts": [c["text"]]}
    segs.append(cur)
    for s in segs:
        s["desc_zh"] = "；".join(dict.fromkeys(s.pop("texts")))[:120]
    return [{"start": round(s["start"], 3), "end": round(s["end"], 3),
             "desc_zh": s["desc_zh"], "modality": ["visual"]}
            for s in segs if s["end"] - s["start"] >= _MIN_EVENT_SEC]


def _consolidate_window(client: LLMClient, cfg: RunConfig, ws: float, we: float,
                        draft: List[Dict], subs: str, ocr: str) -> Optional[List[Dict]]:
    """大模型汇总:把草稿候选在窗内合并/拆分/补全 desc 双语与类型(Res 3.1 两次提升)。"""
    briefs = [{"span": [d["start"], d["end"]], "desc": d.get("desc_zh"), "modality": d["modality"]}
              for d in draft]
    user = (
        f"时间窗 [{ws:.1f}s, {we:.1f}s] 内的候选事件草稿(帧描述轮廓):\n"
        f"{prompts._j(briefs)}\n窗内字幕:\n{subs or '(无)'}\n窗内 OCR:\n{ocr or '(无)'}\n\n"
        "请输出严格 JSON: {\"events\": [{\"span\": [float, float], "
        "\"desc\": {\"zh\": str, \"en\": str}, \"type\": str, "
        "\"modality\": [\"visual\"|\"audio\"|\"speech\"]}]}\n"
        "要求:1) 合并重复草稿、修正边界;2) 只保留推动叙事/因果的事件;"
        "3) 时间必须落在本窗内;4) 每事件给出中英双语 1 句描述与类型(如 dialogue/action/music)。"
    )
    out = client.complete([{"role": "system", "content": prompts.SCENE_SYS},
                           {"role": "user", "content": user}], vision=False)
    events = out.get("events") or [] if isinstance(out, dict) else []
    ok = []
    for ev in events:
        span = ev.get("span")
        if not span or len(span) != 2 or span[1] <= span[0]:
            continue
        ok.append({"span": [float(span[0]), float(span[1])],
                   "desc": ev.get("desc") or {"zh": "", "en": ""},
                   "type": ev.get("type") or "other",
                   "modality": [m for m in (ev.get("modality") or ["visual"])
                                if m in _MODALITIES] or ["visual"]})
    return ok


def _mine_one_window(client: LLMClient, cfg: RunConfig, ws: float, we: float,
                     captions: List[Dict], subtitles: List[Dict],
                     ocr: List[Dict]) -> List[Dict]:
    """单个滑窗的事件挖掘(可并发):草稿 → 大模型汇总;汇总失败回退草稿。"""
    in_caps = [c for c in captions if ws <= c["t"] <= we] if captions else []
    if not in_caps:
        # 无帧描述:用字幕句组作草稿
        draft = [{"start": s["start"], "end": s["end"], "desc_zh": s["text"],
                  "modality": ["speech"]} for s in subtitles
                 if ws <= s["start"] <= we and (s["end"] - s["start"]) >= 0.5]
    else:
        draft = _merge_caption_segments(in_caps)
    if not draft:
        return []
    subs_txt = "\n".join(f"[{s['start']:.0f}s] {s['text']}" for s in subtitles
                         if ws <= s["start"] <= we)
    ocr_txt = " | ".join(o["text"] for o in ocr if ws <= o["start"] <= we)
    try:
        cons = _consolidate_window(client, cfg, ws, we, draft, subs_txt, ocr_txt)
    except Exception as e:  # noqa: BLE001 —— 汇总失败用草稿兜底,不静默丢窗
        log.warning("窗 %.0f-%.0fs 汇总失败,回退草稿: %s", ws, we, e)
        cons = [{"span": [d["start"], d["end"]],
                 "desc": {"zh": d["desc_zh"], "en": d["desc_zh"]},
                 "type": "other", "modality": d["modality"]} for d in draft]
    for c in cons or []:
        c["_win"] = (ws, we)
    return list(cons or [])


def mine_event_candidates(captions: List[Dict], subtitles: List[Dict], ocr: List[Dict],
                          duration: float, cfg: RunConfig,
                          client: LLMClient) -> List[Dict]:
    """事件挖掘主入口:滑窗(60s,50% 步进叠加)文本事件轮廓。

    captions 为空时退化为字幕边界候选(以字幕句组为单位)。窗口之间互不依赖,
    按窗并发;结果仍按窗口时间顺序拼接,保证下游 id 分配可复现。
    """
    windows: List[Tuple[float, float]] = []
    step = cfg.event_sliding_window * 0.5
    t = 0.0
    while t < duration:
        windows.append((t, min(t + cfg.event_sliding_window, duration)))
        t += step
    if not windows:
        return []

    workers = max(1, min(cfg.workers, len(windows)))
    per_window: List[List[Dict]] = [[] for _ in windows]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_mine_one_window, client, cfg, ws, we, captions,
                            subtitles, ocr): i
                for i, (ws, we) in enumerate(windows)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                per_window[i] = fut.result()
            except Exception as e:  # noqa: BLE001 —— 单窗异常不拖垮整个挖掘阶段
                log.warning("窗 %d 事件挖掘异常,该窗无候选: %s", i + 1, e)
            done += 1
            if done % 10 == 0:
                log.info("事件挖掘进度:%d/%d 窗", done, len(windows))

    candidates = [c for chunk in per_window for c in chunk]
    log.info("事件挖掘完成:%d 个候选(滑窗 %d 个 / %d 并发)",
             len(candidates), len(windows), workers)
    return candidates


# =========================================================================
# 2) 边界复核(Res 3.2)—— 8×8 帧马赛克,定位可信度核心
# =========================================================================
def _refine_one(client: LLMClient, cfg: RunConfig, store: FrameStore,
                cand: Dict) -> Dict:
    s, e = cand["span"]
    # 非对称补边:候选 span 不再恒定落在马赛克正中,"照抄原 span"拿不到白送的 verified
    pad_lo, pad_hi = asym_pad([cand.get("id"), s, e], e - s, cfg.mosaic_pad_sec)
    packed = store.mosaic((s - pad_lo, e + pad_hi), k=cfg.mosaic_frames) if store else None
    if not packed:
        cand["boundary_state"] = "unreviewed"
        return cand
    mosaic, tile_times, side = packed
    # 模型只能在马赛克实际覆盖的时间范围内做判断,超出即幻觉
    lo, hi = (tile_times[0], tile_times[-1]) if tile_times else (s, e)

    def _span_in_view(sp) -> Optional[List[float]]:
        try:
            a, b = float(sp[0]), float(sp[1])
        except (TypeError, ValueError, IndexError):
            return None
        a, b = clamp(a, lo, hi), clamp(b, lo, hi)
        return [round(a, 3), round(b, 3)] if b - a >= _MIN_EVENT_SEC else None

    try:
        content = [text_part(prompts.boundary_user(cand, tile_times=tile_times, side=side))]
        content.append(image_part(mosaic))
        out = client.complete([{"role": "system", "content": prompts.BOUNDARY_SYS},
                               {"role": "user", "content": content}], vision=True)
        out = out if isinstance(out, dict) else {}
        decision = out.get("decision", "verified")
        cand["boundary_view"] = [round(lo, 3), round(hi, 3), len(tile_times)]
        if decision == "refined":
            new_span = _span_in_view(out.get("new_span") or [])
            if new_span:
                cand["span"] = new_span
                cand["boundary_state"] = "refined"
                cand["refine_reason"] = out.get("reason")
            else:
                cand["boundary_state"] = "verified"   # 修正值不可用,维持原 span
        elif decision == "split" and len(out.get("split_spans") or []) >= 2:
            parts = [_span_in_view(sp) for sp in out["split_spans"][:2]]
            if all(parts):
                cand["_split"] = parts
                cand["boundary_state"] = "split"
                cand["refine_reason"] = out.get("reason")
            else:
                cand["boundary_state"] = "verified"
        elif decision == "rejected":
            cand["boundary_state"] = "rejected_final"
            cand["refine_reason"] = out.get("reason")
        else:
            cand["boundary_state"] = "verified"
        return cand
    except Exception as e:  # noqa: BLE001 —— 复核失败不丢事件,标记未复核
        log.warning("边界复核失败(%s),保持 unreviewed: %s", cand.get("id"), e)
        cand["boundary_state"] = "unreviewed"
        return cand


def _flatten_split(c: Dict) -> List[Dict]:
    """把 boundary_state=split 且带 _split 的候选拆成两个独立事件。"""
    spans = c.pop("_split", None)
    if not spans:
        return [c]
    a, b = spans[0], spans[1]
    c2 = copy.deepcopy(c)          # 修复 D8:此前浅拷贝,两片共享同一个 desc 字典
    c["span"], c2["span"] = a, b
    c["boundary_state"] = c2["boundary_state"] = "split"
    base_id = str(c.get("id") or "c")
    c2["id"] = f"{base_id}b"
    c["id"] = f"{base_id}a"
    return [c, c2]


def _refine_pass(candidates: List[Dict], store: FrameStore, client: LLMClient,
                 cfg: RunConfig) -> Tuple[List[Dict], List[Dict]]:
    """一轮马赛克复核,结果按【输入顺序】返回 (保留, 被否决)。

    不能按完成顺序收集:span 起点相同的候选,先后次序会决定去重时谁被保留、
    以及后续 id 的分配,按线程完成序收集会让同一份输入在不同 --workers 下
    跑出不同的阶段哈希,进而白白重跑下游阶段。
    """
    if not candidates:
        return [], []
    results: List[Optional[Dict]] = [None] * len(candidates)
    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
        futures = {pool.submit(_refine_one, client, cfg, store, c): i
                   for i, c in enumerate(candidates)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001 —— 取帧/拼图失败不丢事件,标记未复核
                log.warning("边界复核异常(%s),保持 unreviewed: %s",
                            candidates[i].get("id"), e)
                candidates[i]["boundary_state"] = "unreviewed"
                results[i] = candidates[i]

    out, rejected = [], []
    for cand in results:
        for item in _flatten_split(cand):
            if item.get("boundary_state") == "rejected_final":
                rejected.append(item)
            else:
                out.append(item)
    return out, rejected


def refine_boundaries(candidates: List[Dict], store: FrameStore,
                      client: LLMClient, cfg: RunConfig
                      ) -> Tuple[List[Dict], List[Dict], Dict]:
    """逐候选 8×8 马赛克复核;Refined/Split 二次复核 ≤ boundary_max_rounds 轮(Res 3.2)。

    返回 (保留事件, 被否决事件, 统计)。修复 A5:decision=rejected 的候选是挖掘器
    误报,此前被 append 进结果继续流经去重/分级/落盘,现在单独剥离交人工复核。
    """
    out, rejected = _refine_pass(candidates, store, client, cfg)

    # Refined/Split 二次复核(修复 B10:此前第二轮不再 _flatten,split 结果被吞掉)
    rounds = 1
    while rounds < cfg.boundary_max_rounds:
        second = [c for c in out if c.get("boundary_state") in ("refined", "split")]
        if not second:
            break
        log.info("边界二次复核第 %d 轮:%d 个", rounds, len(second))
        # 按对象身份剔除,不能用 `c not in second` —— dict 是值比较,
        # 两个内容相同的事件会被一起误删
        pending = {id(c) for c in second}
        out = [c for c in out if id(c) not in pending]
        again, again_rejected = _refine_pass(second, store, client, cfg)
        out.extend(again)
        rejected.extend(again_rejected)
        rounds += 1

    out.sort(key=lambda c: (c["span"][0], c["span"][1]))
    n = {st: 0 for st in ("verified", "refined", "split", "unreviewed")}
    for c in out:
        st = c.get("boundary_state", "unreviewed")
        n[st] = n.get(st, 0) + 1
    total = max(1, len(out) + len(rejected))
    stats = {"refined_ratio": round((n["refined"] + n["split"]) / total, 3),
             "rejected_ratio": round(len(rejected) / total, 3)}
    log.info("边界复核:%d 候选 -> verified %d / refined %d / split %d / unreviewed %d / "
             "rejected %d;refined_ratio=%.2f rejected_ratio=%.2f",
             len(candidates), n["verified"], n["refined"], n["split"], n["unreviewed"],
             len(rejected), stats["refined_ratio"], stats["rejected_ratio"])
    return out, rejected, stats


# =========================================================================
# 3) 跨窗去重(Res 3.3)
# =========================================================================
def _overlap_clusters(events: List[Dict], threshold: float) -> List[List[Dict]]:
    """按时间重叠 >threshold 聚类(去重的候选集合)。"""
    ordered = sorted(events, key=lambda e: e["span"][0])
    clusters: List[List[Dict]] = []
    for ev in ordered:
        placed = False
        for cl in clusters:
            for m in cl:
                if interval_overlap_ratio(ev["span"], m["span"]) > threshold:
                    cl.append(ev)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            clusters.append([ev])
    return [c for c in clusters if len(c) > 1]


def _llm_dedup_group(client: LLMClient, group: List[Dict]) -> List[Dict]:
    """大模型裁决 ≥3 事件的去重组(Res 3.3 三要素对齐)。"""
    try:
        out = client.complete([{"role": "system", "content": prompts.DEDUP_SYS},
                               {"role": "user", "content": prompts.dedup_user(group)}],
                              vision=False)
        groups = (out or {}).get("dedup_groups") or []
        for g in groups:
            keep_id, merge_ids = g.get("keep"), g.get("merge") or []
            if keep_id not in {e["id"] for e in group}:
                continue
            keep = next(e for e in group if e["id"] == keep_id)
            for m in group:
                if m["id"] in merge_ids and m["id"] != keep_id:
                    keep.setdefault("dedup_source_ids", []).append(m["id"])
                    m["_merged_into"] = keep_id
            if keep.get("dedup_source_ids"):
                keep["desc"] = keep.get("desc") or {}
        return [e for e in group if not e.get("_merged_into")]
    except Exception as e:  # noqa: BLE001
        log.warning("LLM 去重失败,回退规则去重: %s", e)
        return group


def dedup_events(events: List[Dict], client: LLMClient, cfg: RunConfig) -> List[Dict]:
    """跨窗/跨摘要去重:时间重叠>50% 且(相似度≥0.85 或 LLM 三要素对齐)→ 合并。"""
    for i, ev in enumerate(events):
        ev["id"] = ev.get("id") or f"c{i + 1:04d}"
    clusters = _overlap_clusters(events, cfg.dedup_time_overlap)
    for cl in clusters:
        if len(cl) == 2:
            a, b = cl
            sim = text_similarity(client,
                                  a.get("desc", {}).get("zh") or a.get("desc", {}).get("en"),
                                  b.get("desc", {}).get("zh") or b.get("desc", {}).get("en"))
            if sim >= cfg.dedup_similarity:
                keep = a if len(str(a.get("desc", {}))) >= len(str(b.get("desc", {}))) else b
                drop = b if keep is a else a
                keep.setdefault("dedup_source_ids", []).append(drop["id"])
                drop["_merged_into"] = keep["id"]
        else:
            _llm_dedup_group(client, cl)

    kept = [e for e in events if not e.get("_merged_into")]
    for e in kept:
        e.pop("_merged_into", None)
        if e.get("dedup_source_ids"):
            e["dedup_id"] = e["id"]
            e["id"] = f"{e['id']}_m{len(e['dedup_source_ids'])}"
    n_dedup = len(events) - len(kept)
    if n_dedup:
        log.info("事件去重:合并 %d 个重复候选(剩 %d)", n_dedup, len(kept))
    else:
        log.info("事件去重:无重复候选")
    return kept


# =========================================================================
# 4) 关键性分级(Res 3.4)+ span 合法性(Res 3.5)
# =========================================================================
def grade_importance(events: List[Dict], client: LLMClient, cfg: RunConfig,
                     duration: float) -> List[Dict]:
    """关键性分级:key/minor + 1 句依据;规则约束由 _normalize 强制。"""
    for i in range(0, len(events), 40):
        chunk = events[i:i + 40]
        try:
            out = client.complete([{"role": "system", "content": prompts.IMPORTANCE_SYS},
                                   {"role": "user",
                                    "content": prompts.importance_user(chunk, duration)}],
                                  vision=False)
            grade_map = {g.get("event"): g for g in (out or {}).get("importance") or []
                         if g.get("event")}
            for ev in chunk:
                g = grade_map.get(ev["id"])
                if g:
                    ev["importance"] = g.get("level") if g.get("level") in ("key", "minor") else "minor"
                    ev["importance_reason"] = (g.get("reason") or "")
                else:
                    ev.setdefault("importance", "minor")
                    ev["importance_reason"] = "分级调用未返回,默认 minor"
        except Exception as e:  # noqa: BLE001
            log.warning("事件分级调用失败,默认 minor: %s", e)
            for ev in chunk:
                ev.setdefault("importance", "minor")
                ev["importance_reason"] = "分级调用失败,默认 minor"
    n_key = sum(1 for e in events if e.get("importance") == "key")
    log.info("关键性分级:%d key / %d minor(目标 key 15~40 个)", n_key, len(events) - n_key)
    return events


def normalize_events(events: List[Dict], duration: float,
                     cfg: RunConfig) -> Tuple[List[Dict], List[Dict]]:
    """span 合法性状态机(Res 3.5):clamp [0,dur];start<end;重叠报警。

    非法事件置 state=rejected 并进入 human_todo(不参与出题)。
    返回 (合法事件, 人工清单项)。
    """
    valid, todos = [], []
    ordered = []
    for ev in events:
        s, e = ev.get("span") or [0.0, 0.0]
        s, e = clamp(float(s), 0.0, duration), clamp(float(e), 0.0, duration)
        if s >= e:
            s, e = max(0.0, s - 1.0), min(duration, s + 1.0)
        if e <= s:
            todos.append({"kind": "event_span_invalid", "event_id": ev.get("id"),
                          "span": [s, e], "reason": "span 为空或越界后仍无效"})
            ev["span"] = [round(s, 3), round(e, 3)]
            ev["importance"] = "minor"
            ev["_rejected"] = True
            continue
        ev["span"] = [round(s, 3), round(e, 3)]
        # key 事件规则:span 非空、∈[0,duration]、宽度 ≥ event_min_width(Res 3.4)
        if ev.get("importance") == "key" and (e - s) < cfg.event_min_width:
            log.warning("key 事件 %s 宽度 %.1fs < %.1fs,降级 minor",
                        ev.get("id"), e - s, cfg.event_min_width)
            ev["importance"] = "minor"
            ev["importance_reason"] = (ev.get("importance_reason") or "") + "; 宽度不足自动降级"
        ordered.append(ev)
    # 相邻事件重叠 >80% 报警(不丢弃,只留痕)
    ordered.sort(key=lambda e: e["span"][0])
    for a, b in zip(ordered, ordered[1:]):
        if a is not b and interval_overlap_ratio(a["span"], b["span"]) > 0.8:
            log.warning("相邻事件重叠>80%%:%s[%s] 与 %s[%s]", a.get("id"),
                        a["span"], b.get("id"), b["span"])
    valid = [e for e in ordered if not e.get("_rejected")]
    if todos:
        log.warning("%d 个非法事件进入人工清单", len(todos))
    return valid, todos


# =========================================================================
# 5) 装配
# =========================================================================
def link_scenes(events: List[Dict], scenes: List[Dict]) -> List[Dict]:
    """scene_id 按事件中点落入场景判定(跨越窗口/场景边界的事件取中点)。"""
    # 修复 D1:scenes 来自 LLM 输出,缺 start/end 或类型不对时不应整阶段崩溃
    usable = []
    for s in scenes or []:
        try:
            usable.append((float(s["start"]), float(s["end"]), s.get("id")))
        except (KeyError, TypeError, ValueError):
            log.warning("场景缺少可用的 start/end,跳过 scene_id 归属:%s", str(s)[:120])
    for ev in events:
        mid = (ev["span"][0] + ev["span"][1]) / 2.0
        ev["scene_id"] = next((sid for ss, se, sid in usable if ss <= mid <= se), None)
    return events


def _as_span(obj: Dict, key_start: str = "start", key_end: str = "end"
             ) -> Optional[List[float]]:
    try:
        s, e = float(obj[key_start]), float(obj[key_end])
    except (KeyError, TypeError, ValueError):
        return None
    return [s, e] if e > s else None


def run_event_engine(client: LLMClient, cfg: RunConfig, structure: Dict,
                     meta: Dict, captions: List[Dict], store: FrameStore,
                     window_events: List[Dict], scenes: List[Dict],
                     duration: float) -> Tuple[List[Dict], Dict, List[Dict]]:
    """M2 主入口。window_events 为场景标注阶段的粗事件(并入候选池)。

    返回 (合法事件, 统计, 人工清单)。人工清单含被否决的误报事件与非法 span
    事件(修复 A5:此前二者都被静默丢弃)。
    """
    subtitles = structure.get("subtitles", [])
    ocr = structure.get("ocr", [])

    # 1) 挖掘
    mined = mine_event_candidates(captions, subtitles, ocr, duration, cfg, client)
    # 并入窗口粗事件(去重前统一打 id)
    for i, ev in enumerate(window_events, start=len(mined) + 1):
        ev = dict(ev)
        ev.setdefault("id", f"w{i:04d}")
        ev.setdefault("desc", {"zh": "", "en": ""})
        ev.setdefault("type", "other")
        ev.setdefault("modality", ["visual"])
        if not (isinstance(ev.get("span"), (list, tuple)) and len(ev["span"]) == 2):
            continue
        mined.append(ev)
    # 音频事件并入(speech 之外的 audio 事件)
    for i, ae in enumerate(structure.get("audio_events") or [], start=1):
        span = _as_span(ae)
        if span is None:
            continue
        mined.append({"id": f"a{i:04d}", "span": span,
                      "desc": ae.get("desc") or {"zh": "", "en": ""},
                      "type": ae.get("type", "music"), "modality": ["audio"]})

    if not mined:
        log.warning("事件挖掘无候选(检查帧描述/字幕)")
        return [], {"refined_ratio": 0.0, "rejected_ratio": 0.0, "event_density": 0.0,
                    "key_events": 0}, []

    # 2) 边界复核(rejected 剥离,不进数据集)
    refined, rejected, bstats = refine_boundaries(mined, store, client, cfg)
    # 3) 去重
    deduped = dedup_events(refined, client, cfg)
    # 4) 关键性分级
    graded = grade_importance(deduped, client, cfg, duration)
    # 5) span 合法性 + scene 归属
    valid, todos = normalize_events(graded, duration, cfg)
    valid = link_scenes(valid, scenes)

    for ev in rejected:
        todos.append({"kind": "event_rejected", "event_id": ev.get("id"),
                      "span": ev.get("span"),
                      "reason": "边界复核判定为挖掘器误报(rejected),已剔除待人工确认"})
    for ev in valid:
        if ev.get("boundary_state") == "split":
            todos.append({"kind": "event_split", "event_id": ev.get("id"),
                          "span": ev.get("span"), "reason": "边界复核拆分,建议人工确认"})

    hours = max(duration / 3600.0, 1e-6)
    stats = {
        "refined_ratio": bstats["refined_ratio"],
        "rejected_ratio": bstats["rejected_ratio"],
        "event_density": round(len(valid) / hours, 1),
        "key_events": sum(1 for e in valid if e.get("importance") == "key"),
        "events_rejected": len(rejected),
    }
    log.info("事件引擎完成:%d 个事件(density=%.1f/h, key=%d, 剔除误报 %d, 人工清单 %d)",
             len(valid), stats["event_density"], stats["key_events"], len(rejected),
             len(todos))
    return valid, stats, todos
