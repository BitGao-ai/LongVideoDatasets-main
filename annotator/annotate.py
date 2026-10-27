# -*- coding: utf-8 -*-
"""标注阶段 v2:逐窗口场景/事件标注(失败→指数退避重试→gaps 上报,Res 3.6)
+ 全局聚合(委托 aggregate.py M3)。

- annotate_structure: 每窗失败重试 ≤max_window_retries 次;仍失败写
  annotation_meta.gaps[]={start,end,reason};阶段结束做覆盖校验
  union(windows ∪ gaps) 覆盖 [0,duration],未覆盖显式暴露(不静默)。
- 支持 StageCache 窗口级断点续跑(输入哈希不变直接复用)。
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from . import prompts
from .aggregate import aggregate_global
from .config import RunConfig
from .llm_client import LLMClient, image_part, text_part
from .media import build_windows, fixed_windows, ocr_text, sample_frames, subs_text
from .utils import coverage_gaps, sha256_of

log = logging.getLogger("annotator.annotate")


def annotate_structure(client: LLMClient, cfg: RunConfig, video_path: str,
                       structure: Dict, meta: Dict,
                       store=None, cache=None, duration: float = 0.0,
                       gaps: List[Dict] = None) -> Tuple[List[Dict], List[Dict]]:
    """逐窗口调用多模态模型,产出 scenes 与 events(带全局唯一 id)。

    窗口失败:重试 ≤max_window_retries(指数退避)→ 仍失败写 gaps[](Res 3.6);
    已失败的窗口用 FrameStore 兜底补抽帧重试,缓存命中直接复用。
    """
    shots = structure.get("shots", [])
    if shots:
        windows = build_windows(shots, cfg.window_sec)
    else:
        dur = duration or float(meta["meta"]["duration_sec"])
        windows = fixed_windows(dur, cfg.window_sec)
        log.warning("无镜头信息,回退固定步长切窗:%d 个", len(windows))

    if cfg.max_windows > 0:
        windows = windows[: cfg.max_windows]
    log.info("共 %d 个时间窗待标注", len(windows))

    # 调用方通常在 retry_exp_backoff 外层持有同一个 gaps 列表,重试时必须先清空,
    # 否则同一个失败窗口会被重复写入 annotation_meta.gaps
    if gaps is None:
        gaps = []
    else:
        gaps.clear()
    scenes: List[Dict] = []
    events: List[Dict] = []
    sid = 0
    eid = 0

    # ---- 预处理循环:逐窗做缓存查取,未命中的窗进并发池(Res 修复 C/S1)----
    jobs: List[Tuple[int, float, float, str, str]] = []
    results: Dict[int, Optional[Dict]] = {}
    for wi, (ws, we) in enumerate(windows, 1):
        sub = subs_text(structure.get("subtitles", []), ws, we)
        ocr = ocr_text(structure.get("ocr", []), ws, we)
        key = {"stage": "window_annotate", "span": (ws, we), "subs": sub, "ocr": ocr}

        cached = cache.get("structure", key) if cache else None
        if cached:
            results[wi] = cached
            log.info("窗口 %d/%d 缓存复用", wi, len(windows))
        else:
            jobs.append((wi, ws, we, sub, ocr))
            results[wi] = None

    # ---- S1 窗口级并发:此前是纯串行,而 S1 是整条流水线调用次数最多的阶段 ----
    # 每个 worker 内部独立重试(指数退避);pool.map 按提交序返回,保证下面的
    # 编号(scene id / event id)与缓存落盘次序和串行完全一致 —— 产物不允许
    # 随完成顺序漂移。
    def _annotate_window(job: Tuple[int, float, float, str, str]) -> Optional[Dict]:
        wi, ws, we, sub, ocr = job
        frames_reuse: List[Tuple[float, str]] = []   # P2-c:重试间复用已抽帧
        for attempt in range(cfg.max_window_retries):
            frames = frames_reuse or sample_frames(video_path, ws, we,
                                                   cfg.frames_per_window,
                                                   cfg.max_image_edge, store=store)
            if not frames:
                # 抽帧为空才重新抽;其余失败帧复用 —— 相同输入下重试的价值在 LLM 端
                frames_reuse = []
                log.warning("窗口 %d [%.0f-%.0fs] 抽帧为空,重试 %d/%d",
                            wi, ws, we, attempt + 1, cfg.max_window_retries)
                time.sleep(min(2 ** attempt, 30))
                continue
            frames_reuse = frames
            content = [text_part(prompts.scene_user(ws, we, sub, ocr, meta))]
            content += [image_part(b64) for _, b64 in frames]
            try:
                out = client.complete(
                    [{"role": "system", "content": prompts.SCENE_SYS},
                     {"role": "user", "content": content}],
                    vision=True,
                )
                if isinstance(out, dict) and (out.get("scenes") or out.get("events")):
                    return out
                log.warning("窗口 %d 输出为空(第 %d/%d 次),重试", wi, attempt + 1,
                            cfg.max_window_retries)
            except Exception as e:  # noqa: BLE001
                log.error("窗口 %d 标注失败(第 %d/%d 次): %s", wi, attempt + 1,
                          cfg.max_window_retries, e)
            time.sleep(min(2 ** attempt, 30))
        return None

    if jobs:
        uncached_ids = {j[0] for j in jobs}
        workers = max(1, min(cfg.workers, len(jobs)))
        if workers > 1:
            log.info("S1 窗口标注并发:%d 窗 / %d 线程", len(jobs), workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            concurrent = list(pool.map(_annotate_window, jobs))
        for (wi, _ws, _we, _sub, _ocr), out in zip(jobs, concurrent):
            results[wi] = out

    # ---- 汇编循环:按窗口索引序统一编号 / 记 gap / 落缓存(确定性核心)----
    for wi, (ws, we) in enumerate(windows, 1):
        sub = subs_text(structure.get("subtitles", []), ws, we)
        ocr = ocr_text(structure.get("ocr", []), ws, we)
        key = {"stage": "window_annotate", "span": (ws, we), "subs": sub, "ocr": ocr}
        result = results.get(wi)
        if result is None:
            gaps.append({"start": ws, "end": we,
                         "reason": f"窗口标注重试 {cfg.max_window_retries} 次仍失败"})
            log.error("窗口 %d [%.0f-%.0fs] 已写入 gaps,后续出题/验收会拦截", wi, ws, we)
            continue

        for sc in result.get("scenes", []) or []:
            sid += 1
            sc["id"] = sid
            sc.setdefault("shot_ids", [])
            scenes.append(sc)
        for ev in result.get("events", []) or []:
            eid += 1
            ev["id"] = f"e{eid}"
            ev.setdefault("causes", [])
            events.append(ev)
        if cache and wi in uncached_ids:
            cache.put("structure", key, {"scenes": result.get("scenes", []),
                                         "events": result.get("events", [])})
        log.info("窗口 %d/%d 完成:累计 %d 场景, %d 事件",
                 wi, len(windows), len(scenes), len(events))

    # 覆盖校验(Res 3.6):union(windows) 应覆盖 [0,duration];失败窗口另行暴露
    dur = duration or float(meta["meta"]["duration_sec"])
    holes, failed = _check_coverage(windows, gaps, dur)
    for h in holes:
        gaps.append({"start": h["start"], "end": h["end"],
                     "reason": "无任何窗口覆盖该区间(镜头信息缺失?)"})
    return scenes, events


def _check_coverage(windows: List[Tuple[float, float]], gaps: List[Dict],
                    duration: float) -> Tuple[List[Dict], List[Dict]]:
    """覆盖校验(修复 A4)。

    此前把 gaps 当 windows 传给 coverage_gaps,语义完全颠倒:0 个 gap 会报
    "全片未覆盖",全部窗口失败反而报 "覆盖通过"。

    返回 (无窗口覆盖的空洞, 有窗口但标注失败的区间)。两者都非空才算覆盖合格。
    """
    holes = coverage_gaps(list(windows), duration)
    failed = [dict(g) for g in (gaps or [])]
    if holes:
        log.warning("覆盖校验:%d 个区间没有任何窗口覆盖:%s", len(holes), holes[:5])
    if failed:
        log.error("覆盖校验:%d 个窗口标注失败,已写入 gaps(发布门禁要求 0 gaps):%s",
                  len(failed), [(g["start"], g["end"]) for g in failed[:5]])
    if not holes and not failed:
        log.info("覆盖校验通过:union(窗口) 覆盖 [0, %.1fs] 且无失败窗口", duration)
    return holes, failed


def annotate_global(client: LLMClient, cfg: RunConfig, scenes: List[Dict],
                    events: List[Dict], meta: Dict) -> Tuple[Dict, List[Dict]]:
    """兼容入口:委托 M3 aggregate_global(返回 global_block, events)。"""
    block, events, report = aggregate_global(client, cfg, scenes, events, meta)
    if report.get("causal_invalid"):
        log.warning("%d 条非法因果边已丢弃(详见 annotation_meta.human_todo)", len(report["causal_invalid"]))
    return block, events
