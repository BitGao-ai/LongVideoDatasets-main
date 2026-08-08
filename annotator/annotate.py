# -*- coding: utf-8 -*-
"""标注阶段:逐窗口场景/事件标注 + 全局聚合(梗概/人物/关系/时间线/因果)。"""

import logging
from typing import Dict, List, Tuple

from . import prompts
from .config import RunConfig
from .llm_client import LLMClient, image_part, text_part
from .media import (build_windows, fixed_windows, ocr_text, sample_frames,
                    subs_text)

log = logging.getLogger("annotator.annotate")


def annotate_structure(client: LLMClient, cfg: RunConfig, video_path: str,
                       structure: Dict, meta: Dict) -> Tuple[List[Dict], List[Dict]]:
    """逐窗口调用多模态模型,产出 scenes 与 events(带全局唯一 id)。"""
    shots = structure.get("shots", [])
    if shots:
        windows = build_windows(shots, cfg.window_sec)
    else:
        dur = float(meta["meta"]["duration_sec"])
        windows = fixed_windows(dur, cfg.window_sec)
        log.warning("无镜头信息,回退固定步长切窗:%d 个", len(windows))

    if cfg.max_windows > 0:
        windows = windows[: cfg.max_windows]
    log.info("共 %d 个时间窗待标注", len(windows))

    scenes: List[Dict] = []
    events: List[Dict] = []
    sid = 0
    eid = 0

    for wi, (ws, we) in enumerate(windows, 1):
        frames = sample_frames(video_path, ws, we, cfg.frames_per_window, cfg.max_image_edge)
        if not frames:
            log.warning("窗口 %d [%.0f-%.0fs] 抽帧为空,跳过", wi, ws, we)
            continue
        sub = subs_text(structure.get("subtitles", []), ws, we)
        ocr = ocr_text(structure.get("ocr", []), ws, we)

        content = [text_part(prompts.scene_user(ws, we, sub, ocr, meta))]
        content += [image_part(b64) for _, b64 in frames]

        try:
            out = client.complete(
                [{"role": "system", "content": prompts.SCENE_SYS},
                 {"role": "user", "content": content}],
                vision=True,
            )
        except Exception as e:  # noqa: BLE001
            log.error("窗口 %d 标注失败,跳过: %s", wi, e)
            continue

        for sc in out.get("scenes", []) or []:
            sid += 1
            sc["id"] = sid
            sc.setdefault("shot_ids", [])
            scenes.append(sc)
        for ev in out.get("events", []) or []:
            eid += 1
            ev["id"] = f"e{eid}"
            ev.setdefault("causes", [])
            # 将事件挂到覆盖它的场景上(取中点落入的场景)
            mid = (ev["span"][0] + ev["span"][1]) / 2.0 if ev.get("span") else ws
            ev["scene_id"] = next((s["id"] for s in scenes
                                   if s["start"] <= mid <= s["end"]), None)
            events.append(ev)
        log.info("窗口 %d/%d 完成:累计 %d 场景, %d 事件", wi, len(windows), len(scenes), len(events))

    return scenes, events


def annotate_global(client: LLMClient, cfg: RunConfig, scenes: List[Dict],
                    events: List[Dict], meta: Dict) -> Tuple[Dict, List[Dict]]:
    """纯文本聚合出全局块,并把因果边回填到 events。"""
    if not scenes and not events:
        return {"synopsis": {"zh": None, "en": None}, "characters": [],
                "relations": [], "timeline": []}, events

    out = client.complete(
        [{"role": "system", "content": prompts.GLOBAL_SYS},
         {"role": "user", "content": prompts.global_user(scenes, events, meta)}],
        vision=False,
    )

    # 回填因果边
    cause_map = {c["event"]: c.get("causes", []) for c in out.get("causal", []) or []}
    valid_ids = {e["id"] for e in events}
    for ev in events:
        if ev["id"] in cause_map:
            ev["causes"] = [c for c in cause_map[ev["id"]] if c in valid_ids]

    timeline = [t for t in out.get("timeline", []) if t in valid_ids] or [e["id"] for e in events]
    global_block = {
        "synopsis": out.get("synopsis", {"zh": None, "en": None}),
        "characters": out.get("characters", []),
        "relations": out.get("relations", []),
        "timeline": timeline,
    }
    log.info("全局聚合完成:%d 人物, %d 关系, timeline %d 事件",
             len(global_block["characters"]), len(global_block["relations"]), len(timeline))
    return global_block, events
