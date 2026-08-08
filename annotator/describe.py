# -*- coding: utf-8 -*-
"""全片详细描述:逐窗生成【详细描述】(而非概要)+ 汇总成全片详述。

产出:
- structure.segments[]        : [{start, end, description:{zh,en}}] 带时间戳的密集描述轨
- global.detailed_description  : {zh, en} 全片连贯详述

复用与场景标注相同的抽帧+字幕+OCR;段数多时两级汇总,支撑 30min–2h。
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple

from . import native_video, prompts
from .config import RunConfig
from .llm_client import LLMClient, extract_json, image_part, text_part
from .media import (build_windows, fixed_windows, ocr_text, sample_frames,
                    subs_text)

log = logging.getLogger("annotator.describe")


def _describe_window_frames(client: LLMClient, cfg: RunConfig, video_path: str,
                            ws: float, we: float, sub: str, ocr: str, meta: Dict) -> Optional[Dict]:
    k = cfg.describe_frames_per_window or cfg.frames_per_window
    frames = sample_frames(video_path, ws, we, k, cfg.max_image_edge)
    if not frames:
        return None
    content = [text_part(prompts.describe_user(ws, we, sub, ocr, meta))]
    content += [image_part(b64) for _, b64 in frames]
    return client.complete(
        [{"role": "system", "content": prompts.DESCRIBE_SYS},
         {"role": "user", "content": content}],
        vision=True,
    )


def _describe_window_native(client: LLMClient, cfg: RunConfig, video_path: str,
                            ws: float, we: float, sub: str, ocr: str, meta: Dict) -> Optional[Dict]:
    api_key = os.environ.get(client.pc.api_key_env)
    clipdir = cfg.native_clip_dir or tempfile.gettempdir()
    os.makedirs(clipdir, exist_ok=True)
    clip = os.path.join(clipdir, f"clip_{meta.get('video_id','v')}_{int(ws)}_{int(we)}.mp4")
    native_video.cut_clip(video_path, ws, we, clip, cfg.max_image_edge)
    try:
        user = prompts.describe_user(ws, we, sub, ocr, meta)
        text = native_video.describe_clip(clip, prompts.DESCRIBE_SYS, user,
                                           client.pc.vision_model, api_key, cfg.native_fps)
        return extract_json(text)
    finally:
        if not cfg.native_clip_dir:      # 临时片段用完即删
            try:
                os.remove(clip)
            except OSError:
                pass


def describe_segments(client: LLMClient, cfg: RunConfig, video_path: str,
                      structure: Dict, meta: Dict) -> List[Dict]:
    shots = structure.get("shots", [])
    if shots:
        windows = build_windows(shots, cfg.window_sec)
    else:
        windows = fixed_windows(float(meta["meta"]["duration_sec"]), cfg.window_sec)
    if cfg.max_windows > 0:
        windows = windows[: cfg.max_windows]

    native = cfg.describe_mode == "native_video"
    if native:
        if client.pc.name != "qwen":
            raise RuntimeError("native_video 模式仅支持 provider=qwen;Kimi 无原生视频输入,请用 --describe-mode frames")
        native_video.ensure_available()
    log.info("详述模式: %s;共 %d 窗", cfg.describe_mode, len(windows))

    segments: List[Dict] = []
    for wi, (ws, we) in enumerate(windows, 1):
        sub = subs_text(structure.get("subtitles", []), ws, we)
        ocr = ocr_text(structure.get("ocr", []), ws, we)
        try:
            out = (_describe_window_native if native else _describe_window_frames)(
                client, cfg, video_path, ws, we, sub, ocr, meta)
        except Exception as e:  # noqa: BLE001
            log.error("详述窗口 %d [%.0f-%.0fs] 失败,跳过: %s", wi, ws, we, e)
            continue
        if not out:
            log.warning("详述窗口 %d 无产出,跳过", wi)
            continue
        desc = out.get("description") or {"zh": None, "en": None}
        segments.append({"start": ws, "end": we, "description": desc})
        log.info("详述窗口 %d/%d 完成", wi, len(windows))
    return segments


def _agg_once(client: LLMClient, items: List[Dict], meta: Dict) -> Dict:
    out = client.complete(
        [{"role": "system", "content": prompts.DESCRIBE_AGG_SYS},
         {"role": "user", "content": prompts.describe_agg_user(items, meta)}],
        vision=False,
    )
    return out.get("detailed_description") or {"zh": None, "en": None}


def aggregate_description(client: LLMClient, cfg: RunConfig,
                          segments: List[Dict], meta: Dict) -> Dict:
    if not segments:
        return {"zh": None, "en": None}
    items = [{"start": s["start"], "end": s["end"],
              "text": (s["description"].get("en") or s["description"].get("zh") or "")}
             for s in segments]

    group = cfg.describe_agg_group or 15
    if len(items) <= group:
        return _agg_once(client, items, meta)

    # 两级汇总:分组 -> 组内详述 -> 合并
    parts: List[Dict] = []
    for i in range(0, len(items), group):
        chunk = items[i:i + group]
        d = _agg_once(client, chunk, meta)
        parts.append({"start": chunk[0]["start"], "end": chunk[-1]["end"],
                      "text": (d.get("en") or d.get("zh") or "")})
    log.info("两级汇总:%d 段 -> %d 组 -> 合并", len(items), len(parts))
    return _agg_once(client, parts, meta)


def describe_video(client: LLMClient, cfg: RunConfig, video_path: str,
                   structure: Dict, meta: Dict) -> Tuple[List[Dict], Dict]:
    segments = describe_segments(client, cfg, video_path, structure, meta)
    detailed = aggregate_description(client, cfg, segments, meta)
    log.info("详述完成:%d 段 + 全片汇总", len(segments))
    return segments, detailed
