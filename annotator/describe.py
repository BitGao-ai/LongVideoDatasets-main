# -*- coding: utf-8 -*-
"""全片详细描述(兼容入口)—— 实际实现迁移至 dense_caption.py(M1,Res 2)。

保留旧接口 describe_video(client, cfg, video_path, structure, meta),
供 run_describe.py 等既有调用方使用;新调用方请直接使用
dense_caption.describe_video(..., store=..., cache=...)。
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

from .config import RunConfig
from .dense_caption import describe_video as _describe_video_v2
from .frames_store import FrameStore
from .llm_client import LLMClient

log = logging.getLogger("annotator.describe")


def describe_video(client: LLMClient, cfg: RunConfig, video_path: str,
                   structure: Dict, meta: Dict,
                   store: Optional[FrameStore] = None,
                   cache=None) -> Tuple[List[Dict], Dict, Dict]:
    """v2 实现;无帧索引时自动构建(兜底)。

    返回 (segments, detailed, report)。此前只返回前两项,导致 run_describe.py
    产出的文件缺 quiet_segments,QC 的 segment_gate 会假通过(D10)。
    """
    if store is None:
        frames_dir = cfg.frame_index_dir or os.path.join(
            os.path.dirname(str(structure.get("_path") or ".")) or ".", "frames")
        store = FrameStore(frames_dir, fps=cfg.frame_rate, max_edge=cfg.max_image_edge)
        dur = float(meta.get("meta", {}).get("duration_sec", 0) or 0)
        store.set_duration(dur)
        store.build(video_path, force=cfg.force_frame_index)
        store.verify_index(dur)
    segments, detailed, report = _describe_video_v2(
        client, cfg, video_path, structure, meta, store=store, cache=cache)
    if report.get("quiet_segments"):
        log.warning("quiet_segments 已上报:%d 段(见 annotation_meta)", len(report["quiet_segments"]))
    return segments, detailed, report
