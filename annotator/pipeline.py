# -*- coding: utf-8 -*-
"""端到端编排(兼容入口)—— 实际实现迁移至 orchestrator.py(Res 10.2)。

annotate_video() 签名不变;断点续跑 / 阶段状态机 / 窗口缓存 / LLM trace
等能力见 orchestrator.py。
"""

import logging
from typing import Dict, Optional

from .config import RunConfig
from .orchestrator import annotate_video as _orchestrate

log = logging.getLogger("annotator.pipeline")


def annotate_video(meta_path: str, structure_path: str, out_path: str,
                   cfg: RunConfig, schema_path: Optional[str] = None) -> Dict:
    """端到端编排:meta + structure -> LLM 标注 -> 出题 -> 抗捷径 -> schema v2 落盘。"""
    return _orchestrate(meta_path, structure_path, out_path, cfg, schema_path)
