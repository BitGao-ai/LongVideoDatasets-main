# -*- coding: utf-8 -*-
"""出题(兼容入口)—— 实际实现迁移至 qa_engine.py(M4,Res 5)。

v2 变化:
- 证据锚定出题(anchor pool),输入不再包含 global.synopsis(防梗概泄漏,Res 5.4);
- 题干 referring 校验 + 能力矩阵硬约束 + 按时长配额(Res 5.1/5.2/5.3);
- 非 MCQ 长度/细节规范(Res 5.5)。
"""

import logging
from typing import Dict, List, Optional, Tuple

from . import qa_engine
from .config import RunConfig
from .llm_client import LLMClient

log = logging.getLogger("annotator.qa")


def generate_qa(client: LLMClient, cfg: RunConfig, scenes: List[Dict],
                events: List[Dict], global_block: Dict, meta: Dict,
                structure: Optional[Dict] = None, duration: float = 0.0
                ) -> Tuple[List[Dict], Dict]:
    """旧接口:返回 (qa, coverage_report)。"""
    return qa_engine.generate_qa(client, cfg, scenes, events, global_block, meta,
                                 structure=structure, duration=duration)
