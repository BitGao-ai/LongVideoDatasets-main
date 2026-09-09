# -*- coding: utf-8 -*-
"""抗捷径过滤(兼容入口)—— 实际实现迁移至 anti_shortcut.py(M5,Res 6)。

★D1 修复:should_drop 现在会把 single_frame_pass 纳入剔除(README 声明一致),
  --keep-single-frame 可 opt-in 保留。五通道 + 非 MCQ 通道见 anti_shortcut.py。
"""

import logging
from typing import Dict, List, Optional, Tuple

from . import anti_shortcut
from .config import RunConfig
from .frames_store import FrameStore
from .llm_client import LLMClient

log = logging.getLogger("annotator.filter")


def check_one(client: LLMClient, cfg: RunConfig, video_path: str,
              q: Dict, structure: Dict, store: Optional[FrameStore] = None,
              synopsis: Optional[str] = None) -> Dict:
    """旧接口。务必传 store —— 否则单帧通道会退化成从第 0 帧顺序解码到目标帧,
    每题一次全解码(D13)。"""
    return anti_shortcut.check_one(client, cfg, video_path, q, structure,
                                   store=store, synopsis=synopsis)


def should_drop(cfg: RunConfig, q: Dict) -> bool:
    return anti_shortcut.should_drop(cfg, q)


def renumber(qa: List[Dict]) -> List[Dict]:
    return anti_shortcut.renumber(qa)


def run_filters(client: LLMClient, cfg: RunConfig, video_path: str,
                qa: List[Dict], structure: Dict,
                store: Optional[FrameStore] = None,
                synopsis: Optional[str] = None) -> List[Dict]:
    """旧接口:返回保留列表(不返回统计)。"""
    kept, _ = anti_shortcut.run_filters(client, cfg, video_path, qa, structure,
                                        store=store, synopsis=synopsis)
    return kept
