# -*- coding: utf-8 -*-
"""出题阶段:基于结构化标注 + 全局信息生成 QA,并做规范化。"""

import logging
from typing import Dict, List

from . import prompts
from .config import RunConfig
from .llm_client import LLMClient

log = logging.getLogger("annotator.qa")

_VALID_CAP = set(prompts.CAPABILITY_VOCAB)


def _normalize(q: Dict, video_id: str, idx: int, video_lang: str) -> Dict:
    q["qid"] = f"{video_id}_q{idx:02d}"
    q.setdefault("provenance", "llm_assisted")   # 机器生成,人工复核后可改 human
    q.setdefault("split", "test")
    q.setdefault("video_lang", video_lang)
    q.setdefault("evidence_modality", [])
    q.setdefault("difficulty", "medium")
    q.setdefault("min_watch", "single_scene")
    q.setdefault("lang_mode", "parallel")
    # 过滤非法能力标签
    caps = [c for c in q.get("capability", []) if c in _VALID_CAP]
    q["capability"] = caps or ["L3_causal"]
    # MCQ 选项补齐 distractor_trap / 规范 id
    if q.get("task_type") == "mcq":
        for o in q.get("options", []):
            o.setdefault("distractor_trap", "none")
    return q


def _valid(q: Dict) -> bool:
    """丢弃结构不完整的题。"""
    if not q.get("question") or not q.get("evidence_spans"):
        return False
    if q.get("task_type") == "mcq":
        opts = q.get("options", [])
        if len(opts) != 4 or q.get("answer") not in {"A", "B", "C", "D"}:
            return False
    if q.get("task_type") == "temporal_grounding" and not q.get("temporal_target"):
        return False
    return True


def generate_qa(client: LLMClient, cfg: RunConfig, scenes: List[Dict],
                events: List[Dict], global_block: Dict, meta: Dict) -> List[Dict]:
    video_id = meta["video_id"]
    video_lang = meta["meta"]["primary_lang"]

    out = client.complete(
        [{"role": "system", "content": prompts.QA_SYS},
         {"role": "user", "content": prompts.qa_user(
             scenes, events, global_block, meta, cfg.qa_per_video, cfg.cross_scene_ratio)}],
        vision=False,
    )
    raw = out.get("qa", []) if isinstance(out, dict) else (out or [])

    qa: List[Dict] = []
    for q in raw:
        if not _valid(q):
            log.warning("丢弃不合规题目: %s", str(q)[:120])
            continue
        qa.append(_normalize(q, video_id, len(qa) + 1, video_lang))

    log.info("出题完成:%d 道有效(请求 %d)", len(qa), cfg.qa_per_video)
    _log_coverage(qa)
    return qa


def _log_coverage(qa: List[Dict]) -> None:
    if not qa:
        return
    layers = {}
    cross = 0
    for q in qa:
        for c in q["capability"]:
            layers[c[:2]] = layers.get(c[:2], 0) + 1
        if q.get("min_watch") in ("cross_scene", "whole_video"):
            cross += 1
    log.info("能力层分布: %s;长程题占比 %.0f%%",
             dict(sorted(layers.items())), 100.0 * cross / len(qa))
