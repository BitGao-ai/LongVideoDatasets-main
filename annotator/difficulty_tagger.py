# -*- coding: utf-8 -*-
"""难度自动标定(Res 6.4)—— 后过滤重算 difficulty,覆盖模型生成值。

- easy  : 单场景且未被任何捷径命中;
- medium: cross_scene(跨场景/全片);
- hard  : 需要因果/计数/多时刻(通过能力标签与证据结构自动推导);
之后以抽测统计校准 hard 占比,校准结果写入 qa_coverage 报告。
"""

import logging
from typing import Dict, List

log = logging.getLogger("annotator.difficulty")

_HARD_CAPS = {"L2_counting", "L2_ordering", "L3_causal", "L3_counterfactual",
              "L3_prediction", "L5_needle", "L5_foreshadow"}


def _is_hard(q: Dict) -> bool:
    caps = set(q.get("capability", []))
    if caps & _HARD_CAPS:
        return True
    spans = q.get("evidence_spans") or []
    if len(spans) >= 2:                      # 多时刻
        return True
    if q.get("min_watch") == "whole_video":
        return True
    return False


def _is_easy(q: Dict) -> bool:
    if q.get("min_watch") not in ("single_scene", None):
        return False
    a = q.get("anti_shortcut", {})
    if any(a.get(k) for k in ("blind_llm_pass", "single_frame_pass", "subtitle_only_pass",
                              "synopsis_pass", "language_prior_pass")):
        return False
    return True


def tag_difficulty(qa: List[Dict]) -> Dict[str, int]:
    """覆盖 difficulty 字段;返回分布统计。"""
    dist = {"easy": 0, "medium": 0, "hard": 0}
    for q in qa:
        if _is_hard(q):
            q["difficulty"] = "hard"
        elif _is_easy(q):
            q["difficulty"] = "easy"
        else:
            q["difficulty"] = "medium"
        dist[q["difficulty"]] += 1
    n = max(1, len(qa))
    log.info("难度标定完成:easy=%d(%.0f%%) medium=%d(%.0f%%) hard=%d(%.0f%%)",
             dist["easy"], 100 * dist["easy"] / n,
             dist["medium"], 100 * dist["medium"] / n,
             dist["hard"], 100 * dist["hard"] / n)
    return dist
