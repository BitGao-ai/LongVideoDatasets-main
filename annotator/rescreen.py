# -*- coding: utf-8 -*-
"""换模型复筛:用与初标【不同】的供应商,对已标注的 QA 重跑抗捷径检查。

动机:初标与其抗捷径过滤用的是同一家模型,存在同源偏置——一道题“初标模型答不出”
不代表“真的难”。这里用另一家模型(Qwen<->Kimi)独立复筛,对两家的判定取【或】
(任一模型能走捷径就算捷径),得到更保守、更可信的题库。

每题追加 `cross_check` 字段(保留两家原始判定),并把 `anti_shortcut` 更新为合并判定。
"""

import datetime
import json
import logging
import os
import re
from typing import Dict, List, Optional

from .config import RunConfig
from .filter import check_one, renumber, should_drop
from .llm_client import LLMClient

log = logging.getLogger("annotator.rescreen")

_CHECK_KEYS = ("blind_llm_pass", "single_frame_pass", "subtitle_only_pass")


def _primary_providers(annotation: Dict) -> List[str]:
    """从 annotation_meta.annotators(如 'auto:qwen:qwen-vl-max')解析初标供应商。"""
    out = []
    for a in annotation.get("annotation_meta", {}).get("annotators", []):
        m = re.match(r"auto:([a-z0-9_-]+)", str(a))
        if m:
            out.append(m.group(1))
    return out


def rescreen_annotation(annotation: Dict, client: LLMClient, cfg: RunConfig) -> Dict:
    """就地复筛一个 annotation dict,返回统计信息。"""
    today = datetime.date.today().isoformat()
    secondary = client.pc.name
    structure = annotation.get("structure", {})
    video_path = annotation.get("media", {}).get("video_path", "")

    before = len(annotation.get("qa", []))
    kept: List[Dict] = []
    dropped: List[str] = []

    for q in annotation.get("qa", []):
        if q.get("task_type") != "mcq":
            kept.append(q)
            continue

        primary = {k: bool(q.get("anti_shortcut", {}).get(k)) for k in _CHECK_KEYS}
        sec = check_one(client, cfg, video_path, q, structure)
        combined = {k: primary[k] or sec[k] for k in _CHECK_KEYS}

        q["cross_check"] = {
            "secondary_provider": secondary,
            "primary": primary,
            "secondary": sec,
            "checked_date": today,
        }
        prev_by = q.get("anti_shortcut", {}).get("checked_by", "auto:?")
        q["anti_shortcut"] = {
            **combined,
            "checked_by": f"{prev_by}+auto:{secondary}",
            "checked_date": today,
        }

        if should_drop(cfg, q):
            dropped.append(q["qid"])
        else:
            kept.append(q)

    renumber(kept)
    annotation["qa"] = kept
    # 复筛后仍是初标态,须人工终审
    annotation.setdefault("annotation_meta", {})["review_status"] = "draft"
    return {"before": before, "after": len(kept), "dropped": dropped}


def rescreen_file(path: str, client: LLMClient, cfg: RunConfig,
                  out_path: str, force: bool = False) -> Optional[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    primaries = _primary_providers(annotation)
    if client.pc.name in primaries and not force:
        log.warning("跳过 %s:复筛供应商(%s)与初标相同;换一家或加 --force",
                    os.path.basename(path), client.pc.name)
        return None

    stat = rescreen_annotation(annotation, client, cfg)
    stat["file"] = os.path.basename(path)
    stat["secondary_provider"] = client.pc.name

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)
    log.info("复筛 %s:%d -> %d 题(剔除 %d)",
             stat["file"], stat["before"], stat["after"], len(stat["dropped"]))
    return stat
