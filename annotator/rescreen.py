# -*- coding: utf-8 -*-
"""换模型复筛(Res 8.1 复核范围全覆盖)—— 用与初标【不同】的供应商重跑抗捷径。

- mcq               : 五通道(盲答/字幕/单帧/梗概/语言先验)重跑;
- temporal_grounding: 字幕定位通道重跑(区间 IoU);
- open/summary      : 盲答相似度通道重跑;
- 两家判定取【或】(任一模型能走捷径就算捷径),更保守可信。
- 每题追加 cross_check 保留两家原始判定,anti_shortcut 更新为合并判定。
"""

import datetime
import json
import logging
import os
import re
from typing import Dict, List, Optional

from . import anti_shortcut
from .config import RunConfig
from .frames_store import FrameStore
from .llm_client import LLMClient

log = logging.getLogger("annotator.rescreen")


def _primary_providers(annotation: Dict) -> List[str]:
    """从 annotation_meta.annotators(如 'auto:qwen:qwen-vl-max')解析初标供应商。"""
    out = []
    for a in annotation.get("annotation_meta", {}).get("annotators", []):
        m = re.match(r"auto:([a-z0-9_-]+)", str(a))
        if m:
            out.append(m.group(1))
    return out


def _channel_keys(q: Dict) -> List[str]:
    tt = q.get("task_type")
    if tt == "mcq":
        return ["blind_llm_pass", "single_frame_pass", "subtitle_only_pass",
                "synopsis_pass", "language_prior_pass"]
    if tt == "temporal_grounding":
        return ["subtitle_locate_pass"]
    if tt in ("open", "summary"):
        return ["blind_similar_pass"]
    return []


def rescreen_annotation(annotation: Dict, client: LLMClient, cfg: RunConfig) -> Dict:
    """就地复筛一个 annotation dict,返回统计信息。"""
    today = datetime.date.today().isoformat()
    secondary = client.pc.name
    structure = annotation.get("structure", {})
    video_path = annotation.get("media", {}).get("video_path", "")
    meta = annotation.get("meta", {})
    dur = float(meta.get("duration_sec", 0) or 0)

    # 帧索引(有则用,无则 cv2 兜底)
    store = None
    frames_dir = annotation.get("annotation_meta", {}).get("frames_index", {}).get("frames_dir")
    if frames_dir and os.path.isdir(frames_dir):
        try:
            store = FrameStore(frames_dir, fps=cfg.frame_rate, max_edge=cfg.max_image_edge)
            store.verify_index(dur)
        except Exception as e:  # noqa: BLE001
            log.warning("帧索引不可用,回退 cv2: %s", e)
            store = None

    syn = (annotation.get("global", {}).get("synopsis") or {})
    synopsis = syn.get("zh") or syn.get("en")

    before = len(annotation.get("qa", []))
    kept: List[Dict] = []
    dropped: List[str] = []

    for q in annotation.get("qa", []):
        keys = _channel_keys(q)
        if not keys:
            kept.append(q)
            continue
        primary = {k: bool(q.get("anti_shortcut", {}).get(k)) for k in keys}
        sec = anti_shortcut.check_one(client, cfg, video_path, q, structure,
                                      store=store, synopsis=synopsis)
        combined = {k: primary[k] or bool(sec.get(k)) for k in keys}

        q["cross_check"] = {
            "secondary_provider": secondary,
            "primary": primary,
            "secondary": {k: bool(sec.get(k)) for k in keys},
            "checked_date": today,
        }
        prev_by = q.get("anti_shortcut", {}).get("checked_by", "auto:?")
        q["anti_shortcut"] = {
            **combined,
            "checked_by": f"{prev_by}+auto:{secondary}",
            "checked_date": today,
        }

        if anti_shortcut.should_drop(cfg, q):
            dropped.append(q["qid"])
        else:
            kept.append(q)

    anti_shortcut.renumber(kept)
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
