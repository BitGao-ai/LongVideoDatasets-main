# -*- coding: utf-8 -*-
"""端到端编排:读取 meta + 预处理结构 -> LLM 标注 -> 出题 -> 抗捷径过滤
-> 组装为 schema JSON -> 校验 -> 落盘。"""

import datetime
import json
import logging
import os
from typing import Dict, Optional

from .annotate import annotate_global, annotate_structure
from .config import RunConfig
from .filter import run_filters
from .llm_client import LLMClient
from .qa_generate import generate_qa

log = logging.getLogger("annotator.pipeline")


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _duration_bucket(sec: float) -> str:
    if sec < 300:
        return "<5m"
    if sec < 900:
        return "5-15m"
    if sec < 2400:
        return "15-40m"
    if sec <= 7200:
        return "40m-2h"
    return ">2h"


def _validate(annotation: Dict, schema_path: Optional[str]) -> None:
    if not schema_path or not os.path.exists(schema_path):
        return
    try:
        import jsonschema
        jsonschema.validate(annotation, _load_json(schema_path))
        log.info("Schema 校验通过 ✅")
    except ImportError:
        log.warning("未安装 jsonschema,跳过校验")
    except Exception as e:  # noqa: BLE001 —— 校验失败不阻断落盘,仅告警
        log.warning("Schema 校验未通过(仍会落盘,请人工核对): %s", str(e)[:300])


def annotate_video(meta_path: str, structure_path: str, out_path: str,
                   cfg: RunConfig, schema_path: Optional[str] = None) -> Dict:
    meta = _load_json(meta_path)                 # {video_id, meta, media}
    structure_in = _load_json(structure_path)    # {video_id, structure:{...}}
    structure = structure_in.get("structure", structure_in)
    video_path = meta["media"]["video_path"]

    # 补全 duration_bucket
    dur = float(meta["meta"].get("duration_sec", 0))
    meta["meta"].setdefault("duration_bucket", _duration_bucket(dur))

    client = LLMClient(cfg)
    log.info("=== 标注 %s (provider=%s) ===", meta["video_id"], cfg.provider)

    # 1) 场景/事件
    scenes, events = annotate_structure(client, cfg, video_path, structure, meta)
    # 2) 全局聚合 + 因果回填
    global_block, events = annotate_global(client, cfg, scenes, events, meta)
    # 2.5) 可选:全片详细描述
    segments = []
    if cfg.describe:
        from .describe import describe_video
        segments, detailed = describe_video(client, cfg, video_path, structure, meta)
        global_block["detailed_description"] = detailed
    # 3) 出题
    qa = generate_qa(client, cfg, scenes, events, global_block, meta)
    # 4) 抗捷径过滤
    qa = run_filters(client, cfg, video_path, qa, structure)

    if not qa:
        log.warning("过滤后无有效题目,请检查视频/字幕或放宽 drop_shortcut")

    annotation = {
        "schema_version": "1.0",
        "video_id": meta["video_id"],
        "meta": meta["meta"],
        "media": meta["media"],
        "structure": {
            "shots": structure.get("shots", []),
            "scenes": scenes,
            "events": events,
            "segments": segments,
            "subtitles": structure.get("subtitles", []),
            "ocr": structure.get("ocr", []),
        },
        "global": global_block,
        "qa": qa,
        "annotation_meta": {
            "created_date": datetime.date.today().isoformat(),
            "annotators": [f"auto:{client.pc.name}:{client.pc.vision_model}"],
            "iaa_alpha": None,
            "review_status": "draft",   # 机器初标,须经人工复核才可置 verified/final
        },
    }

    _validate(annotation, schema_path)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)
    log.info("已写出标注结果: %s (%d 题)", out_path, len(qa))
    return annotation
