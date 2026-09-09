#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给【已标注】文件补全全片详细描述(M1 密集描述金字塔:镜头级+片段级+全片级)。

用法:
  python run_describe.py --in annotations/ --out-dir annotations_desc/ \
      --provider qwen --frames 12
也可对单文件:--in annotations/doc_00001.json

写入 structure.shots[].desc(镜头级)、structure.segments[] 与
global.detailed_description;不改动其它字段。质量门:片段中文≥400 字/
英文≥350 词、要素≥6/8、≥2 处时间锚点;未过门重试后进 quiet_segments。
"""

import argparse
import glob
import json
import logging
import os

from annotator.config import RunConfig
from annotator.describe import describe_video
from annotator.frames_store import FrameStore
from annotator.llm_client import LLMClient
from annotator.utils import quiet_http_loggers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
quiet_http_loggers()
log = logging.getLogger("run_describe")


def _collect(inp: str):
    files = sorted(glob.glob(os.path.join(inp, "*.json"))) if os.path.isdir(inp) \
        else sorted(glob.glob(inp))
    return [f for f in files if not f.endswith(".manifest.json")]


def describe_file(path: str, client: LLMClient, cfg: RunConfig, out_path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    structure = ann.get("structure", {})
    video_path = ann.get("media", {}).get("video_path", "")
    dur = float(ann.get("meta", {}).get("duration_sec", 0) or 0)

    # 帧索引:优先复用上次标注的 frames_dir,否则就地构建
    frames_dir = cfg.frame_index_dir or ann.get("annotation_meta", {}) \
        .get("frames_index", {}).get("frames_dir") or os.path.join(
            os.path.dirname(path) or ".", "frames", ann.get("video_id", "v"))
    store = FrameStore(frames_dir, fps=cfg.frame_rate, max_edge=cfg.max_image_edge)
    store.set_duration(dur)
    store.build(video_path, force=cfg.force_frame_index)
    store.verify_index(dur)

    segments, detailed, report = describe_video(client, cfg, video_path, structure, ann,
                                                store=store)
    structure["segments"] = segments
    ann.setdefault("global", {})["detailed_description"] = detailed
    am = ann.setdefault("annotation_meta", {})
    am["review_status"] = "draft"
    # quiet_segments 必须落盘,否则 QC 的 segment_gate 会假通过(D10)
    stats = am.setdefault("stats", {})
    stats["quiet_segments"] = report.get("quiet_segments", [])
    stats["shot_desc_ok"] = f"{report.get('shot_ok')}/{report.get('shot_total')}"
    stats["full_len_zh"] = report.get("full_len_zh")
    for issue in report.get("lang_issues") or []:
        am.setdefault("human_todo", []).append({"kind": "lang_issue", "reason": issue})

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)
    log.info("详述写入 %s:%d 段(镜头级 desc %s;未过门 %d 段)",
             os.path.basename(path), len(segments), stats["shot_desc_ok"],
             len(stats["quiet_segments"]))
    return {"file": os.path.basename(path), "segments": len(segments),
            "quiet_segments": len(stats["quiet_segments"])}


def main():
    p = argparse.ArgumentParser(description="给已标注文件补全全片详细描述(v2)")
    p.add_argument("--in", dest="inp", required=True, help="标注文件目录或 glob")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--provider", default="qwen", choices=["qwen", "kimi", "local"],
                    help="标注供应商;local=本地端点(须配 --local-base-url/--local-model)")
    p.add_argument("--local-base-url", default="", help="本地模型端点(vLLM),provider=local 时使用")
    p.add_argument("--local-model", default="", help="本地视觉模型名(如 Qwen/Qwen3-VL-8B-Instruct)")
    p.add_argument("--describe-mode", default="frames", choices=["frames", "native_video"],
                   help="frames=抽帧(所有 provider 通用);native_video=原生视频片段(仅云端 Qwen)")
    p.add_argument("--native-fps", type=float, default=2.0, help="native_video 模式下的视频抽帧率")
    p.add_argument("--window-sec", type=float, default=180.0)
    p.add_argument("--frames", type=int, default=12, help="每窗口采样帧数(详述可略多)")
    p.add_argument("--frame-rate", type=float, default=1.0, help="帧索引抽帧率")
    p.add_argument("--agg-group", type=int, default=15, help="两级汇总的分组大小")
    p.add_argument("--max-windows", type=int, default=0)
    args = p.parse_args()

    cfg = RunConfig(provider=args.provider, window_sec=args.window_sec,
                    describe_mode=args.describe_mode, native_fps=args.native_fps,
                    describe_frames_per_window=args.frames, describe_agg_group=args.agg_group,
                    frame_rate=args.frame_rate, max_windows=args.max_windows,
                    local_base_url=args.local_base_url, local_vision_model=args.local_model)
    client = LLMClient(cfg)

    files = _collect(args.inp)
    if not files:
        raise SystemExit(f"未找到标注文件: {args.inp}")
    log.info("待补详述 %d 个文件", len(files))
    failed = []
    for i, f in enumerate(files, 1):
        log.info("[%d/%d] 详述: %s", i, len(files), f)
        try:
            describe_file(f, client, cfg, os.path.join(args.out_dir, os.path.basename(f)))
        except Exception as e:  # noqa: BLE001
            log.error("详述失败: %s (%s)", f, e)
            failed.append(f)
    if failed:
        raise SystemExit(f"详述完成,失败 {len(failed)} 个: {', '.join(failed)}")
    log.info("详述全部完成,共 %d 个", len(files))


if __name__ == "__main__":
    main()
