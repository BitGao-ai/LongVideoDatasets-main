#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给【已标注】文件补全全片详细描述(分段密集 + 全片汇总)。

用法:
  python run_describe.py --in annotations/ --out-dir annotations_desc/ \
      --provider qwen --frames 12
也可对单文件:--in annotations/doc_00001.json

写入 structure.segments[] 与 global.detailed_description;不改动其它字段。
若初标 JSON 缺少 structure(仅有 meta),请改用 run_annotate.py --describe 从头跑。
"""

import argparse
import glob
import json
import logging
import os

from annotator.config import RunConfig
from annotator.describe import describe_video
from annotator.llm_client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("run_describe")


def _collect(inp: str):
    if os.path.isdir(inp):
        return sorted(glob.glob(os.path.join(inp, "*.json")))
    return sorted(glob.glob(inp))


def describe_file(path: str, client: LLMClient, cfg: RunConfig, out_path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    structure = ann.get("structure", {})
    video_path = ann.get("media", {}).get("video_path", "")

    segments, detailed = describe_video(client, cfg, video_path, structure, ann)
    ann.setdefault("structure", {})["segments"] = segments
    ann.setdefault("global", {})["detailed_description"] = detailed
    ann.setdefault("annotation_meta", {})["review_status"] = "draft"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)
    log.info("详述写入 %s:%d 段", os.path.basename(path), len(segments))
    return {"file": os.path.basename(path), "segments": len(segments)}


def main():
    p = argparse.ArgumentParser(description="给已标注文件补全全片详细描述")
    p.add_argument("--in", dest="inp", required=True, help="标注文件目录或 glob")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--provider", default="qwen", choices=["qwen", "kimi"])
    p.add_argument("--describe-mode", default="frames", choices=["frames", "native_video"],
                   help="frames=抽帧(Qwen/Kimi 通用);native_video=原生视频片段(仅 Qwen,需 ffmpeg+dashscope)")
    p.add_argument("--native-fps", type=float, default=2.0, help="native_video 模式下的视频抽帧率")
    p.add_argument("--window-sec", type=float, default=180.0)
    p.add_argument("--frames", type=int, default=12, help="每窗口采样帧数(详述可略多)")
    p.add_argument("--agg-group", type=int, default=15, help="两级汇总的分组大小")
    p.add_argument("--max-windows", type=int, default=0)
    args = p.parse_args()

    cfg = RunConfig(provider=args.provider, window_sec=args.window_sec,
                    describe_mode=args.describe_mode, native_fps=args.native_fps,
                    describe_frames_per_window=args.frames, describe_agg_group=args.agg_group,
                    max_windows=args.max_windows)
    client = LLMClient(cfg)

    files = _collect(args.inp)
    if not files:
        raise SystemExit(f"未找到标注文件: {args.inp}")
    log.info("待补详述 %d 个文件", len(files))
    for f in files:
        describe_file(f, client, cfg, os.path.join(args.out_dir, os.path.basename(f)))


if __name__ == "__main__":
    main()
