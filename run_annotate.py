#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""长视频 LLM 自动标注 CLI。

用法:
  export DASHSCOPE_API_KEY=sk-xxx        # provider=qwen
  # 或 export MOONSHOT_API_KEY=sk-xxx    # provider=kimi

  python run_annotate.py \
      --meta       examples/meta_template.json \
      --structure  structure/doc_00001.json \
      --out        annotations/doc_00001.json \
      --provider   qwen \
      --qa 12
"""

import argparse
import logging

from annotator.config import RunConfig
from annotator.pipeline import annotate_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main():
    p = argparse.ArgumentParser(description="长视频 LLM 自动标注(Qwen / Kimi)")
    p.add_argument("--meta", required=True, help="视频元信息 JSON(video_id/meta/media)")
    p.add_argument("--structure", required=True, help="预处理结构 JSON(preprocess.py 产出)")
    p.add_argument("--out", required=True, help="标注结果输出路径")
    p.add_argument("--schema", default="schema/annotation.schema.json", help="校验用 schema")
    p.add_argument("--provider", default="qwen", choices=["qwen", "kimi"])
    p.add_argument("--window-sec", type=float, default=180.0)
    p.add_argument("--frames", type=int, default=10, help="每窗口采样帧数")
    p.add_argument("--qa", type=int, default=12, help="每视频出题数")
    p.add_argument("--describe", action="store_true", help="附带生成全片详细描述(分段+汇总)")
    p.add_argument("--describe-mode", default="frames", choices=["frames", "native_video"],
                   help="frames=抽帧(Qwen/Kimi 通用);native_video=原生视频片段(仅 Qwen,需 ffmpeg+dashscope)")
    p.add_argument("--native-fps", type=float, default=2.0, help="native_video 模式下的视频抽帧率")
    p.add_argument("--max-windows", type=int, default=0, help=">0 时只处理前 N 窗(调试)")
    p.add_argument("--keep-shortcut", action="store_true", help="不剔除命中捷径的题(仅标记)")
    args = p.parse_args()

    cfg = RunConfig(
        provider=args.provider,
        window_sec=args.window_sec,
        frames_per_window=args.frames,
        qa_per_video=args.qa,
        describe=args.describe,
        describe_mode=args.describe_mode,
        native_fps=args.native_fps,
        max_windows=args.max_windows,
        drop_shortcut=not args.keep_shortcut,
    )
    annotate_video(args.meta, args.structure, args.out, cfg, schema_path=args.schema)


if __name__ == "__main__":
    main()
