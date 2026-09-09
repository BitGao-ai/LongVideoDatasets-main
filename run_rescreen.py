#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""换模型复筛 CLI:用不同供应商对已标注结果重跑抗捷径检查。

用法:
  # 初标用 qwen,则复筛用 kimi(反之亦然)
  export MOONSHOT_API_KEY=sk-xxx
  python run_rescreen.py --in annotations/ --out-dir annotations_rescreened/ \
      --provider kimi --report reports/rescreen.json
  # 本地模型复筛(local 与初标不同名,通常需 --force 或初标用云端)
  python run_rescreen.py --in annotations/ --out-dir annotations_rescreened/ \
      --provider local --local-base-url http://127.0.0.1:8000/v1 \
      --local-model Qwen/Qwen3-VL-8B-Instruct --force
"""

import argparse
import glob
import json
import logging
import os

from annotator.config import RunConfig
from annotator.llm_client import LLMClient
from annotator.rescreen import rescreen_file
from annotator.utils import quiet_http_loggers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
quiet_http_loggers()
log = logging.getLogger("run_rescreen")


def _collect(inp: str):
    files = sorted(glob.glob(os.path.join(inp, "*.json"))) if os.path.isdir(inp) \
        else sorted(glob.glob(inp))
    return [f for f in files if not f.endswith(".manifest.json")]


def main():
    p = argparse.ArgumentParser(description="换模型复筛(cross-model re-screening)")
    p.add_argument("--in", dest="inp", required=True, help="标注文件目录或 glob")
    p.add_argument("--out-dir", required=True, help="复筛后输出目录")
    p.add_argument("--provider", required=True, choices=["qwen", "kimi", "local"],
                    help="复筛用供应商(须不同于初标;local 须配 --local-base-url/--local-model)")
    p.add_argument("--report", default="", help="汇总报告 JSON 路径(可选)")
    p.add_argument("--keep-shortcut", action="store_true", help="只标记不剔除")
    p.add_argument("--frame-rate", type=float, default=1.0, help="帧索引抽帧率")
    p.add_argument("--local-base-url", default="", help="本地模型端点(vLLM):provider=local 时的主端点,兼做帧级描述/音频判别")
    p.add_argument("--local-model", default="", help="本地视觉模型名(如 Qwen/Qwen3-VL-8B-Instruct)")
    p.add_argument("--force", action="store_true", help="即使与初标同供应商也强制复筛")
    args = p.parse_args()

    cfg = RunConfig(provider=args.provider, drop_shortcut=not args.keep_shortcut,
                    frame_rate=args.frame_rate, local_base_url=args.local_base_url,
                    local_vision_model=args.local_model)
    client = LLMClient(cfg)

    files = _collect(args.inp)
    if not files:
        raise SystemExit(f"未找到标注文件: {args.inp}")
    log.info("待复筛 %d 个文件", len(files))

    stats = []
    for i, f in enumerate(files, 1):
        out_path = os.path.join(args.out_dir, os.path.basename(f))
        log.info("[%d/%d] 复筛: %s", i, len(files), f)
        try:
            st = rescreen_file(f, client, cfg, out_path, force=args.force)
        except Exception as e:  # noqa: BLE001
            log.error("复筛失败(继续下一个): %s (%s)", f, e)
            continue
        if st:
            stats.append(st)

    total_before = sum(s["before"] for s in stats)
    total_after = sum(s["after"] for s in stats)
    total_dropped = sum(len(s["dropped"]) for s in stats)
    summary = {"files": len(stats), "qa_before": total_before, "qa_after": total_after,
               "dropped": total_dropped, "per_file": stats}
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        log.info("报告已写出: %s", args.report)


if __name__ == "__main__":
    main()
