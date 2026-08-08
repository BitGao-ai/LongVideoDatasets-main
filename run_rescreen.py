#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""换模型复筛 CLI:用不同供应商对已标注结果重跑抗捷径检查。

用法:
  # 初标用 qwen,则复筛用 kimi(反之亦然)
  export MOONSHOT_API_KEY=sk-xxx
  python run_rescreen.py --in annotations/ --out-dir annotations_rescreened/ \
      --provider kimi --report reports/rescreen.json
"""

import argparse
import glob
import json
import logging
import os

from annotator.config import RunConfig
from annotator.llm_client import LLMClient
from annotator.rescreen import rescreen_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("run_rescreen")


def _collect(inp: str):
    if os.path.isdir(inp):
        return sorted(glob.glob(os.path.join(inp, "*.json")))
    return sorted(glob.glob(inp))


def main():
    p = argparse.ArgumentParser(description="换模型复筛(cross-model re-screening)")
    p.add_argument("--in", dest="inp", required=True, help="标注文件目录或 glob")
    p.add_argument("--out-dir", required=True, help="复筛后输出目录")
    p.add_argument("--provider", required=True, choices=["qwen", "kimi"], help="复筛用供应商(须不同于初标)")
    p.add_argument("--report", default="", help="汇总报告 JSON 路径(可选)")
    p.add_argument("--keep-shortcut", action="store_true", help="只标记不剔除")
    p.add_argument("--force", action="store_true", help="即使与初标同供应商也强制复筛")
    args = p.parse_args()

    cfg = RunConfig(provider=args.provider, drop_shortcut=not args.keep_shortcut)
    client = LLMClient(cfg)

    files = _collect(args.inp)
    if not files:
        raise SystemExit(f"未找到标注文件: {args.inp}")
    log.info("待复筛 %d 个文件", len(files))

    stats = []
    for f in files:
        out_path = os.path.join(args.out_dir, os.path.basename(f))
        st = rescreen_file(f, client, cfg, out_path, force=args.force)
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
