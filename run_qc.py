#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""质量控制 CLI:一致性 α + 黄金题正确率 + 复核打回清单。

用法:
  python run_qc.py --annotations annotations/ --responses reviews/responses.json \
      --alpha-threshold 0.8 --gold-threshold 0.9 --report reports/qc.json

responses.json 格式见 examples/responses_template.json:
  {"responses": [{"qid": "...", "annotator": "ann_07", "answer": "B"}, ...]}
标准答案与黄金题(is_gold)从 --annotations 的标注文件自动读取。
"""

import argparse
import glob
import json
import logging
import os

from annotator.qc import run_qc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_qc")


def _load_annotations(inp: str):
    files = sorted(glob.glob(os.path.join(inp, "*.json"))) if os.path.isdir(inp) else sorted(glob.glob(inp))
    return [json.load(open(f, encoding="utf-8")) for f in files]


def main():
    p = argparse.ArgumentParser(description="标注质量控制(黄金题 + 一致性)")
    p.add_argument("--annotations", required=True, help="标注文件目录或 glob(提供标准答案/黄金题)")
    p.add_argument("--responses", required=True, help="验证者独立作答 JSON")
    p.add_argument("--alpha-threshold", type=float, default=0.8)
    p.add_argument("--gold-threshold", type=float, default=0.9)
    p.add_argument("--level", default="nominal", choices=["nominal", "ordinal"])
    p.add_argument("--report", default="", help="报告 JSON 输出路径(可选)")
    args = p.parse_args()

    annotations = _load_annotations(args.annotations)
    if not annotations:
        raise SystemExit(f"未找到标注文件: {args.annotations}")
    with open(args.responses, encoding="utf-8") as f:
        responses = json.load(f).get("responses", [])

    report = run_qc(annotations, responses, args.alpha_threshold, args.gold_threshold, args.level)

    # 人类可读摘要
    iaa = report["iaa"]
    print("\n===== 质量控制报告 =====")
    a = iaa["krippendorff_alpha"]
    print(f"[一致性] Krippendorff α = {a:.4f}  (n={iaa['n_items']}题, 门槛 {iaa['threshold']}) "
          f"-> {'通过 ✅' if iaa['pass'] else '未达标 ❌'}")
    print(f"[黄金题] 门槛 {report['gold']['threshold']}")
    for ann, v in sorted(report["gold"]["per_annotator"].items()):
        mark = "⚠️ 需复训" if v["flag"] else "ok"
        print(f"   {ann}: {v['correct']}/{v['total']} = {v['accuracy']:.2f}  {mark}")
    if report["gold"]["flagged_annotators"]:
        print("   触发复训:", ", ".join(report["gold"]["flagged_annotators"]))
    print(f"[打回] {len(report['rework_candidates'])} 道题验证者与出题者不一致")
    for r in report["rework_candidates"][:10]:
        print(f"   {r['qid']}: gold={r['gold']} verifiers={r['verifier_answers']} 一致率={r['agreement']:.2f}")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        log.info("报告已写出: %s", args.report)


if __name__ == "__main__":
    main()
