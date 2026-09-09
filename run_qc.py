#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""质量控制 CLI v2(Res 8):MCQ α + 黄金题 + temporal mIoU + open/summary 一致率
+ 视频级发布门禁,输出 release_report.json。

用法:
  python run_qc.py --annotations annotations/ --responses reviews/responses.json \
      --alpha-threshold 0.8 --report reports/qc.json

responses.json 格式见 examples/responses_template.json(v2):
  {"responses": [
     {"qid": "...", "annotator": "ann_07", "answer": "B"},                    # mcq
     {"qid": "...", "annotator": "ann_07", "interval": [12.0, 34.5]},         # temporal_grounding
     {"qid": "...", "annotator": "ann_07", "text": "自由作答文本"}            # open/summary
  ]}

发布门禁(Res 8.2):任何项不达标 → review_status 保持 draft 并列出拒绝原因;
全部达标 → 置 verified。
"""

import argparse
import datetime
import glob
import json
import logging
import os

from annotator.config import RunConfig
from annotator.llm_client import LLMClient
from annotator.qc_v2 import run_qc_v2
from annotator.utils import quiet_http_loggers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
quiet_http_loggers()
log = logging.getLogger("run_qc")


def _load_annotations(inp: str):
    """返回 (annotations, paths);paths 与 annotations 一一对应,供门禁回写。"""
    files = sorted(glob.glob(os.path.join(inp, "*.json"))) if os.path.isdir(inp) else sorted(glob.glob(inp))
    files = [f for f in files if not f.endswith(".manifest.json")]
    out, paths, bad = [], [], []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:  # noqa: BLE001
            bad.append((f, str(e)))
            continue
        if not isinstance(data, dict) or "qa" not in data:
            log.debug("跳过非标注文件: %s", f)
            continue
        out.append(data)
        paths.append(f)
    for f, e in bad:
        log.warning("跳过无法读取的标注文件: %s (%s)", f, e)
    return out, paths


def main():
    p = argparse.ArgumentParser(description="标注质量控制 v2(黄金题 + 一致性 + 发布门禁)")
    p.add_argument("--annotations", required=True, help="标注文件目录或 glob(提供标准答案/黄金题)")
    p.add_argument("--responses", required=True, help="验证者独立作答 JSON(v2 模板)")
    p.add_argument("--alpha-threshold", type=float, default=0.8)
    p.add_argument("--gold-threshold", type=float, default=0.9)
    p.add_argument("--miou-threshold", type=float, default=0.5)
    p.add_argument("--report", default="reports/qc.json", help="报告 JSON 输出路径")
    p.add_argument("--release-report", default="reports/release_report.json",
                   help="发布报告 JSON 输出路径(门禁清单)")
    p.add_argument("--no-write-back", action="store_true",
                   help="只出报告,不把门禁判定(review_status)回写标注文件")
    p.add_argument("--embedding", action="store_true",
                   help="open/summary 一致率用供应商 embedding(默认 ROUGE-L 纯标准库)")
    p.add_argument("--provider", default="qwen", choices=["qwen", "kimi", "local"],
                    help="embedding 用供应商;local 时需配 --local-base-url/--local-model(本地无 embedding 服务,将自动回退 ROUGE-L)")
    p.add_argument("--local-base-url", default="", help="本地模型端点(vLLM),provider=local 时使用")
    p.add_argument("--local-model", default="", help="本地视觉模型名(如 Qwen/Qwen3-VL-8B-Instruct)")
    args = p.parse_args()

    annotations, ann_paths = _load_annotations(args.annotations)
    if not annotations:
        raise SystemExit(f"未找到标注文件: {args.annotations}")
    with open(args.responses, encoding="utf-8") as f:
        responses = json.load(f).get("responses", [])

    cfg = RunConfig(alpha_threshold=args.alpha_threshold, gold_threshold=args.gold_threshold,
                    miou_threshold=args.miou_threshold, provider=args.provider,
                    local_base_url=args.local_base_url, local_vision_model=args.local_model)
    client = None
    if args.embedding:
        try:
            client = LLMClient(cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("embedding 客户端不可用,回退 ROUGE-L: %s", e)

    report = run_qc_v2(annotations, responses, cfg, client=client)

    # 门禁判定必须落盘,否则 review_status 永远停在 draft(修复 B6)
    if not args.no_write_back:
        for path, ann in zip(ann_paths, annotations):
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(ann, fh, ensure_ascii=False, indent=2)
            except OSError as e:
                log.error("回写 review_status 失败 %s: %s", path, e)
        log.info("已把 review_status / 拒绝原因回写到 %d 个标注文件", len(ann_paths))
    else:
        log.info("--no-write-back:仅出报告,不修改标注文件的 review_status")

    # ---------- 人类可读摘要 ----------
    mcq, iv, op = report["mcq"], report["temporal_miou"], report["open_summary"]

    def _mark(passed, n):
        if not n:
            return "未评估(无验证者样本)"
        return "通过 ✅" if passed else "未达标 ❌"

    print("\n===== 质量控制报告 v2 =====")
    a = mcq["krippendorff_alpha"]
    print(f"[MCQ] Krippendorff α = {a:.4f}  (n={mcq['n_items']}题, 门槛 {cfg.alpha_threshold}) "
          f"-> {_mark(mcq['alpha_pass'], mcq['n_items'])}")
    print(f"[MCQ] 黄金题 门槛 {cfg.gold_threshold};触发复训: {mcq['flagged_annotators'] or '无'}")
    print(f"[MCQ] 打回 {len(mcq['rework'])} 道(验证者与出题者不一致)")
    print(f"[temporal_grounding] mIoU 合格率 = {iv['pass_ratio']} (n={iv['n']}, "
          f"合格线≥{cfg.miou_threshold}, 样本合格率≥{100 * cfg.miou_pass_ratio:.0f}%) "
          f"-> {_mark(iv['pass'], iv['n'])}")
    print(f"[open/summary] 一致率 = {op['agree_ratio']} (n={op['n']}, "
          f"需≥{cfg.open_agree_threshold}) -> {_mark(op['pass'], op['n'])}")
    print("\n[视频级发布门禁]")
    for v in report["per_video"]:
        mark = "✅ 可发布" if v["pass"] else "❌ 拒绝"
        print(f"  {v['video_id']}: {mark}")
        for r in v["reasons"]:
            print(f"      - {r}")
        for u in v.get("unevaluated") or []:
            print(f"      · 未评估(缺样本,不计入拒绝): {u}")
    print(f"\n全部视频门禁: {'通过 ✅' if report['all_gates_pass'] else '存在拒绝 ❌'}"
          f";人工清单合计 {report['human_todo_total']} 项")

    release = {
        "generated_date": datetime.date.today().isoformat(),
        "all_gates_pass": report["all_gates_pass"],
        "per_video": report["per_video"],
        "human_todo_total": report["human_todo_total"],
        "thresholds": {"alpha": cfg.alpha_threshold, "gold": cfg.gold_threshold,
                       "miou": cfg.miou_threshold, "miou_pass_ratio": cfg.miou_pass_ratio,
                       "open_sim": cfg.open_sim_threshold,
                       "open_agree": cfg.open_agree_threshold,
                       "min_qa_after_filter": cfg.min_qa_after_filter},
    }
    for path, payload in ((args.report, report), (args.release_report, release)):
        if not path:
            continue
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        log.info("报告已写出: %s", path)


if __name__ == "__main__":
    main()
