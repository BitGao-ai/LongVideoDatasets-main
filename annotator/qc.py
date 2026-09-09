# -*- coding: utf-8 -*-
"""质量控制:标注一致性(Krippendorff α / Cohen κ)+ 黄金题监控 + 复核打回。

纯标准库实现,无第三方依赖,便于单测。

术语对齐标注手册:
- 每题由 1 人出题、≥2 人独立盲答验证;
- 一致性入库门槛 α ≥ 0.8;
- 黄金题(已知答案)监控标注员,正确率 < 0.9 触发复训 + 该批复核。
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence


# =========================================================================
# 一致性:Krippendorff's alpha(nominal / ordinal)
# =========================================================================
def krippendorff_alpha(units: Sequence[Sequence[Optional[object]]],
                       level: str = "nominal") -> float:
    """计算 Krippendorff's alpha。

    units: 每个元素是【同一题】各评分者给出的标签列表,缺失用 None。
           例如 [["A","A","B"], ["C","C"], ...]。
    level: "nominal"(如 MCQ 选项)或 "ordinal"(如 difficulty 有序等级)。
    返回 α ∈ (-∞, 1];完全一致=1,达到随机=0。
    """
    values = sorted({v for u in units for v in u if v is not None},
                    key=lambda x: (str(type(x)), x))
    idx = {v: i for i, v in enumerate(values)}
    V = len(values)
    if V < 2:
        # 所有验证者都给了同一个标签:观察分歧与期望分歧同时为 0,α 在数学上
        # 未定义。此前返回 1.0,会让"全部人都选 A"这种退化样本轻松通过门禁。
        return float("nan")

    # 重合矩阵 o[c][k]
    o = [[0.0] * V for _ in range(V)]
    for u in units:
        vals = [v for v in u if v is not None]
        m = len(vals)
        if m < 2:
            continue
        cnt = Counter(vals)
        for c, nc in cnt.items():
            for k, nk in cnt.items():
                pairs = nc * (nk - 1) if c == k else nc * nk
                o[idx[c]][idx[k]] += pairs / (m - 1)

    n_c = [sum(row) for row in o]
    n = sum(n_c)
    if n == 0:
        return float("nan")

    def delta(i: int, j: int) -> float:
        if level == "nominal":
            return 0.0 if i == j else 1.0
        if level == "ordinal":
            lo, hi = (i, j) if i <= j else (j, i)
            s = sum(n_c[lo:hi + 1]) - (n_c[lo] + n_c[hi]) / 2.0
            return s * s
        raise ValueError(f"未知 level: {level}")

    do = sum(o[i][j] * delta(i, j) for i in range(V) for j in range(V))
    de = sum(n_c[i] * n_c[j] * delta(i, j) for i in range(V) for j in range(V))
    if de == 0:
        return 1.0
    return 1.0 - (n - 1) * do / de


def cohen_kappa(a: Sequence[object], b: Sequence[object]) -> float:
    """两评分者 Cohen's kappa(名义)。a、b 等长,None 视为缺失并成对丢弃。"""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in pairs if x == y) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    labels = set(ca) | set(cb)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


# =========================================================================
# 黄金题监控
# =========================================================================
def grade_gold(responses: List[Dict], gold_key: Dict[str, str],
               threshold: float = 0.9) -> Dict[str, Dict]:
    """按标注员统计黄金题正确率。

    responses: [{"qid","annotator","answer"}, ...]
    gold_key : {qid: 正确答案};通常来自标注文件中 is_gold=true 的题。
    返回 {annotator: {correct,total,accuracy,flag}};flag=True 表示 < 阈值需复训。
    """
    tally: Dict[str, List[int]] = defaultdict(lambda: [0, 0])  # [correct, total]
    for r in responses or []:
        qid = (r or {}).get("qid")
        annotator = r.get("annotator") if r else None
        if qid in gold_key and annotator:
            tally[annotator][1] += 1
            if r.get("answer") == gold_key[qid]:
                tally[annotator][0] += 1
    out = {}
    for ann, (c, t) in tally.items():
        acc = c / t if t else float("nan")
        out[ann] = {"correct": c, "total": t, "accuracy": acc,
                    "flag": bool(t > 0 and acc < threshold)}
    return out


# =========================================================================
# 复核打回:验证者答案与出题者答案不一致 -> 候选打回
# =========================================================================
def rework_candidates(responses: List[Dict], answer_key: Dict[str, str]) -> List[Dict]:
    """找出验证者答案与标准答案不完全一致的题(需打回重做)。

    answer_key: {qid: 出题者标注的正确答案}(来自标注文件 q["answer"])。
    """
    by_q: Dict[str, List[str]] = defaultdict(list)
    for r in responses or []:
        qid = (r or {}).get("qid")
        if qid in answer_key and r.get("answer") is not None:
            by_q[qid].append(r["answer"])
    flagged = []
    for qid, ans in by_q.items():
        gold = answer_key[qid]
        agree = sum(1 for a in ans if a == gold)
        if agree < len(ans):  # 有验证者不同意出题者
            flagged.append({"qid": qid, "gold": gold, "verifier_answers": ans,
                            "agreement": agree / len(ans) if ans else 0.0})
    return sorted(flagged, key=lambda x: x["agreement"])


# =========================================================================
# 汇总
# =========================================================================
def build_keys(annotations: List[Dict]):
    """从标注文件列表抽取 {qid: answer} 全量键 与 {qid: answer} 黄金键。"""
    answer_key, gold_key = {}, {}
    for ann in annotations:
        for q in ann.get("qa", []):
            if q.get("task_type") == "mcq" and q.get("answer"):
                answer_key[q["qid"]] = q["answer"]
                if q.get("is_gold"):
                    gold_key[q["qid"]] = q["answer"]
    return answer_key, gold_key


def run_qc(annotations: List[Dict], responses: List[Dict],
           alpha_threshold: float = 0.8, gold_threshold: float = 0.9,
           level: str = "nominal") -> Dict:
    answer_key, gold_key = build_keys(annotations)

    # 一致性:按题聚合验证者答案(排除黄金题,避免难度偏置)
    per_q: Dict[str, List[str]] = defaultdict(list)
    for r in responses or []:
        qid = (r or {}).get("qid")
        if qid in answer_key and qid not in gold_key and r.get("answer"):
            per_q[qid].append(r["answer"])
    units = [v for v in per_q.values() if len(v) >= 2]
    alpha = krippendorff_alpha(units, level=level) if units else float("nan")

    gold_report = grade_gold(responses, gold_key, gold_threshold)
    non_gold_key = {q: a for q, a in answer_key.items() if q not in gold_key}
    rework = rework_candidates(responses, non_gold_key)

    flagged_annotators = [a for a, v in gold_report.items() if v["flag"]]
    return {
        "iaa": {
            "krippendorff_alpha": alpha,
            "level": level,
            "n_items": len(units),
            "threshold": alpha_threshold,
            "pass": bool(alpha == alpha and alpha >= alpha_threshold),  # NaN 不通过
        },
        "gold": {
            "threshold": gold_threshold,
            "per_annotator": gold_report,
            "flagged_annotators": flagged_annotators,
        },
        "rework_candidates": rework,
    }
