# -*- coding: utf-8 -*-
"""QC v2(Res 8)—— 把质检从 MCQ 扩展到全 task_type + 发布门禁。

指标(纯标准库 + numpy):
- MCQ                : Krippendorff α(复用 qc.py),黄金题正确率(复用 grade_gold);
- temporal_grounding : 验证者作答区间 [s,e] 与标准区间 mIoU ≥ 0.5;
- open/summary       : 两验证者 ROUGE-L/embedding 相似度一致率 ≥ 0.6 + 人工抽测;
- 发布门槛(视频级,Res 8.2):无 gaps、事件密度 ≥45/h、key 事件 100% 有效 span、
  五通道后 ≥8 题、能力矩阵通过、mIoU 合格率 ≥90%、α ≥0.8、双语完备率 ≥95%;
  有 temporal 题但既无验证者作答、也无 S6 区间核验结果时,mIoU 门直接拦截。

每次 run_qc 输出 release_report.json;任何项不达标 → review_status 保持 draft
并列出拒绝原因(强制门禁,而非仅告警)。
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from .config import RunConfig
from .qc import grade_gold, krippendorff_alpha
from .utils import miou, text_similarity

log = logging.getLogger("annotator.qc_v2")


# =========================================================================
# ROUGE-L(纯标准库,LCS 最长公共子序列 F1)
# =========================================================================
#: CJK 逐字切分,拉丁串按词切分。中文没有空格,直接 str.split() 会把整段当成
#: 一个 token,ROUGE-L 非 0 即 1(B5)。
_TOKEN_RE = re.compile(
    r"[一-鿿㐀-䶿぀-ヿ가-힯]"   # CJK / 假名 / 谚文:逐字
    r"|[A-Za-z]+(?:'[A-Za-z]+)?"                                 # 拉丁词
    r"|\d+(?:\.\d+)?"                                            # 数字
)


def tokenize(text: Optional[str]) -> List[str]:
    """语言无关分词:CJK 逐字 + 拉丁按词 + 数字成串,标点与空白丢弃。"""
    return [t.lower() for t in _TOKEN_RE.findall(str(text or ""))]


def _lcs(x: List[str], y: List[str]) -> int:
    """最长公共子序列长度。滚动两行,内存 O(min(m,n)) 而非 O(m·n)。"""
    if not x or not y:
        return 0
    if len(x) < len(y):
        x, y = y, x
    prev = [0] * (len(y) + 1)
    for xi in x:
        cur = [0] * (len(y) + 1)
        for j, yj in enumerate(y, 1):
            cur[j] = prev[j - 1] + 1 if xi == yj else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l_f1(hyp: str, ref: str) -> float:
    h, r = tokenize(hyp), tokenize(ref)
    if not h or not r:
        return 0.0
    lcs = _lcs(h, r)
    prec = lcs / len(h)
    rec = lcs / len(r)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# =========================================================================
# 验证者作答(全类型)
# =========================================================================
def _responses_by_qid(responses: List[Dict]) -> Dict[str, List[Dict]]:
    by_q: Dict[str, List[Dict]] = {}
    for r in responses or []:
        qid = (r or {}).get("qid")
        if qid:
            by_q.setdefault(qid, []).append(r)
    return by_q


def interval_stats(annotations: List[Dict], responses: List[Dict],
                   threshold: float = 0.5, pass_ratio_min: float = 0.9) -> Dict:
    """temporal_grounding:验证者 [s,e] vs temporal_target 的 mIoU(Res 8.2)。

    单题取【全部验证者的平均 IoU】。此前取 max,等于"只要一个人对就算过",
    会系统性高估合格率(D18)。
    """
    key = {q["qid"]: q for ann in annotations for q in ann.get("qa", [])
           if q.get("task_type") == "temporal_grounding" and q.get("temporal_target")}
    per_q: List[Dict] = []
    by_q = _responses_by_qid(responses)
    for qid, q in key.items():
        target = tuple(q["temporal_target"])
        ious = []
        for r in by_q.get(qid, []):
            iv = r.get("interval")
            if iv and len(iv) == 2 and iv[1] > iv[0]:
                ious.append(miou(tuple(iv), target))
        if ious:
            mean_iou = sum(ious) / len(ious)
            per_q.append({"qid": qid, "miou": round(mean_iou, 3),
                          "n_raters": len(ious),
                          "per_rater": [round(v, 3) for v in ious],
                          "pass": mean_iou >= threshold})
    n = len(per_q)
    ratio = round(sum(1 for p in per_q if p["pass"]) / n, 3) if n else float("nan")
    log.info("temporal mIoU:合格率 %s(%d 题)", ratio, n)
    return {"n": n, "pass_ratio": ratio, "threshold": threshold, "per_q": per_q,
            "pass": bool(n and ratio >= pass_ratio_min)}


def open_stats(annotations: List[Dict], responses: List[Dict],
               client=None, sim_threshold: float = 0.5,
               agree_threshold: float = 0.6) -> Dict:
    """open/summary:验证者文本 vs 参考答案 ROUGE-L/embedding 一致率(Res 8.2)。

    两个阈值语义不同,此前共用一个参数(B5):
      sim_threshold   —— 单条作答与参考答案"算一致"的相似度线;
      agree_threshold —— 该题中一致作答的占比达到多少才算这道题合格。
    """
    key = {q["qid"]: q for ann in annotations for q in ann.get("qa", [])
           if q.get("task_type") in ("open", "summary") and q.get("reference_answer")}
    by_q = _responses_by_qid(responses)
    per_q: List[Dict] = []
    for qid, q in key.items():
        ref = (q["reference_answer"].get("zh") or q["reference_answer"].get("en") or "")
        agreed = 0
        tot = 0
        for r in by_q.get(qid, []):
            text = r.get("text")
            if not text:
                continue
            tot += 1
            rl = rouge_l_f1(text, ref)
            em = text_similarity(client, text, ref) if client is not None else rl
            if max(rl, em) >= sim_threshold:
                agreed += 1
        if tot:
            per_q.append({"qid": qid, "agreement": round(agreed / tot, 3), "n_raters": tot,
                          "pass": agreed / tot >= agree_threshold})
    n = len(per_q)
    ratio = round(sum(1 for p in per_q if p["pass"]) / n, 3) if n else float("nan")
    log.info("open/summary 一致率:%.0f%% 合格(%d 题)", 100 * ratio if n else 0, n)
    return {"n": n, "agree_ratio": ratio, "sim_threshold": sim_threshold,
            "agree_threshold": agree_threshold, "per_q": per_q,
            "pass": bool(n and ratio >= agree_threshold)}


def mcq_stats(annotations: List[Dict], responses: List[Dict],
              alpha_threshold: float = 0.8, gold_threshold: float = 0.9) -> Dict:
    """MCQ:α + 黄金题(复用 qc.py 纯标准库实现)。"""
    from .qc import build_keys, rework_candidates
    answer_key, gold_key = build_keys(annotations)
    per_q: Dict[str, List[str]] = {}
    for r in responses or []:
        qid = (r or {}).get("qid")
        if qid in answer_key and qid not in gold_key and r.get("answer"):
            per_q.setdefault(qid, []).append(r["answer"])
    units = [v for v in per_q.values() if len(v) >= 2]
    alpha = krippendorff_alpha(units) if units else float("nan")
    gold = grade_gold(responses, gold_key, gold_threshold)
    return {
        "krippendorff_alpha": alpha,
        "n_items": len(units),
        "alpha_pass": bool(alpha == alpha and alpha >= alpha_threshold),
        "gold": gold,
        "flagged_annotators": [a for a, v in gold.items() if v.get("flag")],
        "rework": rework_candidates(responses, {k: v for k, v in answer_key.items()
                                                if k not in gold_key}),
    }


def pipeline_temporal_ratio(qa: List[Dict], threshold: float = 0.5) -> float:
    """S6 流水线内区间核验的合格率(qa_verify 写入 q.verification.miou)。

    无任何可用核验结果时返回 nan,由调用方决定拦截还是标为未评估。
    """
    vals = []
    for q in qa or []:
        if q.get("task_type") != "temporal_grounding":
            continue
        v = q.get("verification") or {}
        m = v.get("miou")
        if isinstance(m, (int, float)) and m == m:
            vals.append(float(m))
    if not vals:
        return float("nan")
    return round(sum(1 for m in vals if m >= threshold) / len(vals), 3)


# =========================================================================
# 视频级发布门禁(Res 8.2 门槛清单)
# =========================================================================
def check_video_gates(annotation: Dict, meta_: Optional[Dict] = None,
                      alpha: float = float("nan"),
                      miou_pass_ratio: Optional[float] = None,
                      open_agree: Optional[float] = None,
                      cfg: Optional[RunConfig] = None) -> Dict:
    """视频级门槛;返回 {pass, reasons[], gates{...}, unevaluated[]}。

    覆盖/事件/QA/双语 等可离线计算的指标在此判定;α、mIoU、open 一致率
    由 run_qc 按【本视频自己的作答样本】计算后注入。
    """
    cfg = cfg or RunConfig()
    am = annotation.get("annotation_meta", {})
    reasons: List[str] = []
    unevaluated: List[str] = []
    gates: Dict[str, bool] = {}

    # 1) 覆盖率 100% 无 gaps
    gaps = am.get("gaps") or []
    gates["coverage"] = len(gaps) == 0
    if not gates["coverage"]:
        reasons.append(f"存在未覆盖区间 gaps: {len(gaps)} 个(含失败窗口)")

    # 2) 事件密度 ≥45/h + key 事件 100% 有效 span
    dur = float((annotation.get("meta") or {}).get("duration_sec", 0) or 0)
    events = annotation.get("structure", {}).get("events") or []
    hours = max(dur / 3600.0, 1e-6)
    density = len(events) / hours
    gates["event_density"] = density >= cfg.event_density_min
    if not gates["event_density"]:
        reasons.append(f"事件密度 {density:.1f}/h < {cfg.event_density_min}/h")
    key_events = [e for e in events if e.get("importance") == "key"]
    bad_key = [e for e in key_events
               if not e.get("span") or len(e["span"]) != 2
               or not (0 <= e["span"][0] < e["span"][1] <= max(dur, 1e-9))]
    gates["key_span"] = len(bad_key) == 0
    if not gates["key_span"]:
        reasons.append(f"{len(bad_key)} 个 key 事件 span 无效")

    # 3) 描述:shot.desc 100% 非空;segment 门通过率(quiet_segments 为空)
    shots = annotation.get("structure", {}).get("shots") or []
    empty_shots = [s for s in shots if not (s.get("desc") or {}).get("zh")]
    # 空 shots 不等于"零个镜头缺描述",而是镜头分割整体缺失(预处理失败时会静默
    # 写出 shots: [])。此前 len([]) == 0 让本门自动通过,预处理垮掉的视频反而"合格"。
    gates["shot_desc"] = bool(shots) and not empty_shots
    if not shots:
        reasons.append("structure.shots 为空(镜头分割缺失),镜头级描述无法判定")
    elif empty_shots:
        reasons.append(f"{len(empty_shots)} 个镜头缺 shot.desc")

    segments = annotation.get("structure", {}).get("segments") or []
    quiet = am.get("stats", {}).get("quiet_segments") or []
    # 同理:--no-describe 或 S2 整体失败时 segments 与 quiet_segments 同为空,
    # 只看 quiet_segments 会把"根本没产出详述"判成通过。
    gates["segment_gate"] = bool(segments) and not quiet
    if not segments:
        reasons.append("structure.segments 为空(详述阶段未产出),片段级描述无法判定")
    elif quiet:
        reasons.append(f"{len(quiet)} 个片段未过质量门(quiet_segments)")

    # 4) QA:五通道后 ≥8 题
    qa = annotation.get("qa") or []
    gates["qa_count"] = len(qa) >= cfg.min_qa_after_filter
    if not gates["qa_count"]:
        reasons.append(f"过滤后题目 {len(qa)} < {cfg.min_qa_after_filter}")

    # 5) 双语完备率 ≥95%(question 双语)
    n_bil = sum(1 for q in qa
                if (q.get("question") or {}).get("zh") and (q.get("question") or {}).get("en"))
    bil_ratio = n_bil / len(qa) if qa else 0.0
    gates["bilingual"] = bool(qa) and bil_ratio >= cfg.bilingual_min_ratio
    if not gates["bilingual"]:
        reasons.append(f"双语完备率 {100 * bil_ratio:.1f}% < {100 * cfg.bilingual_min_ratio:.0f}%")

    # 6) 能力矩阵(Res 5.2):S5 落在 stats.qa_coverage,矩阵不过即拦截发布
    coverage = (am.get("stats") or {}).get("qa_coverage")
    if isinstance(coverage, dict):
        gates["qa_coverage"] = bool(coverage.get("pass"))
        if not gates["qa_coverage"]:
            failed_checks = [k for k, v in (coverage.get("checks") or {}).items() if not v]
            reasons.append(f"能力矩阵未达标: {failed_checks or '未知'}")
    else:
        gates["qa_coverage"] = False
        reasons.append("缺 stats.qa_coverage,无法判定能力矩阵(请重跑 S5)")

    # 7) 交互指标注入:α / mIoU / open 一致率
    # 无样本(未提供 responses / 验证者不足 2 人)→ nan,视为"未评估"而非"不合格",
    # 否则任何没有人工作答的批次都会被全部拒绝(B7)。三个指标口径保持一致。
    def _na(v):
        return v is None or (isinstance(v, float) and v != v)

    gates["alpha"] = _na(alpha) or (alpha >= cfg.alpha_threshold)
    if not gates["alpha"]:
        reasons.append(f"Krippendorff α = {alpha:.3f} < {cfg.alpha_threshold}")
    elif _na(alpha):
        unevaluated.append("MCQ 一致性 α(无 ≥2 名验证者的作答样本)")

    n_temporal = sum(1 for q in qa
                     if q.get("task_type") == "temporal_grounding" and q.get("temporal_target"))
    if _na(miou_pass_ratio) and n_temporal:
        # 没有外部作答样本时,回退到 S6 流水线内的区间核验(qa_verify);
        # 两者都没有 → 拦截,不允许未经任何区间验证的 temporal 题发布
        miou_pass_ratio = pipeline_temporal_ratio(qa, cfg.miou_threshold)
        if _na(miou_pass_ratio):
            gates["miou"] = False
            reasons.append(f"{n_temporal} 道 temporal 题既无验证者区间作答,"
                           f"也无流水线区间核验结果(S6),无法判定 mIoU")
        else:
            gates["miou"] = miou_pass_ratio >= cfg.miou_pass_ratio
            if not gates["miou"]:
                reasons.append(f"流水线 mIoU 合格率 {100 * miou_pass_ratio:.0f}% "
                               f"< {100 * cfg.miou_pass_ratio:.0f}%")
    else:
        gates["miou"] = _na(miou_pass_ratio) or (miou_pass_ratio >= cfg.miou_pass_ratio)
        if not gates["miou"]:
            reasons.append(f"mIoU 样本合格率 {100 * miou_pass_ratio:.0f}% "
                           f"< {100 * cfg.miou_pass_ratio:.0f}%")
        elif _na(miou_pass_ratio):
            unevaluated.append("temporal mIoU(本视频无 temporal 题)")

    gates["open_agree"] = _na(open_agree) or (open_agree >= cfg.open_agree_threshold)
    if not gates["open_agree"]:
        reasons.append(f"open/summary 一致率 {100 * open_agree:.0f}% "
                       f"< {100 * cfg.open_agree_threshold:.0f}%")
    elif _na(open_agree):
        unevaluated.append("open/summary 一致率(无验证者自由作答)")

    # 8) 阶段状态:核验(S6)/ 过滤(S7)失败不会体现在上面任何一项指标上,
    # 只要有阶段 failed 就说明产物不完整,即便其余指标凑巧达标也不得发布。
    failed_stages = sorted(k for k, v in (am.get("stages") or {}).items()
                           if (v or {}).get("state") == "failed")
    gates["stages"] = not failed_stages
    if failed_stages:
        reasons.append(f"存在未成功的阶段: {failed_stages}")

    # 9) schema 合法性:落盘时已校验并记录结果,发布环节不能再放行不合法产物
    gates["schema"] = am.get("schema_valid") is not False
    if not gates["schema"]:
        reasons.append(f"产物未通过 schema 校验: {am.get('schema_error') or '(无详情)'}")

    all_pass = all(gates.values())
    gates["all"] = all_pass
    return {"pass": all_pass, "reasons": reasons, "gates": gates,
            "unevaluated": unevaluated}


def human_todo(annotation: Dict) -> List[Dict]:
    """人工清单(Res 8.3):未复核关键事件 / 失败段 / split·rejected 边界 / 无效因果边,
    每个附 span 供人工在工单中修改。

    与 annotation_meta.human_todo(流水线阶段已写入的条目)按 (kind, event_id/span)
    去重,避免同一件事被计两次。
    """
    out: List[Dict] = []
    seen = set()

    def _add(item: Dict) -> None:
        key = (item.get("kind"), item.get("event_id"),
               tuple(item.get("span") or ()) or None)
        if key in seen:
            return
        seen.add(key)
        out.append(item)

    for t in annotation.get("annotation_meta", {}).get("human_todo") or []:
        _add(dict(t))

    events = annotation.get("structure", {}).get("events") or []
    for e in events:
        if e.get("importance") == "key" and e.get("boundary_state") == "unreviewed":
            _add({"kind": "key_event_unreviewed", "event_id": e.get("id"),
                  "span": e.get("span"),
                  "desc": (e.get("desc") or {}).get("zh") or (e.get("desc") or {}).get("en")})
        if e.get("boundary_state") in ("split", "rejected_final"):
            rr = e.get("refine_reason")
            reason = rr if isinstance(rr, dict) else {"zh": rr if isinstance(rr, str) else "", "en": ""}
            _add({"kind": f"boundary_{e.get('boundary_state')}",
                  "event_id": e.get("id"), "span": e.get("span"),
                  "reason": reason.get("zh") or ""})
    for g in annotation.get("annotation_meta", {}).get("gaps") or []:
        _add({"kind": "uncovered_gap", "span": [g.get("start"), g.get("end")],
              "reason": g.get("reason")})
    return out


# =========================================================================
# 汇总
# =========================================================================
def run_qc_v2(annotations: List[Dict], responses: List[Dict], cfg: RunConfig,
              client=None) -> Dict:
    """QC v2 汇总 + 视频级发布门禁;返回完整报告。

    批次级指标仍然汇总输出,但【门禁按每个视频自己的作答样本判定】——
    此前把整批的 α/mIoU/open 注入每个视频,一个视频数据差会拖垮全批(B7)。
    """
    mcq = mcq_stats(annotations, responses, cfg.alpha_threshold, cfg.gold_threshold)
    iv = interval_stats(annotations, responses, cfg.miou_threshold, cfg.miou_pass_ratio)
    op = open_stats(annotations, responses, client,
                    sim_threshold=cfg.open_sim_threshold,
                    agree_threshold=cfg.open_agree_threshold)

    per_video: List[Dict] = []
    all_pass = True
    for ann in annotations:
        vid = ann.get("video_id", "?")
        qids = {q.get("qid") for q in ann.get("qa", [])}
        sub = [r for r in responses if r.get("qid") in qids]

        v_mcq = mcq_stats([ann], sub, cfg.alpha_threshold, cfg.gold_threshold)
        v_iv = interval_stats([ann], sub, cfg.miou_threshold, cfg.miou_pass_ratio)
        v_op = open_stats([ann], sub, client, sim_threshold=cfg.open_sim_threshold,
                          agree_threshold=cfg.open_agree_threshold)

        gates = check_video_gates(ann, cfg=cfg,
                                  alpha=v_mcq["krippendorff_alpha"],
                                  miou_pass_ratio=v_iv["pass_ratio"],
                                  open_agree=v_op["agree_ratio"])
        # 门禁:不达标 → review_status 保持 draft 并列出拒绝原因(Res 8.2)
        am = ann.setdefault("annotation_meta", {})
        if gates["pass"]:
            am["review_status"] = "verified"
            am.pop("release_rejected_reasons", None)
        else:
            am["review_status"] = "draft"
            am["release_rejected_reasons"] = gates["reasons"]
        if gates.get("unevaluated"):
            am["release_unevaluated"] = gates["unevaluated"]
        per_video.append({"video_id": vid, **gates,
                          "alpha": v_mcq["krippendorff_alpha"],
                          "miou_pass_ratio": v_iv["pass_ratio"],
                          "open_agree_ratio": v_op["agree_ratio"]})
        all_pass = all_pass and gates["pass"]

    return {
        "mcq": mcq,
        "temporal_miou": iv,
        "open_summary": op,
        "per_video": per_video,
        "all_gates_pass": all_pass,
        "human_todo_total": sum(len(human_todo(a)) for a in annotations),
    }
