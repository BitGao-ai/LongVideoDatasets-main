# -*- coding: utf-8 -*-
"""QA 答案核验 M4.5(Res 5.3/5.4)。

1) 双模型(VIDEO)核验:出题后,用【另一家供应商】对每道 mcq 看真实证据画面
   (证据区间采样帧,Res 5.3 “真实视频切片 ±2s”)独立作答
   → {correct_answer, confidence, reasons};
   与生成答案不一致 → 打回重写 ≤verify_max_regen 次;报告 verify_agreement。
2) 梗概泄漏防护(Res 5.4):题干+梗概 纯文本可答 → 剔除并记 quality_flag。
3) temporal_grounding 区间核验(Res 8.2):验证者只看 target 前后的帧马赛克独立定位,
   与 temporal_target 算 mIoU;低于 miou_threshold 打 quality_flag 并由发布门禁拦截。

核验结果写入 q.verification {provider, agreement, pass};KPI 为 verify_agreement。
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from . import prompts
from .config import RunConfig
from .frames_store import FrameStore
from .llm_client import LLMClient, image_part, text_part
from .utils import asym_pad, clamp, miou

log = logging.getLogger("annotator.qa_verify")

_FRAME_PER_QUESTION = 6


def _evidence_frames(q: Dict, store: FrameStore, cfg: RunConfig,
                     max_frames: int = _FRAME_PER_QUESTION) -> List[Dict]:
    """证据区间采样帧(每 span 均匀取,≤max_frames 张)。"""
    frames: List[Dict] = []
    for span in (q.get("evidence_spans") or [])[:2]:
        times = store.sample_times(span[0], span[1], max_frames)
        for t in times[:max_frames]:
            b64 = store.frame_b64(t, cfg.max_image_edge)
            if b64:
                frames.append({"t": t, "b64": b64})
        if len(frames) >= max_frames:
            break
    return frames


def _ask_verifier(verifier: LLMClient, q: Dict, frames: List[Dict],
                  cfg: RunConfig) -> Optional[Dict]:
    """让验证者看真实证据帧独立作答。任何异常都收敛为 None(修复 D3)。

    此前本函数没有 try/except,且 `float(out["confidence"])` 遇到 "high" 这类
    字符串会 ValueError —— 一道题的异常会把整个 S6 拖垮并触发阶段级重跑。
    """
    if q.get("task_type") != "mcq":
        return None
    try:
        content = [text_part(prompts.verify_user(q, "以下画面为证据区间的真实采样帧(按时间顺序)。"))]
        content += [image_part(f["b64"]) for f in frames]
        out = verifier.complete(
            [{"role": "system", "content": prompts.VERIFY_SYS},
             {"role": "user", "content": content}],
            vision=True)
    except Exception as e:  # noqa: BLE001
        log.warning("验证者调用失败(%s): %s", q.get("qid"), e)
        return None
    out = out if isinstance(out, dict) else {}
    ans = out.get("answer")
    if ans not in {"A", "B", "C", "D"}:
        return None
    try:
        conf = float(out.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    return {"answer": ans, "confidence": conf,
            "reasons": out.get("reasons") or {"zh": "", "en": ""}}


def _verify_temporal(q: Dict, verifier: LLMClient, cfg: RunConfig,
                     store: FrameStore) -> Optional[Dict]:
    """temporal_grounding 区间核验:验证者只看马赛克独立定位,与标准区间算 mIoU。

    视野窗口总宽 = 标准区间 + 前后各一个区间宽度(下限 mosaic_pad_sec),但左右余量
    按题目内容哈希非对称拆分 —— 对称补边会让标准区间恒定居中,"答正中间那段"必得
    mIoU=1.0,mIoU 门就成了空转。
    """
    target = q.get("temporal_target") or []
    if len(target) != 2 or target[1] <= target[0]:
        return None
    s, e = float(target[0]), float(target[1])
    pad_lo, pad_hi = asym_pad([q.get("qid"), s, e], e - s, cfg.mosaic_pad_sec)
    packed = store.mosaic((s - pad_lo, e + pad_hi), k=cfg.mosaic_frames)
    if not packed:
        return None
    mosaic, tile_times, side = packed
    try:
        content = [text_part(prompts.temporal_verify_user(q, tile_times, side)),
                   image_part(mosaic)]
        out = verifier.complete(
            [{"role": "system", "content": prompts.TEMPORAL_VERIFY_SYS},
             {"role": "user", "content": content}],
            vision=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("temporal 区间核验调用失败(%s): %s", q.get("qid"), exc)
        return None
    out = out if isinstance(out, dict) else {}
    iv = out.get("interval")
    if not (isinstance(iv, (list, tuple)) and len(iv) == 2):
        return None
    try:
        a, b = float(iv[0]), float(iv[1])
    except (TypeError, ValueError):
        return None
    if b <= a:
        return None
    lo, hi = (tile_times[0], tile_times[-1]) if tile_times else (s, e)
    a, b = clamp(a, lo, hi), clamp(b, lo, hi)
    if b <= a:
        return None
    try:
        conf = float(out.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    return {"interval": [round(a, 3), round(b, 3)],
            "miou": round(miou((a, b), (s, e)), 3),
            "confidence": conf,
            "reasons": out.get("reasons") or {"zh": "", "en": ""}}


def _regen_answer(client: LLMClient, q: Dict, feedback: str) -> Optional[Dict]:
    try:
        out = client.complete(
            [{"role": "system", "content": prompts.QA_SYS},
             {"role": "user", "content": prompts.qa_rewrite_user(q, feedback)}],
            vision=False)
        raw = (out or {}).get("qa") or []
        return raw[0] if raw else None
    except Exception as e:  # noqa: BLE001
        log.warning("核验打回重写失败: %s", e)
        return None


def _verify_one(q: Dict, primary: LLMClient, verifier: Optional[LLMClient],
                cfg: RunConfig, store: Optional[FrameStore],
                synopsis: Optional[str]) -> Tuple[Optional[Dict], Dict]:
    """单题:梗概泄漏 → 双模型核验 → 打回重写。返回 (保留的题 or None, 计数)。"""
    cnt = {"verified": 0, "agreed": 0, "regen": 0, "synopsis_leak_dropped": 0,
           "mismatch": 0, "temporal_verified": 0, "temporal_pass": 0}

    # 梗概泄漏通道(Res 5.4):题干+梗概可答 → 剔除
    if cfg.synopsis_leak_check and synopsis and q.get("task_type") == "mcq":
        try:
            out = primary.complete(
                [{"role": "system", "content": prompts.SYNOPSIS_SYS},
                 {"role": "user", "content": prompts.synopsis_user(q, synopsis)}],
                vision=False)
            if isinstance(out, dict) and out.get("answer") == q.get("answer"):
                q.setdefault("quality_flags", []).append("synopsis_leak=1")
                cnt["synopsis_leak_dropped"] = 1
                log.info("梗概泄漏命中,剔除 %s", q.get("qid"))
                return None, cnt
        except Exception as e:  # noqa: BLE001
            log.debug("梗概检测调用失败: %s", e)

    if q.get("task_type") == "temporal_grounding":
        # 区间核验用验证者优先;无另一家 key 时退回主模型自检(强度较弱但可留痕)
        tv_client = verifier or primary
        if not cfg.verify_answers or not store or tv_client is None:
            q["verification"] = {"provider": None, "miou": None, "pass": None,
                                 "reason": "未开启核验或无帧索引"}
            return q, cnt
        verdict = _verify_temporal(q, tv_client, cfg, store)
        if verdict is None:
            q["verification"] = {"provider": tv_client.pc.name, "miou": None, "pass": None,
                                 "reason": "区间核验无有效返回"}
            return q, cnt
        passed = verdict["miou"] >= cfg.miou_threshold
        cnt["temporal_verified"] = 1
        cnt["temporal_pass"] = 1 if passed else 0
        q["verification"] = {"provider": tv_client.pc.name, "self_check": verifier is None,
                             "miou": verdict["miou"], "pass": passed,
                             "verifier_interval": verdict["interval"],
                             "confidence": verdict["confidence"]}
        if not passed:
            q.setdefault("quality_flags", []).append("temporal_miou_low=1")
            log.info("temporal 区间核验不合格 %s(mIoU=%.2f < %.2f)",
                     q.get("qid"), verdict["miou"], cfg.miou_threshold)
        return q, cnt

    if not cfg.verify_answers or verifier is None or q.get("task_type") != "mcq" or not store:
        q["verification"] = {"provider": None, "agreement": None, "pass": None}
        return q, cnt

    frames = _evidence_frames(q, store, cfg)
    if not frames:
        q["verification"] = {"provider": None, "agreement": None, "pass": None,
                             "reason": "无证据帧,跳过核验"}
        return q, cnt

    verdict = _ask_verifier(verifier, q, frames, cfg)
    if verdict is None:
        q["verification"] = {"provider": verifier.pc.name, "agreement": None, "pass": None,
                             "reason": "验证者调用失败"}
        return q, cnt

    gold = q.get("answer")
    agreed = verdict["answer"] == gold
    cnt["verified"] = 1
    if agreed:
        cnt["agreed"] = 1
        q["verification"] = {"provider": verifier.pc.name, "agreement": True, "pass": True,
                             "verifier": verdict["answer"],
                             "confidence": verdict["confidence"]}
        return q, cnt

    # 打回重写 ≤verify_max_regen 次(Res 5.3)
    for _ in range(cfg.verify_max_regen):
        regen = _regen_answer(primary, q,
                              f"答案核验不一致:验证者({verifier.pc.name})认为是 "
                              f"{verdict['answer']}(置信 {verdict['confidence']:.2f}),"
                              f"理由:{str(verdict['reasons'])[:200]}")
        if not (regen and regen.get("answer") in {"A", "B", "C", "D"}):
            continue
        q["question"] = regen.get("question", q.get("question"))
        q["options"] = regen.get("options", q.get("options"))
        q["answer"] = regen["answer"]
        cnt["regen"] += 1
        verdict2 = _ask_verifier(verifier, q, _evidence_frames(q, store, cfg), cfg)
        if verdict2 and verdict2["answer"] == q["answer"]:
            cnt["agreed"] = 1
            q["verification"] = {"provider": verifier.pc.name, "agreement": True,
                                 "pass": True, "verifier": verdict2["answer"],
                                 "confidence": verdict2["confidence"]}
            return q, cnt
        verdict = verdict2 or verdict

    # 两次重写仍不一致
    cnt["mismatch"] = 1
    q.setdefault("quality_flags", []).append("verify_mismatch=1")
    q["verification"] = {"provider": verifier.pc.name, "agreement": False, "pass": False,
                         "verifier": verdict.get("answer") if verdict else None,
                         "confidence": verdict.get("confidence") if verdict else None}
    if cfg.drop_verify_mismatch:
        # 另一家模型看着真实证据帧仍不认可这个答案 —— 默认剔除(修复 B13:
        # 此前只打 flag 照常入库,且发布门禁不检查该 flag,信号等于没用)
        log.info("核验不一致且重写无效,剔除 %s", q.get("qid"))
        return None, cnt
    return q, cnt


def verify_qa_answers(qa: List[Dict], primary: LLMClient, verifier: Optional[LLMClient],
                      cfg: RunConfig, store: Optional[FrameStore],
                      global_block: Optional[Dict] = None) -> Tuple[List[Dict], Dict]:
    """双模型核验 + 梗概泄漏检测。返回 (qa, stats)。

    verifier 为另一家供应商;None/缺 key 时跳过核验(verification.pass=None)。
    题目之间彼此独立,按题并发(C7);保留顺序与输入一致。
    """
    stats = {"verified": 0, "agreed": 0, "regen": 0, "verify_agreement": None,
             "synopsis_leak_dropped": 0, "verify_mismatch_dropped": 0,
             "temporal_verified": 0, "temporal_pass": 0, "temporal_miou_ratio": None}
    if not qa:
        return [], stats

    synopsis = None
    if global_block:
        syn = global_block.get("synopsis") or {}
        synopsis = syn.get("zh") or syn.get("en")

    workers = max(1, min(cfg.workers, len(qa)))
    results: List[Optional[Dict]] = [None] * len(qa)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_verify_one, q, primary, verifier, cfg, store, synopsis): i
                for i, q in enumerate(qa)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                kept_q, cnt = fut.result()
            except Exception as e:  # noqa: BLE001 —— 单题异常不拖垮整个阶段
                log.warning("题目 %s 核验异常,按未核验保留: %s", qa[i].get("qid"), e)
                qa[i]["verification"] = {"provider": None, "agreement": None,
                                         "pass": None, "reason": f"核验异常: {e}"}
                kept_q, cnt = qa[i], {}
            results[i] = kept_q
            for k in ("verified", "agreed", "regen", "synopsis_leak_dropped",
                      "temporal_verified", "temporal_pass"):
                stats[k] += cnt.get(k, 0)
            if cnt.get("mismatch") and kept_q is None:
                stats["verify_mismatch_dropped"] += 1

    kept = [q for q in results if q is not None]
    n_ver = stats["verified"]
    stats["verify_agreement"] = round(stats["agreed"] / n_ver, 3) if n_ver else None
    n_tv = stats["temporal_verified"]
    stats["temporal_miou_ratio"] = round(stats["temporal_pass"] / n_tv, 3) if n_tv else None
    log.info("双模型核验:%d 题(%d 并发),一致率 %s;temporal 区间核验 %d 题,合格率 %s;"
             "梗概泄漏剔除 %d;打回重写 %d 次;核验不一致剔除 %d",
             n_ver, workers, stats["verify_agreement"], n_tv,
             stats["temporal_miou_ratio"], stats["synopsis_leak_dropped"],
             stats["regen"], stats["verify_mismatch_dropped"])
    return kept, stats
