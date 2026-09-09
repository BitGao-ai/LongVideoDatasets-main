# -*- coding: utf-8 -*-
"""抗捷径过滤 v2(Res 6,修复 REVIEW D1)—— 五通道 + 非 MCQ 捷径。

通道清单(Res 6.2):
| 盲答    | 题干+选项(纯文本)答对                              | 一律剔除 |
| 字幕    | 字幕文本答对(视觉类)                                | 剔除(音频原件题除外) |
| 单帧    | 证据区 1 帧答对(视觉类)      ★修复:以前从不剔除      | 剔除(--keep-single-frame 可保留) |
| 梗概    | 题干+梗概答对                                       | 剔除/重写 |
| 语言先验 | 选项文本可从常识/语言模板全等排除                  | 剔除 |
非 MCQ(Res 6.3):
| temporal_grounding | 只给字幕+题面定位区间 IoU ≥0.6 → 打回 |
| open/summary       | 文本盲答与 reference embedding 余弦 ≥0.85 → 剔除 |

任一通道命中 → 剔除;通道判定结果统一写入 anti_shortcut 留痕。
"""

import datetime
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from . import prompts
from .config import RunConfig
from .frames_store import FrameStore
from .llm_client import LLMClient, image_part, text_part
from .media import one_frame, subs_text
from .utils import miou, text_similarity

log = logging.getLogger("annotator.shortcut")

# 这些能力本就以字幕/语音为证据,字幕能答不算作弊,不因 subtitle_only 而剔除
_SUBTITLE_OK_CAPS = {"L1_audio_asr", "L1_ocr", "L4_subtitle_visual", "L4_narration_align"}

# 语言先验模板(选项文本全等模板,答案可从语言排除)
_PRIOR_TEMPLATES = ("以上都不对", "以上都对", "以上均不正确", "以上均正确",
                    "none of the above", "all of the above", "not listed",
                    "不是以上任何一项", "无法确定", "cannot be determined")

_CJK_RE = re.compile(r"[一-鿿]")


def _ask(client: LLMClient, user_content, vision: bool) -> Optional[str]:
    try:
        out = client.complete(
            [{"role": "system", "content": prompts.ANSWER_SYS},
             {"role": "user", "content": user_content}],
            vision=vision, temperature=0.0,
        )
        ans = (out or {}).get("answer")
        return ans if ans in {"A", "B", "C", "D"} else None
    except Exception as e:  # noqa: BLE001
        log.warning("捷径作答调用失败: %s", e)
        return None


def _evidence_mid(q: Dict) -> Optional[float]:
    spans = q.get("evidence_spans") or []
    if not spans:
        return None
    s, e = spans[0]
    return (s + e) / 2.0


def _evidence_subs(q: Dict, structure: Dict) -> str:
    texts = []
    for span in q.get("evidence_spans", []):
        texts.append(subs_text(structure.get("subtitles", []), span[0], span[1]))
    return "\n".join(t for t in texts if t)


def _is_visual(q: Dict) -> bool:
    return not any(c in _SUBTITLE_OK_CAPS for c in q.get("capability", []))


def _opt_text(o: Dict) -> str:
    """安全取选项文本(LLM 可能少给 text / 给成字符串)。"""
    t = (o or {}).get("text")
    if isinstance(t, str):
        return t.strip()
    if isinstance(t, dict):
        return (t.get("zh") or "").strip() or (t.get("en") or "").strip()
    return ""


def _normalize_option(t: str) -> str:
    """去掉首尾标点空白,便于与语言模板做等值比较(D16)。"""
    return re.sub(r"^[\s　]*|[\s　。.,，;;!!??、]*$", "", t).lower()


def _language_prior(q: Dict) -> Tuple[bool, str]:
    """语言先验通道(Res 6.2):选项全等模板 / 选项文本原样出现在题干。"""
    opts = q.get("options") or []
    if len(opts) != 4:
        return False, ""
    texts = [_opt_text(o) for o in opts]
    templates = {_normalize_option(t) for t in _PRIOR_TEMPLATES}
    for t in texts:
        if _normalize_option(t) in templates:
            return True, f"选项含语言模板:{t}"
    q_zh = (q.get("question", {}).get("zh") or "").lower()
    q_en = (q.get("question", {}).get("en") or "").lower()
    for t in texts:
        tl = t.lower()
        # 中文 4 字短语在长题干里撞车概率高,阈值抬到 6;英文仍用 4
        min_len = 6 if _CJK_RE.search(tl) else 4
        if len(tl) >= min_len and (tl in q_zh or tl in q_en):
            return True, f"选项文本原样出现在题干:{t[:30]}"
    return False, ""


def _temporal_subtitle_locate(client: LLMClient, q: Dict, sub_text: str,
                              cfg: RunConfig) -> Tuple[bool, Optional[list]]:
    """temporal 字幕定位捷径(Res 6.3):只给字幕定位,与标准区间 IoU ≥ 阈值。"""
    if not sub_text or not q.get("temporal_target"):
        return False, None
    try:
        out = client.complete(
            [{"role": "system", "content": prompts.ANSWER_SYS},
             {"role": "user", "content": prompts.temporal_locate_user(q, sub_text)}],
            vision=False, temperature=0.0)
        iv = (out or {}).get("interval")
        if iv and len(iv) == 2 and iv[1] > iv[0]:
            return miou(tuple(iv), tuple(q["temporal_target"])) >= cfg.temporal_shortcut_iou, iv
    except Exception as e:  # noqa: BLE001
        log.debug("temporal 字幕定位调用失败: %s", e)
    return False, None


def _open_blind_similar(client: LLMClient, q: Dict, cfg: RunConfig) -> Tuple[bool, float]:
    """open/summary 盲答相似度(Res 6.3):盲答文本与 reference embedding ≥ 阈值。"""
    ref = (q.get("reference_answer") or {}).get("zh") or \
          (q.get("reference_answer") or {}).get("en") or ""
    if not ref:
        return False, 0.0
    try:
        out = client.complete(
            [{"role": "system", "content": prompts.OPEN_BLIND_SYS},
             {"role": "user", "content": prompts.open_blind_user(q)}],
            vision=False, temperature=0.0)
        ans = (out or {}).get("answer") or ""
        if not ans.strip():
            return False, 0.0
        sim = text_similarity(client, ans, ref)
        return sim >= cfg.open_shortcut_sim, round(sim, 3)
    except Exception as e:  # noqa: BLE001
        log.debug("open 盲答调用失败: %s", e)
        return False, 0.0


def check_one(client: LLMClient, cfg: RunConfig, video_path: str,
              q: Dict, structure: Dict, store: Optional[FrameStore] = None,
              synopsis: Optional[str] = None) -> Dict:
    """对单题跑全部捷径通道,返回判定字典(含留痕 note)。非 mcq 走对应专用通道。"""
    tt = q.get("task_type")
    base: Dict = {"checked_note": [],
                  "blind_llm_pass": False, "single_frame_pass": False,
                  "subtitle_only_pass": False, "synopsis_pass": False,
                  "language_prior_pass": False}

    if tt == "mcq":
        gold = q.get("answer")
        if gold not in {"A", "B", "C", "D"}:
            return base
        visual = _is_visual(q)

        # --- 盲答:所有 MCQ 一律跑 ---
        base["blind_llm_pass"] = (_ask(client, prompts.blind_user(q), vision=False) == gold)

        # --- 字幕 / 单帧:仅视觉题(音频·字幕原生题以字幕为证据,不算作弊)---
        if visual:
            sub_text = _evidence_subs(q, structure)
            if sub_text:
                base["subtitle_only_pass"] = (_ask(
                    client, prompts.subtitle_user(q, sub_text), vision=False) == gold)
            mid = _evidence_mid(q)
            if mid is not None:
                # 取帧失败(无帧索引 / 解码错误)只让本通道失效,不能连带把同一题的
                # 盲答/梗概/语言先验判定一起丢掉
                try:
                    b64 = one_frame(video_path, mid, cfg.max_image_edge, store=store)
                except Exception as e:  # noqa: BLE001
                    log.warning("单帧通道取帧失败(%s),该通道跳过: %s", q.get("qid"), e)
                    b64 = None
                    base["checked_note"].append(f"single_frame_skipped: {e}")
                if b64:
                    content = [text_part(prompts.frame_user(q)), image_part(b64)]
                    base["single_frame_pass"] = (_ask(client, content, vision=True) == gold)

        # --- 梗概 / 语言先验:与题目模态无关,所有 MCQ 都要跑(修复 B4)---
        # 此前这两个通道被写在 `if _is_visual(q):` 内部,凡是 L1_audio_asr /
        # L1_ocr / L4_* 的题永远不检测,而 README 只给字幕通道声明了豁免。
        if cfg.synopsis_leak_check and synopsis:
            try:
                out = client.complete(
                    [{"role": "system", "content": prompts.SYNOPSIS_SYS},
                     {"role": "user", "content": prompts.synopsis_user(q, synopsis)}],
                    vision=False)
                base["synopsis_pass"] = ((out or {}).get("answer") == gold)
            except Exception as e:  # noqa: BLE001
                log.debug("梗概通道调用失败: %s", e)
        if cfg.language_prior_check:
            hit, note = _language_prior(q)
            base["language_prior_pass"] = hit
            if hit:
                base["checked_note"].append(f"language_prior: {note}")

    elif tt == "temporal_grounding":
        hit, iv = _temporal_subtitle_locate(client, q, _evidence_subs(q, structure), cfg)
        base["subtitle_locate_pass"] = hit
        if hit:
            base["checked_note"].append(f"subtitle_locate: {iv}")
    elif tt in ("open", "summary"):
        hit, sim = _open_blind_similar(client, q, cfg)
        base["blind_similar_pass"] = hit
        if hit:
            base["checked_note"].append(f"blind_similar: {sim}")
    return base


def should_drop(cfg: RunConfig, q: Dict) -> bool:
    """按剔除策略判断(Res 6.2):
    盲答一律剔;字幕命中且为视觉题剔;单帧命中且为视觉题剔(★D1 修复,
    --keep-single-frame 可保留);梗概/语言先验命中剔;非 MCQ 命中剔。
    """
    if not cfg.drop_shortcut:
        return False
    a = q.get("anti_shortcut", {})
    if a.get("blind_llm_pass"):
        return True
    if a.get("synopsis_pass"):
        return True
    if a.get("language_prior_pass"):
        return True
    if a.get("subtitle_locate_pass") or a.get("blind_similar_pass"):
        return True
    if a.get("subtitle_only_pass") and _is_visual(q):
        return True
    if a.get("single_frame_pass") and _is_visual(q) and not cfg.keep_single_frame:
        return True
    return False


def renumber(qa: List[Dict]) -> List[Dict]:
    """按当前顺序重排 qid 后缀,保证连续。"""
    for i, q in enumerate(qa, 1):
        q["qid"] = f"{q['qid'].rsplit('_q', 1)[0]}_q{i:02d}"
    return qa


def run_filters(client: LLMClient, cfg: RunConfig, video_path: str,
                qa: List[Dict], structure: Dict, store: Optional[FrameStore] = None,
                synopsis: Optional[str] = None) -> Tuple[List[Dict], Dict]:
    """五通道过滤主入口;返回 (保留题目, 统计)。

    每题的通道判定彼此独立,按题并发(C7);保留顺序仍按原 qa 顺序,
    保证 renumber 后的 qid 稳定可复现。
    """
    today = datetime.date.today().isoformat()
    checker = f"auto:{client.pc.name}"
    stats = {"total": len(qa), "kept": 0, "dropped": 0, "by_channel": {}}
    if not qa:
        return [], stats

    workers = max(1, min(cfg.workers, len(qa)))
    results: List[Optional[Dict]] = [None] * len(qa)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(check_one, client, cfg, video_path, q, structure,
                            store=store, synopsis=synopsis): i
                for i, q in enumerate(qa)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001 —— 单题失败不拖垮整个阶段
                log.warning("题目 %s 捷径检测异常,按未命中处理: %s", qa[i].get("qid"), e)
                results[i] = {"checked_note": [f"check_failed: {e}"]}

    kept: List[Dict] = []
    for q, res in zip(qa, results):
        res = res or {}
        q["anti_shortcut"] = {**res, "checked_by": checker, "checked_date": today}
        if should_drop(cfg, q):
            stats["dropped"] += 1
            hit = sorted(k for k, v in res.items() if v is True and k.endswith("_pass"))
            for ch in hit or ["unknown"]:
                stats["by_channel"][ch] = stats["by_channel"].get(ch, 0) + 1
            log.info("剔除捷径题 %s (命中通道=%s)", q.get("qid"), hit or "unknown")
        else:
            kept.append(q)

    renumber(kept)
    stats["kept"] = len(kept)
    log.info("抗捷径过滤:保留 %d / %d 题(%d 并发);剔除分布 %s",
             len(kept), len(qa), workers, stats["by_channel"])
    return kept, stats
