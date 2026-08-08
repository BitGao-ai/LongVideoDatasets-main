# -*- coding: utf-8 -*-
"""抗捷径过滤:对每道 MCQ 跑三种“捷径作答”,命中则标记(可选剔除)。

- blind_llm_pass : 只给题干+选项,纯文本模型能答对 -> 语言先验作弊
- subtitle_only_pass: 只给证据区间字幕文本,能答对 -> 字幕捷径
- single_frame_pass: 只给证据中点 1 帧,能答对 -> 单帧捷径(非长程)
"""

import datetime
import logging
from typing import Dict, List, Optional

from . import prompts
from .config import RunConfig
from .llm_client import LLMClient, image_part, text_part
from .media import one_frame, subs_text

log = logging.getLogger("annotator.filter")

# 这些能力本就以字幕/语音为证据,字幕能答不算作弊,不因 subtitle_only 而剔除
_SUBTITLE_OK_CAPS = {"L1_audio_asr", "L1_ocr", "L4_subtitle_visual", "L4_narration_align"}


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


def check_one(client: LLMClient, cfg: RunConfig, video_path: str,
              q: Dict, structure: Dict) -> Dict[str, bool]:
    """对单题跑三种“捷径作答”,返回三个布尔标记(不含 checked_by/date)。

    非 mcq 或答案缺失时返回全 False。可被换模型复筛(rescreen)复用。
    """
    res = {"blind_llm_pass": False, "single_frame_pass": False, "subtitle_only_pass": False}
    if q.get("task_type") != "mcq":
        return res
    gold = q.get("answer")
    if gold not in {"A", "B", "C", "D"}:
        return res

    # 1) 盲答(纯文本)
    res["blind_llm_pass"] = (_ask(client, prompts.blind_user(q), vision=False) == gold)

    # 2) 字幕捷径
    sub_text = _evidence_subs(q, structure)
    if sub_text:
        res["subtitle_only_pass"] = (_ask(client, prompts.subtitle_user(q, sub_text), vision=False) == gold)

    # 3) 单帧捷径
    mid = _evidence_mid(q)
    if mid is not None:
        b64 = one_frame(video_path, mid, cfg.max_image_edge)
        if b64:
            content = [text_part(prompts.frame_user(q)), image_part(b64)]
            res["single_frame_pass"] = (_ask(client, content, vision=True) == gold)
    return res


def should_drop(cfg: RunConfig, q: Dict) -> bool:
    """按剔除策略判断:盲答命中一律剔;字幕捷径命中且为视觉题剔。"""
    if not cfg.drop_shortcut:
        return False
    a = q.get("anti_shortcut", {})
    if a.get("blind_llm_pass"):
        return True
    if a.get("subtitle_only_pass") and _is_visual(q):
        return True
    return False


def renumber(qa: List[Dict]) -> List[Dict]:
    """按当前顺序重排 qid 后缀,保证连续。"""
    for i, q in enumerate(qa, 1):
        q["qid"] = f"{q['qid'].rsplit('_q', 1)[0]}_q{i:02d}"
    return qa


def run_filters(client: LLMClient, cfg: RunConfig, video_path: str,
                qa: List[Dict], structure: Dict) -> List[Dict]:
    today = datetime.date.today().isoformat()
    checker = f"auto:{client.pc.name}"
    kept: List[Dict] = []

    for q in qa:
        res = check_one(client, cfg, video_path, q, structure)
        q["anti_shortcut"] = {**res, "checked_by": checker, "checked_date": today}
        if q.get("task_type") == "mcq" and should_drop(cfg, q):
            log.info("剔除捷径题 %s (blind=%s, sub=%s)", q.get("qid"),
                     res["blind_llm_pass"], res["subtitle_only_pass"])
        else:
            kept.append(q)

    renumber(kept)
    log.info("抗捷径过滤:保留 %d / %d 题", len(kept), len(qa))
    return kept
