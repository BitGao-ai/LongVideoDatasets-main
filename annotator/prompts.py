# -*- coding: utf-8 -*-
"""双语提示词模板。

所有 LLM 阶段都要求严格输出 JSON(键名固定,便于解析)。文本内容需中英双语
(zh/en 同时给出);时间一律用「绝对秒」浮点数。
"""

import json
from typing import Dict, List

# 受控能力词表(与 schema 的 capability 枚举一致)
CAPABILITY_VOCAB = [
    "L1_object", "L1_ocr", "L1_action", "L1_audio_asr", "L1_audio_event", "L1_speaker",
    "L2_grounding", "L2_ordering", "L2_counting", "L2_duration", "L2_state_change",
    "L3_causal", "L3_intent", "L3_counterfactual", "L3_prediction", "L3_math_rule",
    "L4_subtitle_visual", "L4_narration_align", "L4_audio_visual",
    "L5_summary", "L5_relation_graph", "L5_cross_scene", "L5_needle", "L5_foreshadow",
]


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


# =========================================================================
# 1) 场景 / 事件标注(多模态,逐窗口)
# =========================================================================
SCENE_SYS = (
    "你是长视频结构化标注专家。根据给定时间窗内的采样画面、字幕与屏幕文字(OCR),"
    "输出该时间窗内的场景、事件与出现的人物。只输出 JSON,不要解释。"
    "所有文本字段都要给出中英双语(zh 与 en);时间用绝对秒(float)。"
)


def scene_user(win_start: float, win_end: float, sub_text: str, ocr_text: str,
               meta: Dict) -> str:
    return (
        f"视频题材: {meta['meta']['genre']};主语言: {meta['meta']['primary_lang']}。\n"
        f"当前时间窗: [{win_start:.1f}s, {win_end:.1f}s]。\n"
        f"窗内字幕(可能为空):\n{sub_text or '(无)'}\n"
        f"窗内屏幕文字 OCR:\n{ocr_text or '(无)'}\n\n"
        "请综合画面+字幕+OCR,输出严格 JSON:\n"
        "{\n"
        '  "scenes": [{"start": float, "end": float, '
        '"summary": {"zh": str, "en": str}, '
        '"characters": [str], "location": {"zh": str, "en": str}, "emotion": str}],\n'
        '  "events": [{"span": [float, float], "desc": {"zh": str, "en": str}, "type": str}],\n'
        '  "characters_seen": [{"tmp_id": str, "appearance": {"zh": str, "en": str}}]\n'
        "}\n"
        "要求:1) 时间必须落在当前窗口内;2) 一个窗口可含 1~3 个场景;"
        "3) event 是推动叙事/因果的最小动作单位,不要把静态背景当事件;"
        "4) characters 用可复用的临时称谓(如 'man_in_red'),同一人跨事件保持一致。"
    )


# =========================================================================
# 2) 全局聚合(纯文本):梗概 / 人物表 / 关系 / 时间线 / 因果图
# =========================================================================
GLOBAL_SYS = (
    "你是影视/长视频叙事分析专家。给定全片按时间排列的场景与事件列表,"
    "输出全局理解:剧情梗概、统一人物表(稳定 id)、人物关系、按【故事时间】排序的"
    "事件时间线,以及事件间的因果边。只输出 JSON。所有文本中英双语。"
)


def global_user(scenes: List[Dict], events: List[Dict], meta: Dict) -> str:
    scene_brief = [
        {"id": s["id"], "start": s["start"], "end": s["end"],
         "summary": s.get("summary", {}).get("en") or s.get("summary", {}).get("zh"),
         "characters": s.get("characters", [])}
        for s in scenes
    ]
    event_brief = [
        {"id": e["id"], "span": e["span"],
         "desc": e.get("desc", {}).get("en") or e.get("desc", {}).get("zh")}
        for e in events
    ]
    return (
        f"题材: {meta['meta']['genre']}。\n"
        f"场景列表:\n{_j(scene_brief)}\n\n"
        f"事件列表(id 已分配):\n{_j(event_brief)}\n\n"
        "请输出严格 JSON:\n"
        "{\n"
        '  "synopsis": {"zh": str, "en": str},\n'
        '  "characters": [{"id": "char_01", "name": {"zh": str, "en": str}, '
        '"role": {"zh": str, "en": str}, "description": {"zh": str, "en": str}}],\n'
        '  "relations": [{"src": "char_01", "dst": "char_02", '
        '"type": {"zh": str, "en": str}, "directed": true}],\n'
        '  "timeline": ["e1", "e2", ...],\n'
        '  "causal": [{"event": "e5", "causes": ["e2", "e3"]}]\n'
        "}\n"
        "要求:1) timeline 按故事时间(非叙事顺序)重排所有事件 id,必须覆盖全部事件;"
        "2) causal 只保留有明确因果的边,causes 指向更早的直接前因;"
        "3) 人物表要把不同场景里的同一人合并为一个稳定 id。"
    )


# =========================================================================
# 3) 出题(纯文本):基于结构化标注生成 QA
# =========================================================================
QA_SYS = (
    "你是长视频理解测评的出题专家,面向多模态大模型(Video-LLM)。"
    "基于给定的场景、事件(含因果)与全局信息出题。只输出 JSON。"
    "严守四条底线:每题必须依赖长程信息、不能靠语言先验盲答、不能靠单帧答对、"
    "证据须落在给定时间区间内。"
)


def qa_user(scenes: List[Dict], events: List[Dict], global_block: Dict,
            meta: Dict, n: int, cross_scene_ratio: float) -> str:
    scene_brief = [{"id": s["id"], "start": s["start"], "end": s["end"],
                    "summary": s.get("summary", {})} for s in scenes]
    event_brief = [{"id": e["id"], "span": e["span"], "desc": e.get("desc", {}),
                    "causes": e.get("causes", [])} for e in events]
    vlang = meta["meta"]["primary_lang"]
    return (
        f"题材: {meta['meta']['genre']};视频主语言 video_lang = {vlang}。\n"
        f"场景:\n{_j(scene_brief)}\n\n"
        f"事件(含因果 causes):\n{_j(event_brief)}\n\n"
        f"全局:\n{_j(global_block)}\n\n"
        f"请出 {n} 道题,输出严格 JSON: {{\"qa\": [ ... ]}}。每题字段:\n"
        "{\n"
        '  "task_type": "mcq" | "temporal_grounding" | "open" | "summary",\n'
        f'  "capability": 从该词表多选1~2个: {CAPABILITY_VOCAB},\n'
        '  "lang_mode": "parallel" | "native" | "cross_lingual",\n'
        f'  "video_lang": "{vlang}",\n'
        '  "question": {"zh": str, "en": str},\n'
        '  "options": [{"id":"A","text":{"zh":str,"en":str},"distractor_trap":"..."}, ...4项] (仅 mcq),\n'
        '  "answer": "A|B|C|D" (仅 mcq),\n'
        '  "temporal_target": [float,float] (仅 temporal_grounding),\n'
        '  "reference_answer": {"zh":str,"en":str} (open/summary),\n'
        '  "evidence_spans": [[float,float], ...],  // 回答所需证据的时间区间,可多段\n'
        '  "evidence_modality": ["visual"|"subtitle"|"audio"|"ocr"],\n'
        '  "min_watch": "single_scene" | "cross_scene" | "whole_video",\n'
        '  "difficulty": "easy" | "medium" | "hard"\n'
        "}\n"
        "硬性要求:\n"
        f"1) 至少 {int(round(cross_scene_ratio*100))}% 的题 min_watch 为 cross_scene 或 whole_video,"
        "且其 evidence_spans 含 ≥2 段不相邻区间;\n"
        "2) 覆盖至少 3 个不同能力层(L1~L5);优先出 L3 因果/意图、L5 长程/关系/伏笔;\n"
        "3) MCQ 四选一唯一正确;每题至少 1 个干扰项的 distractor_trap ∈ "
        "{single_frame, language_prior, subtitle_only, plausible_unhappened},"
        "正确项 distractor_trap 填 'none';\n"
        "4) 正确项不得是唯一最长选项;干扰项要“看似合理但片中未发生”,且可在视频中反驳;\n"
        "5) evidence_spans 内时间必须来自上面给定的场景/事件区间;\n"
        "6) 多数题用 lang_mode='parallel'(中英双版),可少量 native / cross_lingual。"
    )


# =========================================================================
# 4) 抗捷径过滤:让模型仅凭有限信息作答,看能否答对
# =========================================================================
ANSWER_SYS = (
    "你是答题者。只根据给出的信息作答,信息不足就猜最可能的一项。"
    '只输出 JSON: {"answer": "A" | "B" | "C" | "D"}。'
)


def _options_text(q: Dict, lang: str = "en") -> str:
    lines = []
    for o in q.get("options", []):
        t = o["text"].get(lang) or o["text"].get("zh") or o["text"].get("en")
        lines.append(f"{o['id']}. {t}")
    return "\n".join(lines)


def _question_text(q: Dict, lang: str = "en") -> str:
    return q["question"].get(lang) or q["question"].get("zh") or q["question"].get("en")


def blind_user(q: Dict) -> str:
    """盲答:只给题干+选项,不给任何视频信息。"""
    return (
        "仅凭常识回答(没有视频):\n"
        f"问题: {_question_text(q)}\n选项:\n{_options_text(q)}"
    )


def subtitle_user(q: Dict, sub_text: str) -> str:
    """字幕捷径:给题干+选项+证据区间字幕文本,不给画面。"""
    return (
        "根据以下字幕文本回答(没有画面):\n"
        f"字幕:\n{sub_text or '(无)'}\n\n"
        f"问题: {_question_text(q)}\n选项:\n{_options_text(q)}"
    )


def frame_user(q: Dict) -> str:
    """单帧捷径:配合 1 张图片一起发送(图片在消息里另附)。"""
    return (
        "根据这一张图片回答(只有单帧,没有其它时刻):\n"
        f"问题: {_question_text(q)}\n选项:\n{_options_text(q)}"
    )


# =========================================================================
# 5) 全片详细描述:分段密集详述 + 全片汇总
# =========================================================================
DESCRIBE_SYS = (
    "你是长视频详细描述专家。根据给定时间窗内的采样画面、字幕与屏幕文字(OCR),"
    "输出该段的【详细描述】(不是概要):按时间顺序描述画面内容、场景与镜头变化、"
    "人物及其动作与互动、显著物体、屏幕文字、对白/解说要点。"
    "只描述可观察到的内容,不要臆造画面中没有的东西。中英双语。只输出 JSON,不要解释。"
)


def describe_user(win_start: float, win_end: float, sub_text: str, ocr_text: str,
                  meta: Dict) -> str:
    return (
        f"视频题材: {meta['meta']['genre']};主语言: {meta['meta']['primary_lang']}。\n"
        f"当前时间窗: [{win_start:.1f}s, {win_end:.1f}s]。\n"
        f"窗内字幕:\n{sub_text or '(无)'}\n"
        f"窗内屏幕文字 OCR:\n{ocr_text or '(无)'}\n\n"
        "请输出严格 JSON:\n"
        '{ "description": {"zh": str, "en": str} }\n'
        "要求:1) 详细、按时间先后组织,可在文中标注关键时刻(如“约 12:30”);"
        "2) 覆盖画面主体、动作、场景/镜头切换、字幕/对白要点;"
        "3) 只写观察到的,不臆造;4) 中英内容对应一致。"
    )


DESCRIBE_AGG_SYS = (
    "你是长视频详述汇总专家。给定按时间排列的【分段详细描述】,"
    "合并成一段连贯、覆盖全片、保持时间顺序的详细描述,可分段落但需前后衔接。"
    "忠于输入,不要新增未提及的情节。中英双语。只输出 JSON。"
)


def describe_agg_user(items: List[Dict], meta: Dict) -> str:
    """items: [{start, end, text}],text 为该段详述(取一种语言即可)。"""
    briefs = [{"start": it["start"], "end": it["end"], "text": it["text"]} for it in items]
    return (
        f"题材: {meta['meta']['genre']}。共 {len(items)} 段,按时间排列:\n"
        f"{_j(briefs)}\n\n"
        "请输出严格 JSON:\n"
        '{ "detailed_description": {"zh": str, "en": str} }\n'
        "要求:按时间顺序融合为连贯详述,保留关键细节与转折,不要逐段罗列口吻。"
    )
