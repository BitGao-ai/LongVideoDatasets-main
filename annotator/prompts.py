# -*- coding: utf-8 -*-
"""双语提示词模板(v2,Res 全案)。

所有 LLM 阶段都要求严格输出 JSON(键名固定,便于解析)。文本内容需中英双语
(zh/en 同时给出);时间一律用「绝对秒」浮点数。

v2 新增(对应 Res 各节):
- 镜头/片段级详述 + 要素清单(§2.1/2.2)
- 事件边界复核 8×8 马赛克(§3.2)、去重(§3.3)、关键性分级(§3.4)
- 证据锚定出题 + referring 上下文(§5.1)、答案核验(§5.3)、梗概泄漏(§5.4)
- 非 MCQ 捷径检测(§6.3)
"""

import json
from typing import Dict, List, Optional

# 受控能力词表(与 schema 的 capability 枚举一致)
CAPABILITY_VOCAB = [
    "L1_object", "L1_ocr", "L1_action", "L1_audio_asr", "L1_audio_event", "L1_speaker",
    "L2_grounding", "L2_ordering", "L2_counting", "L2_duration", "L2_state_change",
    "L3_causal", "L3_intent", "L3_counterfactual", "L3_prediction", "L3_math_rule",
    "L4_subtitle_visual", "L4_narration_align", "L4_audio_visual",
    "L5_summary", "L5_relation_graph", "L5_cross_scene", "L5_needle", "L5_foreshadow",
]

# 详述要素清单(Res 2.1:8 要素,≥6/8 才接受)
ELEMENT_CHECKLIST = [
    "characters",   # 人物与穿着
    "action",       # 主体动作与交互
    "objects",      # 显著物体
    "camera",       # 镜头(景别/运动)
    "scene_light",  # 场景与光影
    "text",         # 字幕/屏幕文字
    "audio",        # 背景声/音效
    "mood",         # 情绪氛围
]


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


# =========================================================================
# 1) 场景 / 事件标注(多模态,逐窗口)—— 窗口失败走重试 + gaps(Res 3.6)
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
# 2) 全局聚合 v2(纯文本):梗概 / 人物 / 关系 / 时间线 / 因果 / 主题 / 转折 / 伏笔
#    (Res 4.1/4.2: themes / turning_points / motifs;因果边只允许指向已验证事件)
# =========================================================================
GLOBAL_SYS = (
    "你是影视/长视频叙事分析专家。给定全片按时间排列的场景与事件列表,"
    "输出全局理解:剧情梗概、统一人物表(稳定 id)、人物关系、按【故事时间】排序的"
    "事件时间线、事件间的因果边,以及主题、故事转折点、伏笔/呼应线索。"
    "只输出 JSON。所有文本中英双语。"
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
         "desc": e.get("desc", {}).get("en") or e.get("desc", {}).get("zh"),
         "importance": e.get("importance", "minor"),
         "boundary_state": e.get("boundary_state", "unreviewed")}
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
        '  "causal": [{"event": "e5", "causes": ["e2", "e3"]}],\n'
        '  "themes": [{"name": {"zh": str, "en": str}, '
        '"evidence_event_ids": ["e1","e5"], "desc": {"zh": str, "en": str}}],\n'
        '  "turning_points": [{"span": [float,float], "type": str, '
        '"desc": {"zh": str, "en": str}, "importance": "high"|"medium"|"low"}],\n'
        '  "motifs": [{"name": {"zh": str, "en": str}, '
        '"clue_spans": [[float,float],...], "desc": {"zh": str, "en": str}}]\n'
        "}\n"
        "要求:1) timeline 按故事时间(非叙事顺序)重排全部事件 id,必须覆盖所有事件;"
        "2) causal 只保留有明确因果的边,causes 指向更早的直接前因;"
        "3) 人物表要把不同场景里的同一人合并为一个稳定 id;"
        "4) themes 2~5 个,每个必须挂 ≥2 个支持事件的 evidence_event_ids;"
        "5) turning_points 3~7 个;6) motifs 是服务于伏笔/呼应(能力 L5_foreshadow)的线索。"
    )


# =========================================================================
# 3) 出题 v2(纯文本,证据锚定)—— 输入【不含梗概/全片详述】(Res 5.1/5.4 防泄漏)
# =========================================================================
QA_SYS = (
    "你是长视频理解测评的出题专家,面向多模态大模型(Video-LLM)。"
    "基于给定的场景、事件(含因果/重要性/边界状态)、主题与人物关系出题。只输出 JSON。"
    "严守五条底线:每题必须锚定具体证据区间、题干必须自带指代上下文(时间/序数/锁定短语)、"
    "不能靠语言先验盲答、不能靠单帧答对、证据须落在给定时间区间内。"
)


def _anchor_brief(anchor: Dict) -> Dict:
    return {
        "anchor_id": anchor.get("anchor_id"),
        "span": anchor.get("span"),
        "desc": anchor.get("desc"),
        "modality": anchor.get("modality", ["visual"]),
        "importance": anchor.get("importance", "minor"),
        "type": anchor.get("type", "event"),
    }


def qa_user(anchors: List[Dict], global_block: Dict, meta: Dict, n: int,
            matrix_req: Dict) -> str:
    """锚定出题:anchors 为锚点池(关键事件优先,Res 5.1)。"""
    vlang = meta["meta"]["primary_lang"]
    anchor_briefs = [_anchor_brief(a) for a in anchors]
    ctx = {k: global_block.get(k) for k in ("characters", "relations", "timeline",
                                            "themes", "turning_points", "motifs")}
    used = matrix_req.get("used_anchor_ids") or []
    used_note = (
        f"\n以下锚点在本视频【已经出过题】,请改用其它锚点,确有必要复用时必须换一个"
        f"完全不同的提问角度:\n{_j(used[:80])}\n" if used else ""
    )
    return (
        f"题材: {meta['meta']['genre']};视频主语言 video_lang = {vlang}。\n"
        f"证据锚点池(共 {len(anchor_briefs)} 个,按时间排列,含 span/描述/模态):\n"
        f"{_j(anchor_briefs)}\n"
        f"{used_note}\n"
        f"全局上下文(人物/关系/时间线/主题/转折/伏笔):\n{_j(ctx)}\n\n"
        f"请出 {n} 道题,输出严格 JSON: {{\"qa\": [ ... ]}}。每题字段:\n"
        "{\n"
        '  "anchor_id": 引用的证据锚点 id(必须来自锚点池),\n'
        '  "task_type": "mcq" | "temporal_grounding" | "open" | "summary",\n'
        f'  "capability": 从该词表多选1~2个: {CAPABILITY_VOCAB},\n'
        '  "lang_mode": "parallel" | "native" | "cross_lingual",\n'
        f'  "video_lang": "{vlang}",\n'
        '  "question": {"zh": str, "en": str},\n'
        '  "referring_ctx": {"zh": str, "en": str},  // 题干引用的上下文(时刻/对象定位)\n'
        '  "options": [{"id":"A","text":{"zh":str,"en":str},"distractor_trap":"..."}, ...4项] (仅 mcq),\n'
        '  "answer": "A|B|C|D" (仅 mcq),\n'
        '  "temporal_target": [float,float] (仅 temporal_grounding,须在锚点 span 内),\n'
        '  "reference_answer": {"zh":str,"en":str} (open/summary,须含 ≥2 个视频内具体细节),\n'
        '  "evidence_spans": [[float,float], ...],  // 必须取自锚点 span,可多段\n'
        '  "evidence_modality": ["visual"|"subtitle"|"audio"|"ocr"],\n'
        '  "min_watch": "single_scene" | "cross_scene" | "whole_video",\n'
        '  "difficulty": "easy" | "medium" | "hard"\n'
        "}\n"
        "硬性要求(代码会逐条校验,不满足会打回):\n"
        f"1) 题干必须自带指代上下文(Res 5.1):含时间描述(\"第 12 分钟\"/\"争吵结束前\")、"
        "或序数消歧词(\"第二次/最后一次\")、或事件名/地点/人物锁定短语;中文题干 ≥25 字;\n"
        f"2) 至少 {int(round(matrix_req.get('cross_scene_min_ratio', 0.5) * 100))}% 的题 "
        "min_watch 为 cross_scene 或 whole_video,且其 evidence_spans 含 ≥2 段不相邻区间;\n"
        "3) 能力层矩阵(待补齐缺口,Res 5.2):"
        f"{_j(matrix_req.get('capability_deficit', {}))},"
        "请优先补足缺口能力;task_type 配额缺口:"
        f"{_j(matrix_req.get('task_type_deficit', {}))};\n"
        "4) MCQ 四选一唯一正确;正确项不得是唯一最长选项,4 个选项长度尽量均匀;"
        "至少 1 个干扰项的 distractor_trap ∈ {single_frame, language_prior, subtitle_only, "
        "plausible_unhappened},正确项填 'none';\n"
        "5) evidence_spans 必须取自给定锚点 span 内;open/summary 的 reference_answer "
        "中文 ≥80 字/英文 ≥220 词,并含 ≥2 个具体细节(时间锚点或字幕片段引用);\n"
        "6) temporal_grounding 的 temporal_target 必须落在锚点 span 内且宽度 ≥5s。"
    )


def qa_rewrite_user(q: Dict, feedback: str) -> str:
    """题干 referring 校验不过时的重写(Res 5.1:重写 ≤2 次)。"""
    return (
        f"下面这道题需要重写:{feedback}\n"
        f"原题: {_j(q)}\n\n"
        "请输出修正后的完整题目(字段与出题要求一致,evidence_spans 不变或缩紧,"
        "题干补上指代上下文:时间描述/序数消歧/锁定短语)。只输出 JSON: {\"qa\": [ ... ]}。"
    )


# =========================================================================
# 4) 抗捷径过滤 v2(Res 6):盲答 / 字幕 / 单帧 / 梗概 / 语言先验 / 非 MCQ
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


SYNOPSIS_SYS = (
    "你是答题者。只根据剧情梗概与题目作答,信息不足就猜最可能的一项。"
    '只输出 JSON: {"answer": "A" | "B" | "C" | "D"}。'
)


def synopsis_user(q: Dict, synopsis: str) -> str:
    """梗概泄漏通道(Res 5.4/6.2):给题干+选项+梗概,能答对 → 梗概题。"""
    return (
        f"剧情梗概:\n{synopsis or '(无)'}\n\n"
        f"问题: {_question_text(q)}\n选项:\n{_options_text(q)}"
    )


def temporal_locate_user(q: Dict, sub_text: str) -> str:
    """temporal_grounding 字幕定位捷径(Res 6.3):只给字幕+题面,输出区间。"""
    return (
        "只根据字幕定位问题描述事件发生的时间区间(没有画面):\n"
        f"字幕:\n{sub_text or '(无)'}\n\n"
        f"问题: {_question_text(q)}\n"
        '只输出 JSON: {"interval": [start_sec, end_sec]}'
    )


OPEN_BLIND_SYS = (
    "你是答题者。只根据题干作答(没有视频),给出你的自由回答。"
    '只输出 JSON: {"answer": str}。'
)


def open_blind_user(q: Dict) -> str:
    """open/summary 盲答通道(Res 6.3):无视频作答,与参考答案比对相似度。"""
    return (
        "仅凭常识回答(没有视频):\n"
        f"问题: {_question_text(q)}\n"
        '只输出 JSON: {"answer": str}'
    )


# =========================================================================
# 5) 详述 v2(Res 2):帧级短描述 / 镜头级 / 片段级(要素清单+时间锚点)/ 全片汇总
# =========================================================================
CAPTION_SYS = (
    "你是视频帧描述员。用不超过 30 个中文字描述这张画面里正在发生什么。"
    "只输出一句短描述,不要解释、不要标点堆砌。"
)


def caption_user(ctx: str = "") -> str:
    return ctx or "描述这张画面的主要内容(≤30字)。"


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
        '{ "description": {"zh": str, "en": str}, '
        '"elements": {"characters": bool, "action": bool, "objects": bool, '
        '"camera": bool, "scene_light": bool, "text": bool, "audio": bool, "mood": bool} }\n'
        "要求:1) 详细、按时间先后组织,文中至少标注 2 处关键时刻(如“约 12:30”);"
        "2) 覆盖画面主体、动作、场景/镜头切换、字幕/对白要点;"
        "3) 只写观察到的,不臆造;4) 中英内容对应一致;"
        "5) elements 逐项如实标注本段描述覆盖到的要素(≥6/8 为合格)。"
    )


def segment_desc_user(seg_start: float, seg_end: float, shot_desc_briefs: List[Dict],
                      sub_text: str, ocr_text: str, meta: Dict,
                      retry_feedback: str = "") -> str:
    """片段级(segment≈180s):以 Shot 级描述为输入汇总,保留各镜头起止(Res 2.1)。"""
    fb = f"\n上一次不合格,原因: {retry_feedback}\n请针对缺项补足。" if retry_feedback else ""
    return (
        f"视频题材: {meta['meta']['genre']};主语言: {meta['meta']['primary_lang']}。\n"
        f"片段区间: [{seg_start:.1f}s, {seg_end:.1f}s]。\n"
        f"片段内各镜头的密集描述(按时间):\n{_j(shot_desc_briefs)}\n"
        f"片段内字幕:\n{sub_text or '(无)'}\n片段内 OCR:\n{ocr_text or '(无)'}\n\n"
        "请输出该片段的【详细描述】(非概要),融合镜头描述并按时间组织:\n"
        "{\n"
        '  "description": {"zh": str, "en": str},\n'
        '  "elements": {"characters": bool, "action": bool, "objects": bool, '
        '"camera": bool, "scene_light": bool, "text": bool, "audio": bool, "mood": bool}\n'
        "}\n"
        "硬性要求(机器校验,不达标会重试):1) 中文 ≥400 字 或 英文 ≥350 词;"
        "2) 要素覆盖 ≥6/8;3) 正文含 ≥2 处“约 MM:SS”时间锚点(方便人工回溯);"
        "4) 保留镜头起止标注(如“[00:12-00:45] …”);5) 只写观察到的,不臆造。" + fb
    )


def shot_desc_user(shot_start: float, shot_end: float, sub_text: str, ocr_text: str,
                   meta: Dict, retry_feedback: str = "") -> str:
    fb = f"\n上一次不合格,原因: {retry_feedback}\n请针对缺项补足。" if retry_feedback else ""
    return (
        f"视频题材: {meta['meta']['genre']}。镜头区间: [{shot_start:.1f}s, {shot_end:.1f}s]。\n"
        f"镜头内字幕:\n{sub_text or '(无)'}\n镜头内 OCR:\n{ocr_text or '(无)'}\n"
        "请综合 3 张关键帧画面(首/中/尾)+ 字幕 + OCR,输出该镜头的密集描述:\n"
        "{\n"
        '  "description": {"zh": str, "en": str},\n'
        '  "elements": {"characters": bool, "action": bool, "objects": bool, '
        '"camera": bool, "scene_light": bool, "text": bool, "audio": bool, "mood": bool}\n'
        "}\n"
        "硬性要求:中文 ≥80 字 或 英文 ≥60 词;按【要素清单】组织:人物与穿着 | "
        "主体动作与交互 | 显著物体 | 镜头(景别/运动) | 场景与光影 | 字幕/屏幕文字 | "
        "背景声/音效 | 情绪氛围,覆盖 ≥6/8 才合格;只写观察到的,不臆造。"
        "时间可在文中标注(如“约 MM:SS”)。" + fb
    )


DESCRIBE_AGG_SYS = (
    "你是长视频详述汇总专家。给定按时间排列的【分段详细描述】(zh+en 双轨),"
    "合并成一段连贯、覆盖全片、保持时间顺序的详细描述,可分段落但需前后衔接。"
    "忠于输入,不要新增未提及的情节。中文汇总 ≥1000 字。中英双语。只输出 JSON。"
)


def describe_agg_user(items: List[Dict], meta: Dict) -> str:
    """items: [{start, end, text_zh, text_en}],双语并轨输入(Res 2.1 全片级)。"""
    briefs = [{"start": it["start"], "end": it["end"],
               "zh": it.get("text_zh") or "", "en": it.get("text_en") or ""} for it in items]
    return (
        f"题材: {meta['meta']['genre']}。共 {len(items)} 段,按时间排列:\n"
        f"{_j(briefs)}\n\n"
        "请输出严格 JSON:\n"
        '{ "detailed_description": {"zh": str, "en": str} }\n'
        "要求:按时间顺序融合为连贯详述,保留关键细节与转折,不要逐段罗列口吻;"
        "中文 ≥1000 字(2h 视频应 2000+)。"
    )


TRANSLATE_SYS = (
    "你是专业影视字幕翻译。把给定的详细描述逐段翻译成目标语言,"
    "保持时间锚点(MM:SS)与专有名词一致,不增删内容。只输出 JSON: {\"text\": str}。"
)


def translate_user(text: str, target_lang: str) -> str:
    return f"目标语言: {target_lang}。\n原文:\n{text}"


# =========================================================================
# 6) 事件引擎 v2(Res 3):边界复核 / 去重 / 关键性分级
# =========================================================================
BOUNDARY_SYS = (
    "你是视频事件边界复核专家。给你一个候选事件及其时间 span,以及一张由该事件"
    "前后若干真实帧拼接成的【方阵马赛克】。马赛克按【行优先】排列:左上角是最早的一帧,"
    "沿每一行从左到右递增,换行继续,右下角是最晚的一帧。每一格对应的绝对时刻会在"
    "文字部分逐一给出,请严格按该对照表判断,不要自行推算。\n"
    "判定事件边界是否准确:\n"
    "  - verified: 边界正确,直接通过;\n"
    "  - refined:  边界偏差,给出修正后的新 span;\n"
    "  - split:    当前 span 实际包含两个不同事件,给出两个分立 span;\n"
    "  - rejected: 该事件在画面中不存在(挖掘器误报)。\n"
    "所有输出的时间都必须是对照表里出现过的绝对秒数(可取相邻两格之间的值)。"
    "只输出 JSON。"
)


def _tile_table(times: List[float], side: int) -> str:
    """逐格时间对照表:行优先,(行,列) -> 绝对秒。"""
    lines = []
    for r in range(side):
        row = times[r * side:(r + 1) * side]
        if not row:
            break
        lines.append("  第%d行: %s" % (r + 1, ", ".join(f"{t:.1f}s" for t in row)))
    return "\n".join(lines)


def boundary_user(ev: Dict, tile_times: Optional[List[float]] = None,
                  side: int = 8) -> str:
    s, e = ev["span"]
    desc = ev.get("desc", {})
    tile_times = tile_times or []
    grid_note = (
        f"马赛克为 {side}×{side} 方阵,共 {len(tile_times)} 格,行优先排列;"
        f"每格对应时刻:\n{_tile_table(tile_times, side)}\n"
        f"相邻两格间隔约 {(tile_times[1] - tile_times[0]):.2f}s。\n"
        if len(tile_times) >= 2 else ""
    )
    return (
        f"候选事件: {_j(ev.get('anchor_desc') or desc.get('en') or desc.get('zh'))}\n"
        f"当前 span: [{s:.1f}s, {e:.1f}s]\n"
        f"{grid_note}"
        "注意:上表覆盖的时间范围相对当前 span 并不对称,当前 span 不一定落在画面正中,"
        "请只依据上表逐格判断。\n"
        "只输出 JSON:\n"
        "{\n"
        '  "decision": "verified" | "refined" | "split" | "rejected",\n'
        '  "new_span": [float, float],            // refined 时必填,取自上表时刻\n'
        '  "split_spans": [[float,float],[float,float]],  // split 时必填\n'
        '  "reason": {"zh": str, "en": str}\n'
        "}"
    )


DEDUP_SYS = (
    "你是视频事件去重专家。给定一批【时间重叠】的候选事件,判断哪些其实是同一事件"
    "的重复记录(跨窗/跨摘要重复)。合并规则:同一主体 | 相似动作 | 同一场景 且时间重叠。"
    "只输出 JSON。"
)


def dedup_user(events: List[Dict]) -> str:
    briefs = [{"id": e["id"], "span": e["span"], "desc": e.get("desc", {})} for e in events]
    return (
        f"候选事件列表:\n{_j(briefs)}\n\n"
        "请输出严格 JSON: {\"dedup_groups\": [{\"keep\": \"e_id\", "
        "\"merge\": [\"e_id\", ...], \"reason\": str}]}\n"
        "要求:1) 只合并同一主体+相似动作+同一场景且时间重叠的事件;"
        "2) 不要合并“同一主体多场景”或“同场景不同动作”;3) keep 为信息量最大的那个。"
    )


IMPORTANCE_SYS = (
    "你是视频关键事件分级专家。给定视频的事件列表,把每个事件标注为 "
    "key(关键,对主线/因果/主题有贡献)或 minor(次要)。"
    "只输出 JSON。"
)


def importance_user(events: List[Dict], duration: float) -> str:
    briefs = [{"id": e["id"], "span": e["span"], "desc": e.get("desc", {})} for e in events]
    return (
        f"视频时长 {duration:.0f}s。事件列表:\n{_j(briefs)}\n\n"
        "请输出严格 JSON: {\"importance\": [{\"event\": \"e_id\", "
        "\"level\": \"key\" | \"minor\", \"reason\": \"一句话依据\"}]}\n"
        "要求:key 事件必须对主线/因果/主题有贡献(转折、冲突、关键决定、因果节点);"
        "单场景静态镜头多为 minor;参考:0.4-2h 视频 key 事件 15~40 个。"
    )


# =========================================================================
# 7) 答案双模型核验(Res 5.3:VIDEO 核验)
# =========================================================================
VERIFY_SYS = (
    "你是长视频答题验证员。题目给出证据区间的真实画面(或多帧/片段),"
    "请独立判断正确选项,并给出置信度与理由。"
    '只输出 JSON: {"answer": "A"|"B"|"C"|"D", "confidence": 0~1, '
    '"reasons": {"zh": str, "en": str}}。'
)


def verify_user(q: Dict, evidence_note: str = "") -> str:
    return (
        f"题目: {_question_text(q)}\n"
        f"选项:\n{_options_text(q)}\n"
        f"{evidence_note or '以下画面是题目证据区间的采样。'}\n"
        "请根据画面独立作答(不要参考其它信息)。"
    )


TEMPORAL_VERIFY_SYS = (
    "你是长视频时间定位验证员。给你一个问题,以及一张由候选区间前后真实帧拼接成的"
    "【方阵马赛克】。马赛克按【行优先】排列:左上角最早,沿每行从左到右递增,换行继续,"
    "右下角最晚。每格对应的绝对时刻会在文字部分逐一给出,请严格按对照表判断,不要自行推算。\n"
    "请独立给出问题所描述事件发生的时间区间(绝对秒),端点必须取自对照表中的时刻。"
    '只输出 JSON: {"interval": [start_sec, end_sec], "confidence": 0~1, '
    '"reasons": {"zh": str, "en": str}}。'
)


def temporal_verify_user(q: Dict, tile_times: List[float], side: int) -> str:
    """temporal_grounding 区间核验(Res 8.2 mIoU):不告知标准区间,独立定位。"""
    grid_note = (
        f"马赛克为 {side}×{side} 方阵,共 {len(tile_times)} 格,行优先排列;"
        f"每格对应时刻:\n{_tile_table(tile_times, side)}\n"
        f"相邻两格间隔约 {(tile_times[1] - tile_times[0]):.2f}s。\n"
        if len(tile_times) >= 2 else ""
    )
    lo = tile_times[0] if tile_times else 0.0
    hi = tile_times[-1] if tile_times else 0.0
    return (
        f"问题: {_question_text(q, 'zh')}\n"
        f"可判断的时间范围: [{lo:.1f}s, {hi:.1f}s]\n"
        f"{grid_note}"
        "请只根据画面定位该事件的起止时刻,不要参考问题以外的信息,"
        "也不要假设事件一定占满整个范围。"
    )
