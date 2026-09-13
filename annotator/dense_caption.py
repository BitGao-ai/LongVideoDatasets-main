# -*- coding: utf-8 -*-
"""密集描述引擎 M1(Res 2)—— 核心需求 1:描述必须非常详细。

三级描述金字塔:
  帧级   : 本地小模型 1fps 短描述(≤30 字,纯增量,供事件挖掘与镜头重组);
  镜头级 : Shot.desc 密集描述(关键帧 首/中/尾 + 字幕 + OCR),要素 ≥6/8、
          中文 ≥80 字/英文 ≥60 词;不达标带反馈重试;
  片段级 : segment≈180s 以 Shot 级描述(带起止标注)汇总,中文 ≥400 字/
          英文 ≥350 词、要素 ≥6/8、≥2 处“约 MM:SS”锚点;未过门重试 1 次,
          仍不过写 quiet_segments 进人工清单(不静默丢弃);
  全片级 : zh+en 双轨两级汇总,中文 ≥1000 字门。

语言对齐(Res 2.3):缺侧用纯文本模型单向翻译补齐;zh/en 长度比 ∈ [0.4, 2.5]
粗查;strict_lang_check 时做回译校验。
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from . import prompts
from .config import RunConfig
from .frames_store import FrameStore
from .llm_client import LLMClient, image_part, text_part
from .media import build_windows, fixed_windows, ocr_text, subs_text
from .utils import (count_time_anchors, en_len, lang_len, retry_exp_backoff,
                    zh_len)

log = logging.getLogger("annotator.dense")

# 要素关键词兜底(模型不输出 elements 时启发式判定)
_ELEMENT_KEYWORDS = {
    "characters": ["人物", "角色", "男人", "女人", "男孩", "女孩", "穿着", "衣服",
                   "man", "woman", "boy", "girl", "wear", "character"],
    "action": ["动作", "走", "跑", "说", "拿", "吃", "打", "抱", "开", "关",
               "walk", "run", "speak", "talk", "hold", "fight"],
    "objects": ["物体", "物品", "桌子", "车", "房子", "机器", "书", "杯子",
                "object", "table", "car", "house", "machine", "book"],
    "camera": ["镜头", "特写", "远景", "中景", "摇", "推", "拉", "切", "跟拍",
               "close-up", "wide", "pan", "zoom", "track", "cut"],
    "scene_light": ["场景", "光线", "灯光", "室内", "室外", "夜景", "黄昏", "阳光",
                    "scene", "light", "indoor", "outdoor", "night", "sunlight"],
    "text": ["字幕", "文字", "屏幕上", "标题", "标语", "subtitle", "text", "title"],
    "audio": ["声音", "音乐", "音效", "背景声", "对白", "掌声", "sound", "music",
              "audio", "dialogue", "applause"],
    "mood": ["氛围", "情绪", "紧张", "温馨", "压抑", "欢快", "悲伤", "mood",
             "tense", "warm", "gloomy", "joyful"],
}


def _elements_from_text(text: str) -> Dict[str, bool]:
    text = (text or "").lower()
    out: Dict[str, bool] = {}
    for key, kws in _ELEMENT_KEYWORDS.items():
        out[key] = any(k.lower() in text for k in kws)
    return out


#: 模型可能用来承载描述正文的键(镜头级 prompt 历史上用 "desc",片段级用
#: "description")。统一归一化到 "description",避免 prompt 与解析层键名漂移
#: 导致整级描述静默失败(修复 A3)。
_DESC_KEYS = ("description", "desc", "detailed_description")


def _norm_desc(out: Dict) -> Dict:
    """归一化描述正文:兼容 desc/description 两种键名,以及纯字符串形态。

    模型偶发把描述输出成纯字符串而非 {zh, en};镜头级与片段级 prompt 又曾用过
    不同键名。这里统一收敛为 out["description"] = {"zh":…, "en":…}。
    """
    out = dict(out or {})
    d = None
    for k in _DESC_KEYS:
        v = out.get(k)
        if isinstance(v, str) and v.strip():
            d = {"zh": v, "en": None}
            break
        if isinstance(v, dict) and (v.get("zh") or v.get("en")):
            d = v
            break
    out["description"] = d if isinstance(d, dict) else {"zh": None, "en": None}
    return out


def _elements_count(out: Dict) -> Tuple[Dict[str, bool], int]:
    out = _norm_desc(out)
    els = out.get("elements")
    if isinstance(els, dict):
        els = {k: bool(v) for k, v in els.items() if k in prompts.ELEMENT_CHECKLIST}
    else:
        desc = out.get("description") or {}
        els = _elements_from_text(f"{desc.get('zh') or ''}\n{desc.get('en') or ''}")
    return els, sum(1 for v in els.values() if v)


def _gate_result(out: Dict, cfg: RunConfig, level: str) -> Tuple[bool, str]:
    """质量门(Res 2.2):长度/要素/锚点;返回 (通过, 缺什么)。"""
    out = _norm_desc(out)
    desc = (out or {}).get("description") or {}
    zh, en = desc.get("zh") or "", desc.get("en") or ""
    els, n_el = _elements_count(out)
    problems = []
    if level == "shot":
        if zh_len(zh) < cfg.shot_desc_min_chars and en_len(en) < cfg.shot_desc_min_words:
            problems.append(f"长度不足(中文≥{cfg.shot_desc_min_chars}字 或 英文≥{cfg.shot_desc_min_words}词)")
    else:  # segment
        if zh_len(zh) < cfg.segment_min_zh and en_len(en) < cfg.segment_min_en:
            problems.append(f"长度不足(中文≥{cfg.segment_min_zh}字 或 英文≥{cfg.segment_min_en}词)")
        if count_time_anchors(zh) < cfg.dense_anchor_min and count_time_anchors(en) < cfg.dense_anchor_min:
            problems.append(f"时间锚点不足(需 ≥{cfg.dense_anchor_min} 处“约 MM:SS”)")
    if n_el < cfg.element_checklist_required:
        problems.append(f"要素覆盖不足({n_el}/{len(prompts.ELEMENT_CHECKLIST)},"
                        f"需 ≥{cfg.element_checklist_required}) 缺:{[k for k, v in els.items() if not v]}")
    return (not problems), "; ".join(problems)


def _has_text(out: Dict) -> bool:
    d = (out or {}).get("description") or {}
    return bool((d.get("zh") or "").strip() or (d.get("en") or "").strip())


def _with_feedback(messages: List[Dict], feedback: str) -> List[Dict]:
    """在【保留原始 prompt】的前提下追加一段反馈(修复 B9)。

    此前的实现把 content 重建为 [feedback] + 非文本部分,等于把时间区间、字幕、
    OCR、要素清单、输出格式全部删掉,第 2/3 次尝试必然更差。
    """
    messages = list(messages)
    last = dict(messages[-1])
    content = last.get("content")
    if isinstance(content, list):
        last["content"] = list(content) + [text_part(feedback)]
    else:
        last["content"] = f"{content}\n\n{feedback}"
    messages[-1] = last
    return messages


def _generate_with_gate(client: LLMClient, cfg: RunConfig, messages: List[Dict],
                        vision: bool, level: str) -> Tuple[Optional[Dict], str]:
    """带质量门的生成:未过门带“上一版+缺什么”重试 ≤dense_max_retries 次(Res 2.2)。"""
    base_messages = messages
    feedback = ""
    last_problem = ""
    for attempt in range(cfg.dense_max_retries + 1):
        msgs = _with_feedback(base_messages, feedback) if feedback else base_messages
        try:
            out = client.complete(msgs, vision=vision)
            out = _norm_desc(out) if isinstance(out, dict) else out
        except Exception as e:  # noqa: BLE001 —— 单次调用失败继续重试,不再直接放弃
            log.warning("详述调用失败(第 %d/%d 次): %s", attempt + 1,
                        cfg.dense_max_retries + 1, e)
            last_problem = f"调用失败: {e}"
            feedback = ""
            continue
        if not isinstance(out, dict) or not _has_text(out):
            last_problem = "输出结构不完整(description 为空)"
            feedback = f"上一版未通过质量门: {last_problem}"
            continue
        ok, problem = _gate_result(out, cfg, level)
        if ok:
            return out, ""
        last_problem = problem
        feedback = (f"上一版未通过质量门: {problem}\n"
                    f"上一版文本: {str(out.get('description'))[:200]}\n"
                    "请在满足上述所有硬性要求的前提下重写,输出格式保持不变。")
        log.info("详述质量门未过(%s): %s", level, problem)
    return None, last_problem


def describe_shot(client: LLMClient, cfg: RunConfig, store: FrameStore, shot: Dict,
                  structure: Dict, meta: Dict) -> Optional[Dict]:
    """镜头级描述:关键帧(首/中/尾)+ 字幕 + OCR,质量门(Res 2.1 镜头级)。"""
    span = (shot["start"], shot["end"])
    frames = store.frames_b64(store.shot_keyframe_times(*span), cfg.max_image_edge)
    if not frames:
        log.warning("镜头 %s 关键帧为空,跳过描述", shot.get("id"))
        return None
    sub = subs_text(structure.get("subtitles", []), *span)
    ocr = ocr_text(structure.get("ocr", []), *span)
    content = [text_part(prompts.shot_desc_user(span[0], span[1], sub, ocr, meta))]
    content += [image_part(b64) for _, b64 in frames]
    out, _ = _generate_with_gate(client, cfg,
                                 [{"role": "system", "content": prompts.DESCRIBE_SYS},
                                  {"role": "user", "content": content}],
                                 vision=True, level="shot")
    return out


def describe_segment(client: LLMClient, cfg: RunConfig, store: FrameStore, ws: float,
                     we: float, shot_desc: List[Dict], structure: Dict,
                     meta: Dict) -> Optional[Dict]:
    """片段级描述:Shot 级描述(带起止)汇总 + 字幕/OCR + 少量采样帧(Res 2.1 片段级)。"""
    briefs = [{"start": s.get("start"), "end": s.get("end"), "desc": s.get("desc")}
              for s in shot_desc if s.get("desc")]
    sub = subs_text(structure.get("subtitles", []), ws, we)
    ocr = ocr_text(structure.get("ocr", []), ws, we)
    content = [text_part(prompts.segment_desc_user(ws, we, briefs, sub, ocr, meta))]
    # 少量采样帧兜底画面细节。此前硬编码 6 帧,--frames / describe_frames_per_window
    # 传了也没人用(D11)。
    k = cfg.describe_frames_per_window or max(6, cfg.frames_per_window)
    frames = store.frames_b64(store.sample_times(ws, we, k), cfg.max_image_edge)
    content += [image_part(b64) for _, b64 in frames]
    return _generate_with_gate(client, cfg,
                               [{"role": "system", "content": prompts.DESCRIBE_SYS},
                                {"role": "user", "content": content}],
                               vision=True, level="segment")[0]


def _native_segment(client: LLMClient, cfg: RunConfig, store: FrameStore, ws: float,
                    we: float, shot_desc: List[Dict], structure: Dict,
                    meta: Dict) -> Optional[Dict]:
    """native_video 模式片段详述(仅 Qwen;复用 describe.py 的切片能力)。"""
    from . import native_video
    from .config import get_api_key
    import os
    import tempfile
    api_key = get_api_key(client.pc.api_key_env)
    clipdir = cfg.native_clip_dir or tempfile.gettempdir()
    os.makedirs(clipdir, exist_ok=True)
    clip = os.path.join(clipdir, f"clip_{meta.get('video_id', 'v')}_{int(ws)}_{int(we)}.mp4")
    native_video.cut_clip(_video_path_from_meta(meta), ws, we, clip, cfg.max_image_edge)
    try:
        briefs = [{"start": s.get("start"), "end": s.get("end"), "desc": s.get("desc")}
                  for s in shot_desc if s.get("desc")]
        sub = subs_text(structure.get("subtitles", []), ws, we)
        ocr = ocr_text(structure.get("ocr", []), ws, we)
        user = prompts.segment_desc_user(ws, we, briefs, sub, ocr, meta)
        text = native_video.describe_clip(clip, prompts.DESCRIBE_SYS, user,
                                          client.pc.vision_model, api_key, cfg.native_fps)
        from .llm_client import extract_json
        return extract_json(text)
    finally:
        if not cfg.native_clip_dir:
            try:
                os.remove(clip)
            except OSError:
                pass


def _video_path_from_meta(meta: Dict) -> str:
    return meta.get("media", {}).get("video_path", "")


def _align_language(client: LLMClient, cfg: RunConfig, desc: Dict,
                    issues: List[str]) -> Dict:
    """语言对齐(Res 2.3):缺侧翻译补齐 + 长度比粗查 + 可选回译校验。"""
    zh, en = (desc.get("zh") or "").strip(), (desc.get("en") or "").strip()
    if zh and not en:
        ok, en, err = retry_exp_backoff(
            lambda: client.complete(
                [{"role": "system", "content": prompts.TRANSLATE_SYS},
                 {"role": "user", "content": prompts.translate_user(zh, "en")}],
                want_json=True).get("text", ""))
        if not ok or not en:
            issues.append(f"英文翻译补齐失败: {err}")
            en = ""
        else:
            log.info("已用文本模型补齐英文侧")
    if en and not zh:
        ok, zh, err = retry_exp_backoff(
            lambda: client.complete(
                [{"role": "system", "content": prompts.TRANSLATE_SYS},
                 {"role": "user", "content": prompts.translate_user(en, "zh")}],
                want_json=True).get("text", ""))
        if not ok or not zh:
            issues.append(f"中文翻译补齐失败: {err}")
            zh = ""
        else:
            log.info("已用文本模型补齐中文侧")
    if zh and en:
        ratio = zh_len(zh) / max(1, en_len(en))
        if not (0.4 <= ratio <= 2.5):
            issues.append(f"zh/en 长度比 {ratio:.2f} 超出 [0.4, 2.5],存在语义不一致风险")
        if cfg.strict_lang_check:
            _back_translate_check(client, cfg, zh, en, issues)
    return {"zh": zh or None, "en": en or None}


def _back_translate_check(client: LLMClient, cfg: RunConfig, zh: str, en: str,
                          issues: List[str]) -> None:
    """回译校验(Res 2.3):en 翻回 zh 与原文 n-gram 相似度粗查。"""
    try:
        back = client.complete(
            [{"role": "system", "content": prompts.TRANSLATE_SYS},
             {"role": "user", "content": prompts.translate_user(en, "zh")}],
            want_json=True).get("text", "")
        from .utils import ngram_similarity
        if ngram_similarity(zh, back, n=4) < 0.5:
            issues.append("回译校验未通过:en->zh 与原文差异大")
    except Exception as e:  # noqa: BLE001
        log.debug("回译校验跳过: %s", e)


def _describe_shot_group(client: LLMClient, cfg: RunConfig, store: FrameStore,
                         grp: List[Dict], structure: Dict, meta: Dict,
                         cache) -> Optional[List[Dict]]:
    """单个镜头组的描述,返回组内每个镜头的 desc;未过质量门返回 None。

    抽成独立函数是为了让镜头组之间并发 —— 每组只读共享输入、只写自己那几个镜头,
    彼此无依赖,而这一级是整条流水线里调用次数最多的一段。
    """
    s0, eN = grp[0]["start"], grp[-1]["end"]
    if all((sh.get("desc") or {}).get("zh") for sh in grp):
        return [sh.get("desc") for sh in grp]
    key = {"stage": "shot_desc", "span": (s0, eN),
           "subs": subs_text(structure.get("subtitles", []), s0, eN)}
    cached = cache.get("describe", key) if cache else None
    if cached:
        return cached
    out = describe_shot(client, cfg, store, {"id": grp[0].get("id"), "start": s0, "end": eN},
                        structure, meta)
    if not out:
        return None
    desc = _align_language(client, cfg, out.get("description") or {}, [])
    descs = [desc] * len(grp)
    if cache:
        cache.put("describe", key, descs)
    return descs


def fill_shot_descs(client: LLMClient, cfg: RunConfig, store: FrameStore,
                    shots: List[Dict], structure: Dict, meta: Dict,
                    cache) -> Tuple[int, List[str]]:
    """填充 shot.desc(Res 2.1 镜头级;REVIEW 1-3 修复)。返回 (成功数, 失败清单)。"""
    # 微镜头(过短)合并成组,保证每次调用 ≥15s 内容、控制调用次数
    groups: List[List[Dict]] = []
    cur: List[Dict] = []
    for sh in shots:
        if not cur or (sh["end"] - cur[0]["start"]) < 15.0:
            cur.append(sh)
        else:
            groups.append(cur)
            cur = [sh]
    if cur:
        groups.append(cur)
    if not groups:
        return 0, []

    workers = max(1, min(cfg.workers, len(groups)))
    results: List[Optional[List[Dict]]] = [None] * len(groups)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_describe_shot_group, client, cfg, store, grp, structure,
                            meta, cache): i for i, grp in enumerate(groups)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001 —— 单组异常不拖垮整级描述
                log.warning("镜头组 %d 描述异常: %s", i + 1, e)

    # 结果按窗口顺序回填,保证 failed 清单与日志次序可复现
    ok_n, failed = 0, []
    for gi, (grp, descs) in enumerate(zip(groups, results), 1):
        s0, eN = grp[0]["start"], grp[-1]["end"]
        if descs:
            for sh, d in zip(grp, descs):
                sh["desc"] = d
            ok_n += len(grp)
        else:
            failed.append(f"[{s0:.0f}-{eN:.0f}] 镜头组 {gi} 描述未过质量门")
            log.warning("镜头组 %d [%.0f-%.0fs] 描述未过质量门", gi, s0, eN)
    log.info("镜头级描述:成功 %d / %d 个镜头,失败 %d 组(%d 组 / %d 并发)",
             ok_n, len(shots), len(failed), len(groups), workers)
    return ok_n, failed


def _describe_one_segment(client: LLMClient, cfg: RunConfig, store: FrameStore,
                          ws: float, we: float, shots: List[Dict], structure: Dict,
                          meta: Dict, cache, native: bool
                          ) -> Tuple[Optional[Dict], List[str]]:
    """单个片段的详述(可并发);返回 (segment, 语言对齐告警);未过质量门返回 (None, …)。"""
    in_shot = [sh for sh in shots if sh.get("start", 0) < we and sh.get("end", 0) > ws]
    key = {"stage": "segment_desc", "span": (ws, we),
           "shots": [sh.get("id") for sh in in_shot],
           "subs": subs_text(structure.get("subtitles", []), ws, we)}
    out = cache.get("describe", key) if cache else None
    if not out:
        out = (_native_segment if native else describe_segment)(
            client, cfg, store, ws, we, in_shot, structure, meta)
        if cache and out:
            cache.put("describe", key, out)
    if isinstance(out, dict):
        out = _norm_desc(out)
    if not out or not out.get("description"):
        return None, []
    issues: List[str] = []
    desc = _align_language(client, cfg, out.get("description") or {}, issues)
    return {"start": ws, "end": we, "description": desc,
            "elements": out.get("elements") or {}}, issues


def describe_video(client: LLMClient, cfg: RunConfig, video_path: str,
                   structure: Dict, meta: Dict, store: Optional[FrameStore] = None,
                   cache=None) -> Tuple[List[Dict], Dict, Dict]:
    """M1 主入口:镜头级 → 片段级 → 全片级;返回 (segments, detailed, quality_report)。

    quality_report: {shot_ok, shot_total, segment_ok, segment_total,
                     quiet_segments[], full_len_zh, lang_issues[]}
    """
    shots = structure.get("shots", [])
    if shots:
        windows = build_windows(shots, cfg.window_sec)
    else:
        windows = fixed_windows(float(meta["meta"]["duration_sec"]), cfg.window_sec)
    if cfg.max_windows > 0:
        windows = windows[: cfg.max_windows]

    native = cfg.describe_mode == "native_video"
    if native:
        if client.pc.name != "qwen":
            raise RuntimeError("native_video 模式仅支持云端 provider=qwen;Kimi/本地模型无原生视频接口,请用 --describe-mode frames")
        from . import native_video
        native_video.ensure_available()

    report: Dict = {"shot_ok": 0, "shot_total": len(shots), "segment_ok": 0,
                    "segment_total": len(windows), "quiet_segments": [],
                    "full_len_zh": 0, "lang_issues": []}
    assert store is not None, "M1 依赖帧索引(FrameStore),请先构建"
    store.set_duration(float(meta["meta"].get("duration_sec", 0) or 0))

    # 1) 镜头级
    if shots:
        ok, failed = fill_shot_descs(client, cfg, store, shots, structure, meta, cache)
        report["shot_ok"], report["shot_failed_groups"] = ok, failed
        log.info("镜头描述完成: %d/%d", ok, len(shots))

    # 2) 片段级(segment ≈ 180s 窗口),窗口之间无依赖,按窗并发
    segments: List[Dict] = []
    results: List[Tuple[Optional[Dict], List[str]]] = [(None, []) for _ in windows]
    if windows:
        workers = max(1, min(cfg.workers, len(windows)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_describe_one_segment, client, cfg, store, ws, we,
                                shots, structure, meta, cache, native): i
                    for i, (ws, we) in enumerate(windows)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:  # noqa: BLE001 —— 单段异常进 quiet_segments,不中断整级
                    log.warning("片段 %d 详述异常: %s", i + 1, e)
                    results[i] = (None, [f"片段 {i + 1} 详述异常: {e}"])
        log.info("片段级详述:%d 段 / %d 并发", len(windows), workers)

    # 结果按时间顺序回填,segments 与 quiet_segments 的次序保持可复现
    for wi, ((ws, we), (seg, issues)) in enumerate(zip(windows, results), 1):
        report["lang_issues"].extend(issues)
        if seg is None:
            report["quiet_segments"].append({"start": ws, "end": we,
                                             "reason": "质量门未过(重试后)"})
            log.warning("片段 %d [%.0f-%.0fs] 未过质量门,已列入 quiet_segments", wi, ws, we)
            continue
        segments.append(seg)
        report["segment_ok"] += 1

    # 3) 全片级:zh+en 双轨两级汇总(Res 2.1)
    detailed = aggregate_description(client, cfg, segments, meta, report)
    report["full_len_zh"] = zh_len(detailed.get("zh") or "")
    if report["full_len_zh"] < cfg.full_desc_min_zh:
        report["lang_issues"].append(
            f"全片详述中文 {report['full_len_zh']} 字 < {cfg.full_desc_min_zh}(需人工补写)")
    log.info("详述完成:%d 段 + 全片(中文 %d 字)",
             len(segments), report["full_len_zh"])
    return segments, detailed, report


def aggregate_description(client: LLMClient, cfg: RunConfig,
                          segments: List[Dict], meta: Dict,
                          report: Optional[Dict] = None) -> Dict:
    """两级汇总:段数 ≤ 分组阈值一次完成;否则分组→组内→合并(双语并轨)。"""
    if not segments:
        return {"zh": None, "en": None}
    items = [{"start": s["start"], "end": s["end"],
              "text_zh": (s["description"].get("zh") or ""),
              "text_en": (s["description"].get("en") or "")} for s in segments]

    def _agg(part: List[Dict]) -> Dict:
        out = client.complete(
            [{"role": "system", "content": prompts.DESCRIBE_AGG_SYS},
             {"role": "user", "content": prompts.describe_agg_user(part, meta)}],
            vision=False,
        )
        return out.get("detailed_description") or {"zh": None, "en": None}

    group = cfg.describe_agg_group or 15
    if len(items) <= group:
        detailed = _agg(items)
    else:
        parts: List[Dict] = []
        for i in range(0, len(items), group):
            chunk = items[i:i + group]
            d = _agg(chunk)
            parts.append({"start": chunk[0]["start"], "end": chunk[-1]["end"],
                          "text_zh": d.get("zh") or "", "text_en": d.get("en") or ""})
        log.info("两级汇总:%d 段 -> %d 组 -> 合并", len(items), len(parts))
        detailed = _agg(parts)

    issues: List[str] = []
    aligned = _align_language(client, cfg, detailed, issues)
    if report is not None:
        report.setdefault("lang_issues", []).extend(issues)
    return aligned
