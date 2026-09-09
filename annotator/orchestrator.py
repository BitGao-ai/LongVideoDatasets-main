# -*- coding: utf-8 -*-
"""编排器(Res 10.2)—— 阶段状态机 + 断点续跑 + 覆盖报告。

阶段:M0 帧索引 → S1 structure(窗口标注) → S2 describe(密集描述金字塔)
    → S3 events(事件引擎) → S4 aggregate(全局聚合) → S5 qa(锚定出题)
    → S6 verify(双模型核验) → S7 filter(五通道) → 难度标定 → 落盘。

断点恢复:
- manifest 记录每阶段输入内容哈希,哈希不变且已 done → 跳过重跑;
- 已 done 阶段的产物从上次落盘 JSON 恢复进内存(base 恢复),避免下游空跑;
- 窗口级 LLM 结果存 StageCache,中断后续跑不重复烧钱;
- 全阶段 done 且产物存在 → 直接复用(零成本复跑)。

审计:每次 LLM 调用写 trace(reports/llm_trace_<vid>.jsonl,Res 10.4)。
"""

import datetime
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from . import anti_shortcut, prompts, qa_engine, qa_verify
from .aggregate import aggregate_global
from .annotate import annotate_structure
from .config import RunConfig
from .dense_caption import describe_video as dense_describe_video
from .difficulty_tagger import tag_difficulty
from .events import run_event_engine
from .frames_store import FrameStore
from .llm_client import LLMClient
from .local_client import LocalLLMClient
from .manifest import (ALL_STAGES, STAGE_AGGREGATE, STAGE_DESCRIBE, STAGE_EVENTS,
                       STAGE_FILTER, STAGE_QA, STAGE_STRUCTURE, STAGE_VERIFY,
                       Manifest, StageCache, compute_structure_hash)
from .utils import duration_bucket, retry_exp_backoff, sha256_of

log = logging.getLogger("annotator.orchestrator")


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _video_duration(video_path: str, fallback: float = 0.0) -> float:
    import cv2
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0 and n > 0:
                return n / fps
    except Exception as e:  # noqa: BLE001
        log.warning("cv2 读时长失败: %s", e)
    return fallback


def _validate(annotation: Dict, schema_path: Optional[str]) -> str:
    """按 schema 校验产物;返回 '' 表示通过,否则返回错误摘要。

    调用方负责"先落盘、再报错":产物代表已经花掉的 LLM 开销,不能因为校验不过
    就丢掉;但也绝不能像此前那样只 log.warning —— 非法记录会静默进入数据集,
    连 jsonschema 没装都只是一行日志。
    """
    if not schema_path:
        return ""
    if not os.path.exists(schema_path):
        return f"schema 文件不存在: {schema_path}"
    try:
        import jsonschema
    except ImportError:
        return "未安装 jsonschema,无法校验产物(pip install jsonschema)"
    try:
        jsonschema.validate(annotation, _load_json(schema_path))
    except jsonschema.ValidationError as e:
        where = "/".join(str(p) for p in e.absolute_path) or "<root>"
        return f"{where}: {e.message}"[:300]
    except Exception as e:  # noqa: BLE001 —— schema 自身损坏也要报出来
        return f"schema 校验异常: {e}"[:300]
    log.info("Schema 校验通过")
    return ""


def _stage_hash(tag: str, payload: Dict) -> str:
    # 内部字段(_win / _split / …)不会落盘,也不该参与阶段哈希:它们只是中间过程的
    # 记账,内容相同的两次运行不能因为它们不同就被判成"输入已变"而重跑下游。
    return sha256_of({"tag": tag, **_strip_private(payload)})


def _merge_usage(*clients) -> Dict[str, int]:
    """合并各客户端的 token 用量,写进 manifest 以便核算单视频成本。"""
    total: Dict[str, int] = {}
    for c in clients:
        for k, v in (getattr(c, "usage", None) or {}).items():
            total[k] = total.get(k, 0) + int(v)
    return total


#: 不应随 JSON 落盘的内部字段(修复 D4)
_PRIVATE_PREFIX = "_"


def _strip_private(obj):
    """递归剔除以 _ 开头的内部键(_zh_len / _split / _win / _merged_into …)。"""
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith(_PRIVATE_PREFIX))}
    if isinstance(obj, list):
        return [_strip_private(v) for v in obj]
    return obj


def sync_structure(structure: Dict, scenes=None, events=None, segments=None) -> Dict:
    """把阶段产物写回 structure(修复 A1 / A2)。

    A1: scenes 此前只在局部变量里流转,落盘时从 structure 取到的永远是 preprocess
        写入的 []。
    A2: 跳过分支此前用 `structure.setdefault("events", events)`,而 preprocess 已经
        写入 `"events": []` —— key 存在,setdefault 不生效,断点续跑会清空产物。
        这里改为显式赋值;传 None 表示"该阶段未运行",保留原值。
    """
    for key, value in (("scenes", scenes), ("events", events), ("segments", segments)):
        if value is not None:
            structure[key] = value
    return structure


def build_annotation(video_id: str, meta: Dict, structure: Dict, global_block: Dict,
                     qa: List[Dict], manifest, annotators: List[str],
                     frames_dir: str) -> Dict:
    """装配最终落盘对象。抽成纯函数便于回归测试(A1/A2/D4)。"""
    tracks = ("shots", "scenes", "events", "segments", "subtitles", "ocr", "audio_events")
    annotation = {
        "schema_version": "2.0",
        "video_id": video_id,
        "meta": meta.get("meta", {}),
        "media": meta.get("media", {}),
        "structure": {k: structure.get(k) or [] for k in tracks},
        "global": global_block,
        "qa": qa,
        "annotation_meta": {
            "created_date": datetime.date.today().isoformat(),
            "annotators": annotators,
            "iaa_alpha": None,
            "review_status": "draft",
            "manifest_version": "2.0",
            **manifest.to_annotation_meta_extra(),
        },
    }
    annotation["structure"]["subtitles_meta"] = structure.get("subtitles_meta") or {}
    annotation["structure"]["frames_index_url"] = frames_dir
    annotation["annotation_meta"]["human_todo"] = manifest.data.get("human_todo", [])
    return _strip_private(annotation)


def _frame_captions(cfg: RunConfig, local: Optional[LocalLLMClient],
                    client: LLMClient, store: FrameStore, duration: float,
                    cache: StageCache, meta: Dict) -> List[Dict]:
    """帧级短描述(Res 2.1 帧级轨):本地模型全密度;无本地则云端降频。

    缓存键 = 抽帧率 + 字幕内容哈希;增量复用。
    """
    local_ok = bool(cfg.local_base_url and cfg.local_vision_model)
    fps = cfg.frame_rate if local_ok else cfg.frame_caption_cloud_fps
    key = {"stage": "captions", "fps": fps,
           "subs": sha256_of(meta.get("structure", {}).get("subtitles", []))[:16]}
    cached = cache.get("captions", key) if cache else None
    if cached:
        log.info("帧描述缓存复用:%d 条", len(cached))
        return cached
    times = store.sample_times(0.0, duration, max(1, int(duration * fps)))
    ctx = f"题材:{meta.get('meta', {}).get('genre')};主语言:{meta.get('meta', {}).get('primary_lang', '')}"
    caps: List[Dict] = []
    if local_ok and local is not None:
        caps = local.caption_frames(store, times, ctx)
    else:
        # 云端降频帧描述:逐帧串行时 180 帧 ≈ 6 分钟,按帧并发(C7)
        def _one(t: float) -> Optional[Dict]:
            b64 = store.frame_b64(t)
            if not b64:
                return None
            try:
                out = client.complete(
                    [{"role": "system", "content": prompts.CAPTION_SYS},
                     {"role": "user", "content": [prompts.caption_user(ctx),
                                                  {"type": "image_url",
                                                   "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
                    vision=True, want_json=False)
            except Exception as e:  # noqa: BLE001
                log.debug("云端帧描述失败(t=%.1f): %s", t, e)
                return None
            text = str(out or "").strip()[:60]
            return {"t": round(t, 3), "text": text} if text else None

        workers = max(1, min(cfg.workers, len(times) or 1))
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_one, t): t for t in times}
            for fut in as_completed(futs):
                item = fut.result()
                if item:
                    caps.append(item)
                done += 1
                if done % 100 == 0:
                    log.info("云端帧描述进度:%d/%d", done, len(times))
        caps.sort(key=lambda c: c["t"])
    if cache:
        cache.put("captions", key, caps)
    log.info("帧级描述:%d 条(fps=%.2f, local=%s)", len(caps), fps, local_ok)
    return caps


def annotate_video(meta_path: str, structure_path: str, out_path: str,
                   cfg: RunConfig, schema_path: Optional[str] = None) -> Dict:
    """端到端编排主入口(Res 10.2 状态机 + 断点恢复)。"""
    meta = _load_json(meta_path)
    structure_in = _load_json(structure_path)
    structure = structure_in.get("structure", structure_in)
    video_path = meta["media"]["video_path"]
    video_id = meta["video_id"]

    # 时长兜底(meta 缺失时从视频读)
    dur = float(meta["meta"].get("duration_sec", 0) or 0)
    if dur <= 0:
        dur = _video_duration(video_path)
        meta["meta"]["duration_sec"] = round(dur, 3)
    meta["meta"].setdefault("duration_bucket", duration_bucket(dur))

    # ---- 基础设施 ----
    struct_dir = os.path.dirname(os.path.abspath(structure_path)) or "."
    manifest = Manifest.load_or_create(video_id,
                                       os.path.join(struct_dir, f"{video_id}.manifest.json"))
    frames_dir = cfg.frame_index_dir or os.path.join(struct_dir, "frames")
    store = FrameStore(frames_dir, fps=cfg.frame_rate, max_edge=cfg.max_image_edge)
    store.set_duration(dur)
    store.build(video_path, force=cfg.force_frame_index)
    store.verify_index(dur)
    manifest.set_frames(**store.stats())

    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    cache = StageCache(os.path.join(out_dir, ".cache", video_id))

    client = LLMClient(cfg)
    client.set_trace(os.path.join(cfg.trace_dir, f"llm_trace_{video_id}.jsonl"))
    local = LocalLLMClient(cfg)
    verify_client: Optional[LLMClient] = None
    verify_name = ""
    if cfg.verify_answers:
        try:
            vpc = cfg.verify_provider_cfg()
            if vpc:
                vcfg = RunConfig(**{**cfg.__dict__, "provider": vpc.name})
                verify_client = LLMClient(vcfg)
                verify_client.set_trace(os.path.join(cfg.trace_dir,
                                                     f"llm_trace_{video_id}_verify.jsonl"))
                verify_name = vpc.name
        except Exception as e:  # noqa: BLE001 —— 无另一家 key 时降级跳过核验
            log.warning("另一家供应商不可用,答案核验降级跳过: %s", e)

    log.info("=== 标注 %s (provider=%s%s) 时长=%.0fs ===", video_id, cfg.provider,
             f" +verify:{verify_name}" if verify_client else "", dur)

    # ---- 全 done 且产物存在:直接复用(零成本复跑) ----
    if os.path.exists(out_path) and all(manifest.stage_done(s) for s in ALL_STAGES):
        log.info("全部阶段已完成且产物存在,直接复用: %s", out_path)
        return _load_json(out_path)

    # ---- base 恢复:已 done 阶段的产物从上次落盘 JSON 取回 ----
    base: Dict = _load_json(out_path) if os.path.exists(out_path) else {}
    scenes = list(base.get("structure", {}).get("scenes") or [])
    base_structure = base.get("structure") or {}
    events = list(base_structure.get("events") or [])
    segments = list(base_structure.get("segments") or [])
    base_global = base.get("global") or {}
    qa = list(base.get("qa") or [])

    # ============================================================ S1 structure
    s1_hash = compute_structure_hash(structure, meta)
    if manifest.needs_run(STAGE_STRUCTURE, s1_hash):
        log.info("---- S1 场景/事件窗口标注(重试 + gaps)----")
        manifest.clear_gaps()
        gaps: List[Dict] = []
        ok, res, err = retry_exp_backoff(
            lambda: annotate_structure(client, cfg, video_path, structure, meta,
                                       store=store, cache=cache, duration=dur,
                                       gaps=gaps),
            retries=3, logger=log)
        if ok:
            scenes, events = res
        else:
            log.error("S1 窗口标注重试 3 次仍失败:%s;全片记入 gaps,发布门禁将拦截", err)
            scenes, events = [], []
            # 抛异常退出时 annotate_structure 未必来得及写 gaps。此前 add_gap 只在成功
            # 分支执行,S1 整体失败反而留下一份空 gaps,coverage 门会误判为"全片已覆盖"。
            gaps = [{"start": 0.0, "end": dur,
                     "reason": f"S1 窗口标注整体失败: {str(err)[:200]}"}]
        for g in gaps:
            manifest.add_gap(g["start"], g["end"], g["reason"])
        manifest.set_stage(STAGE_STRUCTURE, "done" if ok else "failed", s1_hash,
                           error=str(err or ""))
    else:
        log.info("S1 已 done(输入哈希未变),恢复自上次产物")
    # 修复 A1:scenes 此前只存在于局部变量,从未写回 structure,落盘恒为 []
    structure["scenes"] = scenes

    # ============================================================ S2 describe
    # 修复 B17:阶段哈希必须包含上游产物,否则 S1 重跑后下游不会失效
    s2_hash = _stage_hash("describe", {"shots": structure.get("shots", []),
                                       "scenes": scenes,
                                       "dur": dur, "describe": cfg.describe})
    if cfg.describe and manifest.needs_run(STAGE_DESCRIBE, s2_hash):
        log.info("---- S2 密集描述金字塔(镜头→片段→全片)----")
        ok, res, err = retry_exp_backoff(
            lambda: dense_describe_video(client, cfg, video_path, structure, meta,
                                         store=store, cache=cache),
            retries=3, logger=log)
        if ok:
            segments, detailed, report = res
            structure["segments"] = segments
            structure["_detailed_description"] = detailed
            for qs in report.get("quiet_segments") or []:
                manifest.add_human_todo({"kind": "segment_quiet", **qs})
            manifest.set_stats(quiet_segments=report.get("quiet_segments", []),
                               shot_desc_ok=f"{report.get('shot_ok')}/{report.get('shot_total')}")
        else:
            log.error("S2 详述失败:%s", err)
        manifest.set_stage(STAGE_DESCRIBE, "done" if ok else "failed", s2_hash,
                           error=str(err or ""))
    elif not cfg.describe:
        log.info("S2 详述已禁用(describe=False)")
    else:
        log.info("S2 已 done(输入哈希未变),恢复自上次产物")
    # 修复 A2:跳过分支此前不回填,断点续跑会把 segments 清空
    structure["segments"] = segments

    # ============================================================ S3 events
    s3_hash = _stage_hash("events", {"subs": structure.get("subtitles", []),
                                     "audio": structure.get("audio_events", []),
                                     "scenes": scenes, "segments": segments,
                                     "dur": dur, "describe": cfg.describe})
    if manifest.needs_run(STAGE_EVENTS, s3_hash):
        log.info("---- S3 事件引擎(挖掘→8×8 边界复核→去重→关键性分级)----")
        captions = _frame_captions(cfg, local, client, store, dur, cache, meta)
        ok, res, err = retry_exp_backoff(
            lambda: run_event_engine(client, cfg, structure, meta, captions, store,
                                     events, scenes, dur),
            retries=3, logger=log)
        if ok:
            evs, ev_stats, ev_todos = res
            events = evs
            # 修复 A5:非法/被否决事件此前只在函数内产出后丢弃,从未进人工清单
            for td in ev_todos or []:
                manifest.add_human_todo(td)
            manifest.set_stats(**{k: v for k, v in ev_stats.items()})
        else:
            log.error("S3 事件引擎失败:%s", err)
        manifest.set_stage(STAGE_EVENTS, "done" if ok else "failed", s3_hash,
                           error=str(err or ""))
    else:
        log.info("S3 已 done(输入哈希未变),恢复自上次产物")
    # 修复 A2:此前用 setdefault,而 preprocess 已写入 "events": [],key 存在则不生效
    structure["events"] = events

    # ============================================================ S4 aggregate
    s4_hash = _stage_hash("aggregate", {"events": structure.get("events", []),
                                        "scenes": scenes, "dur": dur})
    if manifest.needs_run(STAGE_AGGREGATE, s4_hash):
        log.info("---- S4 全局聚合(主题/转折/伏笔/因果核验)----")
        ok, res, err = retry_exp_backoff(
            lambda: aggregate_global(client, cfg, scenes, structure.get("events", []), meta),
            retries=3, logger=log)
        if ok:
            global_block, evs, agg_report = res
            events = evs
            structure["events"] = evs
            for inv in agg_report.get("causal_invalid") or []:
                manifest.add_human_todo({"kind": "causal_invalid", **inv})
            for sa in agg_report.get("scene_alignment") or []:
                manifest.add_human_todo({"kind": "scene_alignment", **sa})
            for cu in agg_report.get("character_unmerged") or []:
                manifest.add_human_todo(cu)
        else:
            log.error("S4 全局聚合失败:%s", err)
            global_block = {"synopsis": {"zh": None, "en": None}, "characters": [],
                            "relations": [], "timeline": [], "themes": [],
                            "turning_points": [], "motifs": []}
        manifest.set_stage(STAGE_AGGREGATE, "done" if ok else "failed", s4_hash,
                           error=str(err or ""))
    else:
        global_block = dict(base_global)

    global_block.setdefault("detailed_description",
                            structure.pop("_detailed_description", None)
                            or base_global.get("detailed_description")
                            or {"zh": None, "en": None})

    # ============================================================ S5 qa
    s5_hash = _stage_hash("qa", {"events": structure.get("events", []),
                                 "scenes": scenes, "dur": dur})
    if manifest.needs_run(STAGE_QA, s5_hash):
        log.info("---- S5 锚定出题 + 能力矩阵硬约束 ----")
        ok, res, err = retry_exp_backoff(
            lambda: qa_engine.generate_qa(client, cfg, scenes, structure.get("events", []),
                                          global_block, meta, structure=structure, duration=dur),
            retries=3, logger=log)
        if ok:
            qa, cov = res
            manifest.set_stats(qa_coverage=cov)
        else:
            log.error("S5 出题失败:%s", err)
        manifest.set_stage(STAGE_QA, "done" if ok else "failed", s5_hash,
                           error=str(err or ""))
    else:
        log.info("S5 已 done(复用缓存)")

    # ============================================================ S6 verify
    s6_hash = _stage_hash("verify", {"qa": qa, "verify": cfg.verify_answers})
    if manifest.needs_run(STAGE_VERIFY, s6_hash):
        log.info("---- S6 双模型答案核验 + 梗概泄漏防护 ----")
        ok, res, err = retry_exp_backoff(
            lambda: qa_verify.verify_qa_answers(qa, client, verify_client, cfg,
                                                store, global_block),
            retries=3, logger=log)
        if ok:
            qa, vstats = res
            manifest.set_stats(**{f"verify_{k}": v for k, v in vstats.items() if v is not None})
        else:
            log.error("S6 核验失败:%s", err)
        manifest.set_stage(STAGE_VERIFY, "done" if ok else "failed", s6_hash,
                           error=str(err or ""))
    else:
        log.info("S6 已 done(复用缓存)")

    # ============================================================ S7 filter
    s7_hash = _stage_hash("filter", {"qa": qa, "keep_single_frame": cfg.keep_single_frame})
    if manifest.needs_run(STAGE_FILTER, s7_hash):
        log.info("---- S7 五通道抗捷径过滤(含单帧剔除修复)----")
        syn = (global_block.get("synopsis") or {})
        synopsis = syn.get("zh") or syn.get("en")
        # S6 已经对每道 MCQ 跑过同口径的梗概泄漏检测并剔除了命中项,这里再跑一遍
        # 是纯重复调用(B12)。S6 被跳过时才让 S7 承担该通道。
        if manifest.stage_done(STAGE_VERIFY) and cfg.synopsis_leak_check:
            synopsis = None
            log.info("梗概泄漏通道已在 S6 执行,S7 跳过以免重复调用")
        ok, res, err = retry_exp_backoff(
            lambda: anti_shortcut.run_filters(client, cfg, video_path, qa, structure,
                                              store=store, synopsis=synopsis),
            retries=3, logger=log)
        if ok:
            qa, fstats = res
            manifest.set_stats(filter_kept=fstats["kept"], filter_dropped=fstats["dropped"],
                               filter_channels=fstats["by_channel"])
        else:
            log.error("S7 过滤失败:%s", err)
        manifest.set_stage(STAGE_FILTER, "done" if ok else "failed", s7_hash,
                           error=str(err or ""))
    else:
        log.info("S7 已 done(复用缓存)")

    # ============================================================ 难度 + 落盘
    tag_difficulty(qa)
    sync_structure(structure, scenes=scenes, events=events, segments=segments)
    manifest.set_stats(qa_final=len(qa), event_count=len(structure.get("events", [])),
                       scene_count=len(scenes), segment_count=len(segments),
                       llm_usage=_merge_usage(client, verify_client))

    annotation = build_annotation(
        video_id=video_id, meta=meta, structure=structure, global_block=global_block,
        qa=qa, manifest=manifest,
        annotators=[f"auto:{client.pc.name}:{client.pc.vision_model}"],
        frames_dir=store.frames_dir,
    )

    schema_error = _validate(annotation, schema_path)
    annotation["annotation_meta"]["schema_valid"] = not schema_error
    if schema_error:
        annotation["annotation_meta"]["schema_error"] = schema_error

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)
    manifest.save()
    log.info("已写出标注结果: %s (%d 题 / %d 事件 / %d 场景 / %d 详述段)", out_path,
             len(qa), len(structure.get("events", [])), len(scenes), len(segments))
    if schema_error:
        raise RuntimeError(f"产物未通过 schema 校验(已落盘 {out_path} 供排查): {schema_error}")
    return annotation
