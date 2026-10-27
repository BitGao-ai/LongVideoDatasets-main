#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""长视频预处理 M0(Res 1):镜头分割 + ASR + OCR + 帧索引化 + 音频事件轨 + manifest。

产出与标注 schema 的 structure 段对齐的 JSON:
  {"video_id": "...", "structure": {"shots":[], "scenes":[], "events":[],
                                    "subtitles":[], "ocr":[], "audio_events":[]}}
scenes / events 故意留空 —— 由 annotator(LLM)或人工生成。

输出位置(视频 ID = 文件名去扩展名,如 videos/doc_00001.mp4 → doc_00001):
  批量模式(--videos-dir):
    <structure-dir>/<id>.json            结构 JSON(默认 structure/,--structure-dir 改)
    <structure-dir>/frames/<id>/         帧索引(1fps jpg,文件名即时间戳)
    <structure-dir>/<id>.manifest.json   断点续跑 manifest
  单视频模式(--video):--out 指定结构 JSON 路径,
    frames/ 与 manifest 放置在 --out 同目录;--out 缺省写入 <structure-dir>/<id>.json

v2 变化(Res 1):
- 帧索引化:ffmpeg 1fps 导出 structure/frames/{video_id}/f_{t:07d}_{ms:03d}.jpg,
  文件名即时间戳,后续抽帧/单帧不再二次 seek(替换 CAP_PROP_POS_MSEC);
  附 verify_index 校验(单调/首帧≈0/末帧≈时长,不一致报错)。
- 镜头关键帧(首/中/尾)路径写入 shot.keyframe_path。
- 音频事件轨:ASR 空白 >0.5s 且能量突增 → {start,end,type,desc};无则 []。
- OCR 保留 bbox 与 confidence(不再置 null,Res 1.4)。
- 说话人分离失败 → speaker=null 且 subtitles.meta.diarize_failed=true(不静默)。
- manifest:structure/{video_id}.manifest.json 固化 阶段产物清单(断点续跑基础)。

依赖(建议独立虚拟环境):
  pip install "scenedetect[opencv]" whisperx paddleocr paddlepaddle-gpu opencv-python
用法:
  # 单视频(与旧版一致)
  python preprocess.py --video videos/doc_00001.mp4 --video-id doc_00001 \
      --lang en --out structure/doc_00001.json --hf-token $HF_TOKEN
  # 批量:videos/ 下所有视频逐个预处理,structure 输出到 structure/<id>.json
  python preprocess.py --videos-dir videos --lang en \
      --structure-dir structure --hf-token $HF_TOKEN
"""

import argparse
import datetime
import gc
import glob
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass

import cv2

from annotator.audio_events import detect_audio_events
from annotator.frames_store import FrameStore
from annotator.local_client import LocalLLMClient
from annotator.manifest import Manifest
from annotator.config import RunConfig
from annotator.utils import quiet_http_loggers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
quiet_http_loggers()
log = logging.getLogger("preprocess")

# whisperx 是懒加载：`import whisperx` 不引 pyannote，真触发是后面的
# load_model -> vads/pyannote.py -> `from pyannote.audio import Model`。
# 按文案过滤不可靠(message 是 re.match 且以换行开头)，按模块过滤，一行管住所有懒加载路径。
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io")


def _tc_sec(tc: object) -> float:
    """取 FrameTimecode 秒数：优先 .seconds(0.7+)，旧版回退 get_seconds()。"""
    sec = getattr(tc, "seconds", None)
    if sec is not None:
        try:
            return float(sec)
        except (TypeError, ValueError):
            pass
    return float(tc.get_seconds())  # noqa: SLF001 —— 仅兼容 scenedetect<0.6


@dataclass
class Config:
    scene_threshold: float = 27.0
    ocr_interval_sec: float = 2.0
    ocr_min_conf: float = 0.80
    ocr_lang: str = "ch"           # 'ch' 同时识别中英;纯英可用 'en'
    whisper_model: str = "large-v3"
    whisper_batch: int = 16
    device: str = "cuda"           # 无 GPU 改 "cpu"
    compute_type: str = "float16"  # cpu 时改 "int8"
    frame_rate: float = 1.0        # 帧索引 fps(Res 1.1)


def free_gpu(*objs) -> None:
    """显式释放显存(Res 1.5 / 修复 C1)。

    WhisperX(PyTorch)与 PaddleOCR(PaddlePaddle)在同一个进程里串行运行,
    但 PyTorch 的 caching allocator 不会把显存还给驱动,PaddlePaddle 又是
    另一套 allocator 拿不到这块显存 —— 16GB 卡上 OCR 阶段极易 OOM。
    这里在每个重模型用完后立刻回收。
    """
    for o in objs:
        try:
            del o
        except NameError:
            pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            free, total = torch.cuda.mem_get_info()
            log.info("显存已回收:可用 %.1f/%.1f GiB", free / 2 ** 30, total / 2 ** 30)
    except Exception as e:  # noqa: BLE001 —— 无 torch / CPU 环境直接跳过
        log.debug("跳过显存回收: %s", e)


def detect_shots(video_path: str, cfg: Config, store: FrameStore,
                 duration_sec: float = 0.0) -> list:
    from scenedetect import ContentDetector, detect
    log.info("镜头分割中 ...(threshold=%.1f)", cfg.scene_threshold)
    # start_in_scene=True: 无切点时返回 1 个全片镜头而非 [](单机位口播/会议最常见)。
    # 不传该参数时 detect() 默认返回 []，调用方会误以为“检出 0 个镜头”是正常结果。
    scenes = detect(video_path, ContentDetector(threshold=cfg.scene_threshold),
                    start_in_scene=True)
    if not scenes:
        # 双保险：极端情况下(如解码失败但未抛错)仍兜底为单镜头，避免下游 M1/M2 空跑。
        dur = duration_sec or (store._duration if store is not None else 0.0)
        if dur > 0:
            log.warning("镜头分割 0 检出，已兜底为单镜头 [0, %.3f](阈值 %.1f 过高或单机位长镜头)。"
                        "如需更细切分可降 --scene-threshold(如 20/15)重跑",
                        dur, cfg.scene_threshold)
            scenes = None  # 标记走兜底分支
        else:
            log.warning("镜头分割 0 检出且未知时长，返回空(下游可能空跑)")
            return []
    if scenes is None:
        kf = store.shot_keyframe_times(0.0, dur) if store is not None else []
        kf_paths = [os.path.join(store.frames_dir, f"f_{int(t):07d}_{int(round((t - int(t)) * 1000)):03d}.jpg")
                    for t in kf] if store is not None else []
        return [{
            "id": 1,
            "start": 0.0,
            "end": round(dur, 3),
            "desc": {"zh": None, "en": None},
            "keyframe_path": kf_paths[1] if len(kf_paths) > 1 else (kf_paths[0] if kf_paths else None),
            "keyframes": kf,
        }]
    shots = []
    for i, (start, end) in enumerate(scenes, start=1):
        s, e = round(_tc_sec(start), 3), round(_tc_sec(end), 3)
        kf = store.shot_keyframe_times(s, e) if store is not None else [s, (s + e) / 2.0, e]
        # 关键帧文件路径(frames 目录内,文件名即时间戳)
        kf_paths = [os.path.join(store.frames_dir, f"f_{int(t):07d}_{int(round((t - int(t)) * 1000)):03d}.jpg")
                    for t in kf] if store is not None else []
        shots.append({
            "id": i,
            "start": s,
            "end": e,
            "desc": {"zh": None, "en": None},   # 由 M1 密集描述引擎填充(Res 2.1)
            "keyframe_path": kf_paths[1] if kf_paths else None,
            "keyframes": kf,
        })
    # 单镜头长视频(如 15min 口播)明确提示，避免用户误以为阈值失效
    if len(shots) == 1:
        log.info("仅 1 个镜头(全片一镜，时长 %.1fs)：属正常(单机位/无硬切)。"
                 "如需更细切分可降 --scene-threshold 重跑", shots[0]["end"] - shots[0]["start"])
    else:
        log.info("检出 %d 个镜头", len(shots))
    return shots


def transcribe(video_path: str, lang: str, cfg: Config, hf_token) -> tuple:
    """WhisperX 转写。每个重模型用完立刻释放显存(C1),否则后续 OCR 会 OOM。"""
    # torchcodec 缺共享 FFmpeg 库时的 UserWarning 已在顶层按模块压制(懒加载路径太多，
    # 函数内 catch 包不住)。这里只保留一行 INFO，避免用户误以为转写卡死。
    # 本阶段 ASR 走 whisperx.load_audio(ffmpeg/静态二进制)，不依赖 torchcodec；
    # 只有传了 --hf-token 的说话人分离才需要它(失败已有 diarize_failed 兜底)。
    import whisperx
    log.info("WhisperX 转写中 (lang=%s, model=%s, device=%s, batch=%d) ...",
             lang, cfg.whisper_model, cfg.device, cfg.whisper_batch)
    try:
        import torchcodec  # noqa: F401 —— 仅做可用性探测，缺失不阻塞 ASR
    except Exception:
        log.info("torchcodec 不可用(多为 Mac 静态 ffmpeg 缺共享库)，ASR 不受影响；"
                 "如需说话人分离请 `brew install ffmpeg` 后重装 torchcodec")
    audio = whisperx.load_audio(video_path)

    model = whisperx.load_model(cfg.whisper_model, cfg.device,
                                compute_type=cfg.compute_type, language=lang)
    result = model.transcribe(audio, batch_size=cfg.whisper_batch)
    free_gpu(model)                      # ASR 主模型:权重 + batch 激活,占用最大

    align_model, meta = whisperx.load_align_model(language_code=result["language"],
                                                  device=cfg.device)
    result = whisperx.align(result["segments"], align_model, meta, audio, cfg.device)
    free_gpu(align_model)

    diarize_failed = False
    if hf_token:
        try:
            diarize = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=cfg.device)
            result = whisperx.assign_word_speakers(diarize(audio), result)
            free_gpu(diarize)
        except Exception as e:
            diarize_failed = True
            log.warning("说话人分离失败(speaker=null,diarize_failed=true): %s", e)
    free_gpu(audio)                      # 2h 视频的整轨 float32 约 460MB(C3)

    subs = []
    for seg in result["segments"]:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        subs.append({
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": text,
            "lang": lang,
            "speaker": seg.get("speaker"),
        })
    subs_meta = {"diarize_failed": diarize_failed}
    log.info("转写得到 %d 条字幕(说话人分离失败=%s)", len(subs), diarize_failed)
    return subs, subs_meta


def run_ocr(video_path: str, cfg: Config, store=None) -> list:
    """OCR:保留 bbox 与 confidence(Res 1.4,不再置 null)。

    修复 C8:优先直接读【已导出的 1fps 帧索引】,不再对视频逐帧
    `cap.set(CAP_PROP_POS_FRAMES, idx)` 随机 seek —— 那既慢(每次要回退到
    关键帧重新解码 GOP,2h 视频 3600 次),又在很多容器上定位不准,而且正是
    frames_store 声明要废除的做法。帧索引不可用时才回退顺序解码。
    """
    from paddleocr import PaddleOCR
    try:
        # PaddleOCR 3.x:use_angle_cls/show_log 已移除,用 use_textline_orientation;
        # predict() 返回含 rec_texts/rec_scores/rec_boxes 的结果对象(见下 _parse_v3)
        ocr = PaddleOCR(lang=cfg.ocr_lang, use_textline_orientation=True)
        _ocr_v3 = True
    except TypeError:
        # PaddleOCR 2.x 回退
        ocr = PaddleOCR(use_angle_cls=True, lang=cfg.ocr_lang, show_log=False)
        _ocr_v3 = False

    raw = []

    def _iter_v2(res):
        """2.x:ocr.ocr() -> [[bbox, (text, conf)], ...]"""
        for line in (res[0] or []) if res else []:
            try:
                txt, conf = line[1][0], line[1][1]
                bbox = line[0]
            except (IndexError, TypeError, KeyError):
                continue
            yield txt, conf, bbox

    def _iter_v3(out):
        """3.x:predict() -> [OCRResult{rec_texts/rec_scores/rec_polys/rec_boxes}]"""
        for r in out or []:
            try:
                if isinstance(r, dict) or hasattr(r, "get"):
                    texts = r.get("rec_texts") or []
                    scores = r.get("rec_scores") or []
                    polys = r.get("rec_polys") or r.get("rec_boxes") or []
                else:
                    texts = getattr(r, "rec_texts", []) or []
                    scores = getattr(r, "rec_scores", []) or []
                    polys = getattr(r, "rec_polys", None)
                    if polys is None:
                        polys = getattr(r, "rec_boxes", []) or []
                for i, txt in enumerate(texts):
                    conf = float(scores[i]) if i < len(scores) else 0.0
                    box = polys[i] if i < len(polys) else None
                    yield txt, conf, box
            except Exception:
                continue

    def _collect(frame, t: float) -> None:
        try:
            if _ocr_v3:
                res = ocr.predict(frame)
                items = _iter_v3(res)
            else:
                res = ocr.ocr(frame, cls=True)
                items = _iter_v2(res)
        except Exception as e:
            log.warning("第 %.1fs OCR 失败: %s", t, e)
            return
        for txt, conf, bbox in items:
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                continue
            if conf >= cfg.ocr_min_conf and (txt or "").strip():
                try:
                    import numpy as _np
                    if isinstance(bbox, _np.ndarray):
                        bbox = bbox.tolist()
                    flat = [round(float(v), 1) for pt in (bbox or []) for v in
                            (pt if isinstance(pt, (list, tuple)) else [pt])]
                except Exception:
                    flat = []
                raw.append({"t": round(t, 3), "text": txt.strip(), "conf": conf,
                            "bbox": flat})

    frame_times = store.index_times() if store is not None else []
    if frame_times:
        step = max(1, int(round(cfg.ocr_interval_sec * (store.fps or 1.0))))
        picked = frame_times[::step]
        log.info("OCR 复用帧索引: %d/%d 帧(每 %.1fs 一帧,零解码)",
                 len(picked), len(frame_times), cfg.ocr_interval_sec)
        for t in picked:
            img = store.read_frame(t)
            if img is not None:
                _collect(img, t)
    else:
        log.info("无帧索引可用,OCR 回退顺序解码")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps * cfg.ocr_interval_sec)))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                _collect(frame, idx / fps)
            idx += 1
        cap.release()

    spans = []
    for item in raw:
        t, text = item["t"], item["text"]
        if spans and spans[-1]["text"] == text and (t - spans[-1]["end"]) <= cfg.ocr_interval_sec * 1.5:
            spans[-1]["end"] = t
        else:
            spans.append({"start": t, "end": t, "text": text,
                          "bbox": item["bbox"], "confidence": round(item["conf"], 3)})
    log.info("OCR 得到 %d 个文本区间(bbox/confidence 已保留)", len(spans))
    return spans


VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".ts")


def collect_videos(videos_dir: str) -> list:
    files = []
    for ext in VIDEO_EXTS:
        files.extend(sorted(glob.glob(os.path.join(videos_dir, f"*{ext}"))))
    return files


def preprocess_video(video: str, video_id: str, lang: str, out: str, cfg: Config,
                     hf_token: str = "", skip_ocr: bool = False,
                     skip_frames: bool = False, local_base_url: str = "",
                     local_model: str = "") -> bool:
    """单视频预处理:帧索引 + 镜头分割 + ASR + OCR + 音频事件轨 + manifest。成功返回 True。"""
    duration = 0.0
    cap = cv2.VideoCapture(video)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = n / fps if n and fps else 0.0
        cap.release()
    if duration <= 0:
        log.error("无法读取视频时长:%s", video)
        return False

    out_dir = os.path.dirname(out) or "."
    frames_dir = os.path.join(out_dir, "frames", video_id)

    # ---- 帧索引化(Res 1.1) ----
    store = None
    if not skip_frames:
        try:
            store = FrameStore(frames_dir, fps=cfg.frame_rate, max_edge=768,
                               video_id=video_id)
            store.set_duration(duration)
            store.build(video)
            store.verify_index(duration)
        except Exception as e:
            log.error("帧索引构建失败:%s(标注阶段会再尝试)", e)
            store = None

    structure = {"shots": [], "scenes": [], "events": [], "subtitles": [], "ocr": [],
                 "audio_events": []}

    for name, fn in (("镜头分割", lambda: detect_shots(video, cfg, store, duration)
                      if store else _detect_shots_no_store(video, cfg, duration)),
                     ("ASR", lambda: transcribe(video, lang, cfg, hf_token))):
        try:
            if name == "镜头分割":
                structure["shots"] = fn()
            else:
                subs, subs_meta = fn()
                structure["subtitles"] = subs
                structure["subtitles_meta"] = subs_meta
        except Exception as e:
            log.error("%s 失败: %s", name, e)

    if not skip_ocr:
        try:
            structure["ocr"] = run_ocr(video, cfg, store=store)
        except Exception as e:
            log.error("OCR 失败: %s", e)
        finally:
            free_gpu()          # PaddleOCR 之后也回收一次,批量模式下逐部累积会 OOM

    # ---- 音频事件轨(Res 1.3) ----
    try:
        rcfg = RunConfig(local_base_url=local_base_url,
                         local_vision_model=local_model)
        local = LocalLLMClient(rcfg)
        structure["audio_events"] = detect_audio_events(
            video, structure.get("subtitles", []), duration, local_client=local)
    except Exception as e:
        log.error("音频事件轨失败(置空): %s", e)
        structure["audio_events"] = []

    os.makedirs(out_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"video_id": video_id, "structure": structure}, f,
                  ensure_ascii=False, indent=2)
    log.info("已写出: %s", out)

    # ---- manifest(Res 1.5) ----
    manifest = Manifest.load_or_create(video_id,
                                       os.path.join(out_dir, f"{video_id}.manifest.json"))
    frame_count = store.stats()["frame_count"] if store else 0
    manifest.set_frames(frames_dir=frames_dir, fps=cfg.frame_rate,
                        frame_count=frame_count, duration_sec=duration)
    manifest.set_stats(shots=len(structure["shots"]), subtitles=len(structure["subtitles"]),
                       ocr=len(structure["ocr"]), audio_events=len(structure["audio_events"]))
    manifest.save()
    log.info("manifest 已写出: %s", manifest.path)
    return True


def main():
    p = argparse.ArgumentParser(description="长视频预处理 M0:镜头分割 + ASR + OCR + 帧索引 + 音频事件轨 + manifest")
    p.add_argument("--video", default="", help="单视频模式:视频路径(与 --videos-dir 二选一)")
    p.add_argument("--video-id", default="", help="单视频模式:视频 ID(缺省取文件名)")
    p.add_argument("--videos-dir", default="", help="批量模式:视频所在目录,处理其中所有视频")
    p.add_argument("--lang", choices=["zh", "en"], required=True)
    p.add_argument("--out", default="", help="单视频模式:structure JSON 输出路径(缺省写入 <structure-dir>/<id>.json)")
    p.add_argument("--structure-dir", default="structure",
                   help="批量模式:输出目录,含 <id>.json / frames/<id>/ / <id>.manifest.json(默认 structure)")
    p.add_argument("--force", action="store_true", help="批量模式:已存在 structure 也重跑(默认跳过)")
    p.add_argument("--hf-token", nargs="?", const="", default="",
                   help="HuggingFace token(说话人分离用);缺省或留空时读环境变量 HF_TOKEN")
    p.add_argument("--skip-ocr", action="store_true")
    p.add_argument("--skip-frames", action="store_true", help="跳过帧索引化(标注时兜底重建)")
    p.add_argument("--frame-rate", type=float, default=1.0, help="帧索引抽帧率")
    # ---- 计算资源 / 模型(修复 C2:此前全部写死在 Config 里,无 GPU 必须改源码)----
    g = p.add_argument_group("计算资源与模型")
    g.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                   help="WhisperX 推理设备(默认 cuda;无 GPU 用 cpu)")
    g.add_argument("--compute-type", default="",
                   help="WhisperX 计算精度(默认 cuda→float16 / cpu→int8;显存不足可用 int8_float16)")
    g.add_argument("--whisper-model", default="large-v3",
                   help="WhisperX 模型(默认 large-v3；CPU/Mac 建议 small/medium，首次需下载数 GB)")
    g.add_argument("--whisper-batch", type=int, default=None,
                   help="WhisperX batch_size(默认 cuda=16/cpu=4；CPU 上 >8 极慢且易 OOM)")
    g.add_argument("--scene-threshold", type=float, default=27.0, help="镜头分割敏感度")
    g.add_argument("--ocr-lang", default="ch", help="PaddleOCR 语言(ch=中英,en=纯英)")
    g.add_argument("--ocr-interval-sec", type=float, default=2.0, help="OCR 采样间隔(秒)")
    g.add_argument("--ocr-min-conf", type=float, default=0.80, help="OCR 置信度下限")
    p.add_argument("--local-base-url", default="", help="本地小模型端点(vLLM),用于音频事件类型判别")
    p.add_argument("--local-model", default="", help="本地视觉小模型名(如 Qwen2.5-VL-8B-Instruct)")
    args = p.parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")

    compute_type = args.compute_type or ("float16" if args.device == "cuda" else "int8")
    if args.device == "cpu" and compute_type == "float16":
        p.error("--device cpu 不支持 --compute-type float16，请用 int8(默认)或 int8_float16")
    # CPU 默认 batch 从 16 降到 4：large-v3+batch16 在 Mac CPU 上又慢又吃内存，
    # 是本次“卡在 tokenizer.json 后无响应”体感的第二主因(第一是数 GB 模型下载)。
    whisper_batch = args.whisper_batch
    if whisper_batch is None:
        whisper_batch = 16 if args.device == "cuda" else 4
    elif args.device == "cpu" and whisper_batch > 8:
        log.warning("CPU 上 batch=%d 过大(建议 ≤4)，转写会显著变慢，已继续但请预期数小时",
                    whisper_batch)
    cfg = Config(frame_rate=args.frame_rate, device=args.device,
                 compute_type=compute_type, whisper_model=args.whisper_model,
                 whisper_batch=whisper_batch, scene_threshold=args.scene_threshold,
                 ocr_lang=args.ocr_lang, ocr_interval_sec=args.ocr_interval_sec,
                 ocr_min_conf=args.ocr_min_conf)
    log.info("计算配置: device=%s compute_type=%s whisper=%s batch=%d frame_rate=%.2f",
             cfg.device, cfg.compute_type, cfg.whisper_model, cfg.whisper_batch,
             cfg.frame_rate)
    if args.device == "cpu" and cfg.whisper_model.startswith("large"):
        log.warning("CPU 跑 %s：首次需下载 ~3GB(出现 .incomplete 属断点续传，未卡死)；"
                    "15min 视频 CPU 转写约需 1~3h。着急请改 --whisper-model small/medium "
                    "+ --whisper-batch 4，或先 --skip-ocr 验证链路",
                    cfg.whisper_model)

    if args.video:
        video_id = args.video_id or os.path.splitext(os.path.basename(args.video))[0]
        tasks = [(args.video, video_id,
                  args.out or os.path.join(args.structure_dir, f"{video_id}.json"))]
    elif args.videos_dir:
        if not os.path.isdir(args.videos_dir):
            p.error(f"目录不存在: {args.videos_dir}")
        os.makedirs(args.structure_dir, exist_ok=True)
        tasks = [(v, os.path.splitext(os.path.basename(v))[0],
                  os.path.join(args.structure_dir,
                               f"{os.path.splitext(os.path.basename(v))[0]}.json"))
                 for v in collect_videos(args.videos_dir)]
    else:
        p.error("必须提供 --video(单视频)或 --videos-dir(批量)")

    if not tasks:
        log.error("未找到视频文件")
        sys.exit(1)
    if not args.video:
        log.info("批量预处理 %d 部视频", len(tasks))
        log.info("输出目录: %s/(<id>.json, frames/<id>/, <id>.manifest.json)",
                 os.path.abspath(args.structure_dir))

    failed = []
    for i, (video, video_id, out) in enumerate(tasks, 1):
        if os.path.exists(out) and not args.force:
            log.info("[%d/%d] 跳过(已存在): %s", i, len(tasks), out)
            continue
        log.info("[%d/%d] 预处理: %s", i, len(tasks), video)
        try:
            ok = preprocess_video(video, video_id, args.lang, out, cfg,
                                  hf_token=hf_token, skip_ocr=args.skip_ocr,
                                  skip_frames=args.skip_frames,
                                  local_base_url=args.local_base_url,
                                  local_model=args.local_model)
        except Exception as e:  # noqa: BLE001
            log.error("预处理异常: %s", e)
            ok = False
        if not ok:
            failed.append(video)

    if failed:
        log.error("批次完成,失败 %d 部: %s", len(failed), ", ".join(failed))
        sys.exit(1)
    log.info("批次全部完成,共 %d 部;产物见 %s/", len(tasks), os.path.abspath(args.structure_dir))


def _detect_shots_no_store(video_path: str, cfg: Config, duration_sec: float = 0.0) -> list:
    """无帧索引时的镜头分割兜底(仅 id/start/end,keyframe 置 None)。"""
    from scenedetect import ContentDetector, detect
    scenes = detect(video_path, ContentDetector(threshold=cfg.scene_threshold),
                    start_in_scene=True)
    if scenes:
        return [{"id": i, "start": round(_tc_sec(s), 3), "end": round(_tc_sec(e), 3),
                 "desc": {"zh": None, "en": None}, "keyframe_path": None, "keyframes": []}
                for i, (s, e) in enumerate(scenes, start=1)]
    if duration_sec <= 0:
        # 最后尝试从视频头读时长，保证单镜头兜底可用
        try:
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(video_path)
            if _cap.isOpened():
                _fps = _cap.get(_cv2.CAP_PROP_FPS) or 25.0
                _n = _cap.get(_cv2.CAP_PROP_FRAME_COUNT)
                if _n and _fps:
                    duration_sec = _n / _fps
                _cap.release()
        except Exception:
            pass
    if duration_sec > 0:
        log.warning("镜头分割 0 检出(--skip-frames 路径)，已兜底为单镜头")
        return [{"id": 1, "start": 0.0, "end": round(duration_sec, 3),
                 "desc": {"zh": None, "en": None}, "keyframe_path": None, "keyframes": []}]
    return []


if __name__ == "__main__":
    main()
