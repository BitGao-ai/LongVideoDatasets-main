#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""长视频预处理:PySceneDetect(镜头) + WhisperX(中英字幕) + PaddleOCR(屏幕文字)。

产出与标注 schema 的 structure 段对齐的 JSON,供 run_annotate.py 消费:
  {"video_id": "...", "structure": {"shots":[], "scenes":[], "events":[],
                                    "subtitles":[], "ocr":[]}}
scenes / events 故意留空 —— 由 annotator(LLM)或人工生成。

依赖(建议独立虚拟环境):
  pip install "scenedetect[opencv]" whisperx paddleocr paddlepaddle-gpu opencv-python
用法:
  python preprocess.py --video videos/doc_00001.mp4 --video-id doc_00001 \
      --lang en --out structure/doc_00001.json --hf-token $HF_TOKEN
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("preprocess")


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


def detect_shots(video_path: str, cfg: Config) -> list:
    from scenedetect import ContentDetector, detect
    log.info("镜头分割中 ...")
    scenes = detect(video_path, ContentDetector(threshold=cfg.scene_threshold))
    shots = []
    for i, (start, end) in enumerate(scenes, start=1):
        shots.append({
            "id": i,
            "start": round(start.get_seconds(), 3),
            "end": round(end.get_seconds(), 3),
            "desc": {"zh": None, "en": None},
            "keyframe_path": None,
        })
    log.info("检出 %d 个镜头", len(shots))
    return shots


def transcribe(video_path: str, lang: str, cfg: Config, hf_token) -> list:
    import whisperx
    log.info("WhisperX 转写中 (lang=%s) ...", lang)
    audio = whisperx.load_audio(video_path)
    model = whisperx.load_model(cfg.whisper_model, cfg.device,
                                compute_type=cfg.compute_type, language=lang)
    result = model.transcribe(audio, batch_size=cfg.whisper_batch)

    align_model, meta = whisperx.load_align_model(language_code=result["language"], device=cfg.device)
    result = whisperx.align(result["segments"], align_model, meta, audio, cfg.device)

    if hf_token:
        try:
            diarize = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=cfg.device)
            result = whisperx.assign_word_speakers(diarize(audio), result)
        except Exception as e:
            log.warning("说话人分离失败,跳过: %s", e)

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
    log.info("转写得到 %d 条字幕", len(subs))
    return subs


def run_ocr(video_path: str, cfg: Config) -> list:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang=cfg.ocr_lang, show_log=False)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps * cfg.ocr_interval_sec)))
    log.info("OCR 采帧: fps=%.2f, 每 %d 帧取一帧", fps, step)

    raw = []
    idx = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        t = round(idx / fps, 3)
        try:
            res = ocr.ocr(frame, cls=True)
        except Exception as e:
            log.warning("第 %ds OCR 失败: %s", t, e)
            idx += step
            continue
        lines = []
        for line in (res[0] or []) if res else []:
            txt, conf = line[1][0], line[1][1]
            if conf >= cfg.ocr_min_conf and txt.strip():
                lines.append(txt.strip())
        if lines:
            raw.append((t, " ".join(lines)))
        idx += step
    cap.release()

    spans = []
    for t, text in raw:
        if spans and spans[-1]["text"] == text and (t - spans[-1]["end"]) <= cfg.ocr_interval_sec * 1.5:
            spans[-1]["end"] = t
        else:
            spans.append({"start": t, "end": t, "text": text, "bbox": None})
    log.info("OCR 得到 %d 个文本区间", len(spans))
    return spans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--video-id", required=True)
    p.add_argument("--lang", choices=["zh", "en"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--skip-ocr", action="store_true")
    args = p.parse_args()

    cfg = Config()
    structure = {"shots": [], "scenes": [], "events": [], "subtitles": [], "ocr": []}

    for name, fn in (("镜头分割", lambda: detect_shots(args.video, cfg)),
                     ("ASR", lambda: transcribe(args.video, args.lang, cfg, args.hf_token))):
        try:
            key = "shots" if name == "镜头分割" else "subtitles"
            structure[key] = fn()
        except Exception as e:
            log.error("%s 失败: %s", name, e)

    if not args.skip_ocr:
        try:
            structure["ocr"] = run_ocr(args.video, cfg)
        except Exception as e:
            log.error("OCR 失败: %s", e)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"video_id": args.video_id, "structure": structure}, f,
                  ensure_ascii=False, indent=2)
    log.info("已写出: %s", args.out)


if __name__ == "__main__":
    main()
