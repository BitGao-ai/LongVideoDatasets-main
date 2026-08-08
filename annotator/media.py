# -*- coding: utf-8 -*-
"""媒体工具:抽帧 + base64 编码、字幕/OCR 按时间窗切片、镜头合并成窗口。"""

import base64
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("annotator.media")


def _encode_frame(frame, max_edge: int) -> Optional[str]:
    h, w = frame.shape[:2]
    scale = min(1.0, max_edge / float(max(h, w)))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("ascii") if ok else None


def sample_frames(video_path: str, start: float, end: float, k: int,
                  max_edge: int = 768) -> List[Tuple[float, str]]:
    """在 [start, end] 内均匀采样 k 帧(去掉端点),返回 [(秒, base64_jpeg)]。"""
    if k <= 0 or end <= start:
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("无法打开视频: %s", video_path)
        return []
    times = np.linspace(start, end, k + 2)[1:-1]
    out: List[Tuple[float, str]] = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if ok:
            b64 = _encode_frame(frame, max_edge)
            if b64:
                out.append((round(float(t), 3), b64))
    cap.release()
    return out


def one_frame(video_path: str, t: float, max_edge: int = 768) -> Optional[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    return _encode_frame(frame, max_edge) if ok else None


# --------- 字幕 / OCR 按窗切片 ---------
def _in_span(item: Dict, start: float, end: float) -> bool:
    return item["end"] >= start and item["start"] <= end


def subs_in_span(subtitles: List[Dict], start: float, end: float) -> List[Dict]:
    return [s for s in (subtitles or []) if _in_span(s, start, end)]


def subs_text(subtitles: List[Dict], start: float, end: float) -> str:
    lines = []
    for s in subs_in_span(subtitles, start, end):
        spk = f"({s['speaker']}) " if s.get("speaker") else ""
        lines.append(f"[{s['start']:.0f}s] {spk}{s['text']}")
    return "\n".join(lines)


def ocr_text(ocr: List[Dict], start: float, end: float) -> str:
    hits = [o for o in (ocr or []) if _in_span(o, start, end)]
    return " | ".join(o["text"] for o in hits)


# --------- 镜头 -> 窗口 ---------
def build_windows(shots: List[Dict], window_sec: float) -> List[Tuple[float, float]]:
    """把连续镜头合并成不超过 window_sec 的窗口,对齐镜头边界。

    若没有镜头信息(shots 为空),调用方应回退到固定步长切窗。
    """
    if not shots:
        return []
    shots = sorted(shots, key=lambda s: s["start"])
    windows: List[Tuple[float, float]] = []
    cur_s, cur_e = shots[0]["start"], shots[0]["end"]
    for sh in shots[1:]:
        if sh["end"] - cur_s <= window_sec:
            cur_e = sh["end"]
        else:
            windows.append((round(cur_s, 3), round(cur_e, 3)))
            cur_s, cur_e = sh["start"], sh["end"]
    windows.append((round(cur_s, 3), round(cur_e, 3)))
    return windows


def fixed_windows(duration_sec: float, window_sec: float) -> List[Tuple[float, float]]:
    """无镜头信息时的回退:固定步长切窗。"""
    out, t = [], 0.0
    while t < duration_sec:
        out.append((round(t, 3), round(min(t + window_sec, duration_sec), 3)))
        t += window_sec
    return out
