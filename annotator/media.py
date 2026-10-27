# -*- coding: utf-8 -*-
"""媒体工具:抽帧(帧索引化,Res 1.1)+ base64 编码、字幕/OCR 按时间窗切片、镜头合并成窗口。

注意:本模块不再使用 cv2 CAP_PROP_POS_MSEC 直接 seek(REVIEW D9/D3-6)。
所有“抽帧/单帧”都优先走 frames_store.FrameStore(ffmpeg 1fps 帧索引);
无帧索引时回退 cv2 顺序解码(自维护时间轴),由 FrameStore.build 的兜底路径提供。
"""

import base64
import logging
from typing import Dict, List, Optional, Tuple

import cv2

from .frames_store import FrameStore

log = logging.getLogger("annotator.media")


def _encode_frame(frame, max_edge: int) -> Optional[str]:
    h, w = frame.shape[:2]
    scale = min(1.0, max_edge / float(max(h, w)))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("ascii") if ok else None


def sample_frames(video_path: str, start: float, end: float, k: int,
                  max_edge: int = 768, store: Optional[FrameStore] = None
                  ) -> List[Tuple[float, str]]:
    """在 [start, end] 内均匀采样 k 帧(去掉端点),返回 [(秒, base64_jpeg)]。

    有帧索引(store)时从帧索引取(精确、无二次 seek);否则顺序解码兜底。
    """
    if k <= 0 or end <= start:
        return []
    if store is not None:
        times = store.sample_times(start, end, k)
        return store.frames_b64(times, max_edge)

    # 兜底:cv2 顺序解码(自维护时间轴,误差 <1 帧)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("无法打开视频: %s", video_path)
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if total and fps:
        i0, i1 = int(start * fps), int(end * fps)
        step = max(1, int((i1 - i0) / (k + 1)))
    else:
        step, i0, i1 = 1, 0, 0
    out: List[Tuple[float, str]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i0 <= idx <= i1 and (idx - i0) % step == 0 and (idx - i0) > 0:
            b64 = _encode_frame(frame, max_edge)
            if b64:
                out.append((round(idx / fps, 3), b64))
            if len(out) >= k:
                break
        idx += 1
    cap.release()
    return out


def one_frame(video_path: str, t: float, max_edge: int = 768,
              store: Optional[FrameStore] = None) -> Optional[str]:
    """取 t 时刻 1 帧(有帧索引则取 ≤t 最近帧,否则顺序解码兜底)。"""
    if store is not None:
        return store.frame_b64(t, max_edge)
    return _one_frame_sequential(video_path, t, max_edge)


def _one_frame_sequential(video_path: str, t: float, max_edge: int) -> Optional[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    target = int(t * fps)
    frame = None
    idx = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx == target:
            frame = f
            break
        idx += 1
    cap.release()
    return _encode_frame(frame, max_edge) if frame is not None else None


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

    单镜头超过 window_sec 时按固定步长硬切(A2-2 修复):lecture / 口播 / 会议
    等单机位长镜头在旧实现下只产出 1 个窗口——2h 视频只标 1 窗,窗口级注释
    完全失去时间定位。硬切后的尾段(可能 < window_sec)保持软边界,继续按
    原有贪心语义与后续镜头合并,窗口仍按时间升序且互不跨界。

    若没有镜头信息(shots 为空),调用方应回退到固定步长切窗。
    """
    if not shots:
        return []
    shots = sorted(shots, key=lambda s: s["start"])
    # 展平成原子段:(start, end, hard_prev)。
    # hard_prev=True 表示与前一段是【同一超长镜头内部】的硬切边界,不得并窗
    # ——否则切开的镜头又会被贪心合并重新拼成长窗。首片硬切前置边界为软
    #(可与上一镜头常规合并,实际受窗口长度约束同样拼不进去)。
    atoms: List[Tuple[float, float, bool]] = []
    for sh in shots:
        s, e = float(sh["start"]), float(sh["end"])
        if e - s <= window_sec:
            atoms.append((s, e, False))
            continue
        t = s
        while t < e:
            atoms.append((t, min(t + window_sec, e), t > s))
            t += window_sec
    # 与旧实现完全一致的贪心合并:能装进一个窗口就延续,否则开新窗
    windows: List[Tuple[float, float]] = []
    cur_s, cur_e = atoms[0][0], atoms[0][1]
    for s, e, hard_prev in atoms[1:]:
        if (not hard_prev) and e - cur_s <= window_sec:
            cur_e = e
        else:
            windows.append((round(cur_s, 3), round(cur_e, 3)))
            cur_s, cur_e = s, e
    windows.append((round(cur_s, 3), round(cur_e, 3)))
    return windows


def fixed_windows(duration_sec: float, window_sec: float) -> List[Tuple[float, float]]:
    """无镜头信息时的回退:固定步长切窗。"""
    out, t = [], 0.0
    while t < duration_sec:
        out.append((round(t, 3), round(min(t + window_sec, duration_sec), 3)))
        t += window_sec
    return out
