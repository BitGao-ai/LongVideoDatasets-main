# -*- coding: utf-8 -*-
"""音频事件轨(Res 1.3)——输入音轨 waveform + ASR 词时间线。

规则管线:非语音区间(ASR 空白)> 0.5s 且能量突增 → 音频事件
{start, end, type, desc, lang, confidence}。

- waveform 用 ffmpeg 解码为 16kHz 单声道 float32(纯标准库 + numpy);
- type 判别:本地小模型可用时交给本地模型,否则规则判别
  (能量强度 + 与语音段能量比 + 突发性);
- 无音频事件时字段为 [] 而非缺省(对齐 Res 1.3)。
"""

import logging
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import clamp

log = logging.getLogger("annotator.audio")

# 允许的类型(与 schema 对齐)
AUDIO_TYPES = ("music", "sfx", "applause", "noise", "other")

_MIN_GAP_SEC = 0.5          # 非语音空白 >0.5s 才可能是音频事件
_MIN_EVENT_SEC = 0.8        # 音频事件最小长度
_FRAME_SEC = 0.05           # 能量分析帧长(50ms)


def _ffmpeg_pcm_cmd(video_path: str, sr: int) -> List[str]:
    return ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
            "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]


def rms_profile_streaming(video_path: str, sr: int = 16000,
                          chunk_frames: int = 4096) -> Optional[np.ndarray]:
    """流式解码音轨并逐块累计 RMS 包络,返回 50ms 一帧的 RMS 序列。

    修复 C3:此前 `subprocess.run(stdout=PIPE)` 把整条 PCM 一次性读进内存,
    2h @16kHz f32 = 460MB,`frames ** 2` 再复制一份,峰值约 0.9GB;8h 视频
    直接 MemoryError。现在常驻内存只有一个 chunk + 结果包络本身
    (2h 视频的包络仅 144k 个 float32 ≈ 0.6MB)。
    """
    if shutil.which("ffmpeg") is None:
        log.warning("未安装 ffmpeg,音频事件轨跳过(waveform 解码不可用)")
        return None

    n = max(1, int(sr * _FRAME_SEC))          # 每个分析帧的采样点数
    read_bytes = n * chunk_frames * 4         # float32
    out: List[np.ndarray] = []
    tail = b""
    proc = subprocess.Popen(_ffmpeg_pcm_cmd(video_path, sr),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(read_bytes)
            if not buf:
                break
            buf = tail + buf
            usable = (len(buf) // (n * 4)) * (n * 4)
            tail = buf[usable:]
            if usable == 0:
                continue
            block = np.frombuffer(buf[:usable], dtype=np.float32).reshape(-1, n)
            # einsum 逐行内积,不产生 block**2 的整块副本
            out.append(np.sqrt(np.einsum("ij,ij->i", block, block) / n).astype(np.float32))
        proc.stdout.close()
        stderr = proc.stderr.read()
        if proc.wait() != 0:
            log.warning("ffmpeg 解码音轨失败(可能无音轨): %s",
                        stderr.decode("utf-8", "ignore")[:200])
            return None
    finally:
        if proc.poll() is None:
            proc.kill()
        for stream in (proc.stdout, proc.stderr):
            try:
                stream.close()
            except (OSError, AttributeError):
                pass
    if not out:
        return None
    return np.concatenate(out)


def decode_waveform(video_path: str, sr: int = 16000) -> Optional[np.ndarray]:
    """[已弃用] 一次性把整轨读进内存。保留给外部调用方,新代码请用
    rms_profile_streaming();2h 视频用它会占约 0.5GB 常驻内存(C3)。"""
    if shutil.which("ffmpeg") is None:
        log.warning("未安装 ffmpeg,音频事件轨跳过(waveform 解码不可用)")
        return None
    proc = subprocess.run(_ffmpeg_pcm_cmd(video_path, sr),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        log.warning("ffmpeg 解码音轨失败(可能无音轨): %s",
                    proc.stderr.decode("utf-8", "ignore")[:200])
        return None
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _non_speech_gaps(subtitles: List[Dict], duration: float) -> List[Tuple[float, float]]:
    """ASR 空白区间:对 [0,duration] 取字幕的补集。"""
    if duration <= 0:
        return []
    speech: List[Tuple[float, float]] = []
    for s in subtitles or []:
        st, et = float(s.get("start", 0.0)), float(s.get("end", 0.0))
        if et > st:
            speech.append((st, et))
    speech.sort()
    merged: List[Tuple[float, float]] = []
    for st, et in speech:
        if merged and st <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], et))
        else:
            merged.append((st, et))
    gaps: List[Tuple[float, float]] = []
    cursor = 0.0
    for st, et in merged:
        if st - cursor >= _MIN_GAP_SEC:
            gaps.append((cursor, st))
        cursor = max(cursor, et)
    if duration - cursor >= _MIN_GAP_SEC:
        gaps.append((cursor, duration))
    return gaps


def _rms_profile(wave: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    """[已弃用] 全内存版 RMS 包络。新代码用 rms_profile_streaming()(C3)。"""
    n = int(sr * _FRAME_SEC)
    n = max(1, n)
    n_frames = len(wave) // n
    if n_frames == 0:
        return np.zeros(1), n
    frames = wave[:n_frames * n].reshape(n_frames, n)
    return np.sqrt(np.mean(frames ** 2, axis=1)), n


def _active_bounds(seg: np.ndarray, threshold: float,
                   pad_frames: int = 5) -> Optional[Tuple[int, int]]:
    """段内能量活跃区的帧下标 [lo, hi],各向外扩 pad_frames 帧。

    修复 B8:此前写作 `act[0] - max(0, act[0] - 5)`,化简即 `min(act[0], 5)`,
    恒 ≤5 帧(0.25s),于是所有音频事件的起点都被钉在 ASR 空白区的开头,
    与真正的能量突增时刻无关。
    """
    act = np.where(seg >= threshold)[0]
    if len(act) == 0:
        return None
    lo = max(0, int(act[0]) - pad_frames)
    hi = min(len(seg) - 1, int(act[-1]) + pad_frames)
    return lo, hi


def _speech_baseline(rms: np.ndarray, subtitles: List[Dict]) -> float:
    """语音段的 RMS 中位数。

    此前用全轨中位数并命名为 speech_median:长视频里非语音占多数时基线偏低,
    `seg_rms < speech_median*2` 的门限会把大量环境噪声收成"音频事件"。
    """
    idx: List[np.ndarray] = []
    for s in subtitles or []:
        try:
            st, et = float(s["start"]), float(s["end"])
        except (KeyError, TypeError, ValueError):
            continue
        f0, f1 = int(st / _FRAME_SEC), min(len(rms), int(et / _FRAME_SEC))
        if f1 > f0:
            idx.append(np.arange(f0, f1))
    if idx:
        picked = rms[np.concatenate(idx)]
        if picked.size:
            return float(np.median(picked)) or 1e-6
    log.info("无可用语音段,能量基线回退为全轨中位数")
    return float(np.median(rms)) or 1e-6


def _classify_rule(rms: float, speech_median: float, rms_ratio: float,
                   burstiness: float) -> Dict[str, object]:
    """规则判别(本地模型不可用时):能量极高→音乐/音效,突发→掌声,否则噪声。"""
    if rms_ratio >= 6.0 or rms >= 0.5:
        t, desc = "music", "高能量音频(疑似音乐或音效)"
    elif burstiness >= 3.0 and rms_ratio >= 2.0:
        t, desc = "applause", "突发密集能量(疑似掌声/鼓掌声)"
    elif speech_median > 0 and rms_ratio >= 2.0:
        t, desc = "sfx", "明显音效(疑似环境音效)"
    else:
        t, desc = "noise", "低能量噪声或环境声"
    return {"type": t, "desc": desc}


def detect_audio_events(video_path: str, subtitles: List[Dict], duration: float,
                        local_client=None, confidence_base: float = 0.6,
                        context_window: float = 30.0) -> List[Dict]:
    """检测音频事件(Res 1.3)。

    规则:ASR 空白 >0.5s 的区间,取能量突增部分 → 音频事件;
    无音频事件时返回 []。type 判别优先本地小模型,失败回退规则。
    """
    rms = rms_profile_streaming(video_path)
    if rms is None or len(rms) < 4:
        log.info("无音轨可分析,音频事件轨为空")
        return []

    speech_median = _speech_baseline(rms, subtitles)
    gaps = _non_speech_gaps(subtitles, duration)
    if not gaps:
        return []

    events: List[Dict] = []
    subs = subtitles or []
    for gs, ge in gaps:
        if ge - gs < _MIN_GAP_SEC:
            continue
        f0, f1 = int(gs / _FRAME_SEC), min(len(rms), int(ge / _FRAME_SEC))
        seg = rms[f0:f1]
        if len(seg) == 0:
            continue
        seg_rms = float(np.max(seg))          # 段内峰值能量
        seg_mean = float(np.mean(seg))        # 段内平均能量
        peak_ratio = seg_mean / speech_median if speech_median else 0.0
        # 突发性:前 50% 与后 50% 的能量比(掌声/爆发式)
        half = max(1, len(seg) // 2)
        burst = (float(np.mean(seg[half:])) / (float(np.mean(seg[:half])) or 1e-6))
        # 能量突增判定:峰值高于语音中位数 2 倍 或 段内能量显著
        if seg_rms < speech_median * 2.0:
            continue

        # 收缩到能量明显段(避免把大片静音算进事件)
        bounds = _active_bounds(seg, threshold=max(speech_median * 1.2, 1e-5),
                                pad_frames=5)
        if bounds is None:
            continue
        es = clamp(gs + bounds[0] * _FRAME_SEC, gs, ge)
        ee = clamp(gs + bounds[1] * _FRAME_SEC, gs, ge)
        if ee - es < _MIN_EVENT_SEC:
            continue

        # 前后文 ASR(±context_window)供本地模型判别
        ctx = " ".join(s.get("text", "") for s in subs
                       if s.get("start", 0) <= ee + context_window
                       and s.get("end", 0) >= es - context_window)

        cls: Dict[str, object] = {}
        if local_client is not None and getattr(local_client, "enabled", False):
            cls = local_client.classify_audio(seg_mean, peak_ratio, ee - es, ctx)
        if not cls.get("type"):
            cls = _classify_rule(seg_mean, speech_median, peak_ratio, burst)
        etype = cls.get("type") if cls.get("type") in AUDIO_TYPES else "other"

        conf = confidence_base
        if etype in ("music", "applause"):
            conf = min(0.95, confidence_base + 0.2)
        events.append({
            "start": round(es, 3),
            "end": round(ee, 3),
            "type": etype,
            "desc": {"zh": cls.get("desc") or "音频事件", "en": cls.get("desc") or "audio event"},
            "lang": None,
            "confidence": round(conf, 3),
        })

    events.sort(key=lambda e: e["start"])
    log.info("音频事件轨:检出 %d 个(non-speech 候选 %d)", len(events), len(gaps))
    return events
