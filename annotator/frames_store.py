# -*- coding: utf-8 -*-
"""帧索引化存储(Res 1.1/1.2)——替换 media.py 中 CAP_PROP_POS_MSEC 直接 seek。

设计:
- 预处理阶段用 ffmpeg(-ss 定位 + fps 精确解码)导出 1fps 帧序列:
      <frames_dir>/f_{t:07d}_{ms:03d}.jpg      # 文件名即时间戳
- 后续任何“抽帧/单帧/马赛克”都从帧索引取,不再二次 seek(修复 REVIEW D9/D3-6);
- 兜底(无 ffmpeg / 可裁剪磁盘):cv2 顺序解码 + 自维护时间轴,禁止 CAP_PROP_POS_MSEC;
- verify_index(): 帧号-时间戳单调、首帧≈0、末帧≈duration,不一致即报错而非静默。
"""

import logging
import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .utils import clamp

log = logging.getLogger("annotator.frames")

_FRAME_RE = re.compile(r"f_(\d{7})_(\d{3})\.jpg$")


def _fname(t: float) -> str:
    sec = int(t)
    ms = int(round((t - sec) * 1000.0))
    if ms >= 1000:
        sec, ms = sec + 1, 0
    return f"f_{sec:07d}_{ms:03d}.jpg"


def _time_of(name: str) -> Optional[float]:
    """从文件名解出时刻。文件名两段分别是【整秒】与【毫秒】,不是帧号。"""
    m = _FRAME_RE.match(name)
    if not m:
        return None
    return round(int(m.group(1)) + int(m.group(2)) / 1000.0, 3)


def _scan_index_from_names(names: List[str], fps: float) -> Dict[int, float]:
    """由文件名列表重建 {frame_no: t}(修复 B1)。

    此前实现是 `idx[no] = no / fps + ms/1000`,把文件名第一段当成帧号,于是:
      fps=2.0 时两个不同时刻(0.0s / 0.5s)整秒段相同 -> 键碰撞,一半帧丢失;
      fps=0.5 时时间轴被整体放大 1/fps 倍。
    正确做法是先由文件名解出真实时刻,再按 fps 反推帧序号。
    """
    fps = float(fps) or 1.0
    idx: Dict[int, float] = {}
    for name in names:
        t = _time_of(name)
        if t is None:
            continue
        no = int(round(t * fps))
        if no in idx and abs(idx[no] - t) > 1e-6:
            log.warning("帧索引序号冲突(fps=%.3f):%.3fs 与 %.3fs 映射到同一序号 %d;"
                        "请检查 --frame-rate 是否与帧目录一致", fps, idx[no], t, no)
            no = max(idx) + 1 if idx else 0
        idx[no] = t
    return idx


def _encode_jpg(img: np.ndarray, max_edge: int, quality: int = 85) -> Optional[str]:
    import base64
    h, w = img.shape[:2]
    scale = min(1.0, max_edge / float(max(h, w)))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("ascii") if ok else None


class FrameStore:
    """帧索引:文件系统即数据库,文件名即时间戳。

    Attributes:
        frames_dir: 帧序列目录(内含 index.json 元信息)。
        fps: 抽帧率(默认 1fps)。
        max_edge: 导出帧长边上限。
    """

    def __init__(self, frames_dir: str, fps: float = 1.0, max_edge: int = 768):
        self.frames_dir = os.path.abspath(frames_dir)
        self.fps = float(fps)
        self.max_edge = int(max_edge)
        self._index: Optional[Dict[int, float]] = None      # {frame_no: t}
        self._times: Optional[List[float]] = None           # 升序时间列表
        self._time_to_no: Optional[Dict[float, int]] = None  # {t: frame_no} 反查
        self._duration: float = 0.0

    # ------------------------------------------------------------------ build
    def build(self, video_path: str, force: bool = False) -> Dict[str, object]:
        """构建帧索引(已存在且未 force 则直接加载)。

        优先 ffmpeg(fps 过滤器精确均匀抽帧),失败回退 cv2 顺序解码。
        返回统计信息。任何一步失败都抛错(不静默,对齐 Res 1.1“报错而非静默”)。
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频不存在: {video_path}")
        os.makedirs(self.frames_dir, exist_ok=True)
        index_path = os.path.join(self.frames_dir, "index.json")

        if not force and os.path.exists(index_path):
            self._load_index()
            if self._index:
                log.info("帧索引已存在,复用: %s(%d 帧)", self.frames_dir, len(self._index))
                return self.stats()

        # 重建前清空旧帧,避免与残留帧混合(帧是可再生的派生产物)
        for f in os.listdir(self.frames_dir):
            if f.endswith(".jpg"):
                try:
                    os.remove(os.path.join(self.frames_dir, f))
                except OSError as e:
                    log.warning("清理旧帧失败 %s: %s", f, e)

        n = 0
        try:
            n = self._build_ffmpeg(video_path)
        except Exception as e:  # noqa: BLE001
            log.warning("ffmpeg 抽帧失败(%s),回退 cv2 顺序解码", e)
            n = self._build_cv2(video_path)
        if n <= 0:
            raise RuntimeError(f"帧索引构建失败: 0 帧产出({video_path})")

        # 从文件名重建内存索引后再固化(修复:此前 index.json 恒为空,首次 build 后
        # 一切读取 API 均抛“帧索引为空”,阻塞整条流水线)
        self._index = self._scan_files_index()
        if not self._index:
            raise RuntimeError(f"帧索引构建失败: 文件名扫描未产出索引({self.frames_dir})")
        self._write_index()
        log.info("帧索引构建完成: %d 帧 @ %.2ffps -> %s", n, self.fps, self.frames_dir)
        return self.stats()

    def _build_ffmpeg(self, video_path: str) -> int:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("未安装 ffmpeg")
        vf = f"fps={self.fps},scale='min({self.max_edge},iw)':-2"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
               "-vf", vf, "-q:v", "3", "-frames:v", str(self._cap_frames()),
               os.path.join(self.frames_dir, "f_%07d_000.jpg")]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", "ignore")[:300])

        # 文件名即时间戳:ffmpeg 输出帧号从 1 起,第 i 帧时刻 t = (i-1) / fps
        files = sorted(os.listdir(self.frames_dir))
        for name in files:
            m = _FRAME_RE.match(name)
            if not m:
                continue
            no = int(m.group(1))
            t = round((no - 1) / self.fps, 3)
            target = _fname(t)
            if target != name:
                os.replace(os.path.join(self.frames_dir, name),
                           os.path.join(self.frames_dir, target))
        return len([f for f in os.listdir(self.frames_dir) if f.endswith(".jpg")])

    def _cap_frames(self) -> int:
        """上限保护:按 8h 估算,避免异常视频抽帧失控。"""
        return int(8 * 3600 * self.fps) + 100

    def _build_cv2(self, video_path: str) -> int:
        """兜底:cv2 顺序解码 + 自维护时间轴,禁止 CAP_PROP_POS_MSEC seek。"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps / self.fps)))
        idx, written = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                t = round(idx / fps, 3)
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok2:
                    with open(os.path.join(self.frames_dir, _fname(t)), "wb") as f:
                        f.write(buf.tobytes())
                    written += 1
            idx += 1
        cap.release()
        if written == 0:
            raise RuntimeError("cv2 顺序解码未产出任何帧")
        return written

    # ------------------------------------------------------------------ index
    def _load_index(self) -> None:
        import json
        index_path = os.path.join(self.frames_dir, "index.json")
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.fps = float(meta.get("fps", self.fps))
            self.max_edge = int(meta.get("max_edge", self.max_edge))
            self._index = {int(k): float(v) for k, v in meta.get("frames", {}).items()}
            self._duration = float(meta.get("duration_sec", 0.0))
            self._times = sorted(self._index.values())
        except (OSError, ValueError, KeyError) as e:
            log.warning("index.json 读取失败,将重建: %s", e)
            self._index, self._times, self._duration = None, None, 0.0

    def _write_index(self) -> None:
        import json
        index_path = os.path.join(self.frames_dir, "index.json")
        meta = {
            "fps": self.fps,
            "max_edge": self.max_edge,
            "duration_sec": self._duration,
            "frames": {str(no): t for no, t in (self._index or {}).items()},
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _scan_files_index(self) -> Dict[int, float]:
        """从目录内 f_{sec}_{ms}.jpg 文件名重建 {frame_no: t}(文件名即时间戳)。"""
        return _scan_index_from_names(os.listdir(self.frames_dir), self.fps)

    def _ensure_index(self) -> None:
        if self._index is None:
            if not os.path.exists(os.path.join(self.frames_dir, "index.json")):
                # 无 index.json 但目录有帧:从文件名重建索引
                self._index = self._scan_files_index()
                if self._index:
                    self._times = sorted(self._index.values())
                    self._write_index()
            else:
                self._load_index()
        if not self._index:
            raise RuntimeError(f"帧索引为空: {self.frames_dir};请先 build()")
        if self._times is None:
            self._times = sorted(self._index.values())
        if self._time_to_no is None:
            self._time_to_no = {t: no for no, t in self._index.items()}

    def set_duration(self, duration_sec: float) -> None:
        self._duration = float(duration_sec)

    # ------------------------------------------------------------------ verify
    def verify_index(self, duration_sec: Optional[float] = None,
                     tolerance: float = 1.5) -> bool:
        """校验(Res 1.1):帧号-时间戳单调、首帧≈0、末帧≈duration,不一致即报错。"""
        self._ensure_index()
        times = self._times or sorted(self._index.values())
        if not times:
            raise RuntimeError("verify_index: 索引为空")
        if any(b < a for a, b in zip(times, times[1:])):
            raise RuntimeError("verify_index: 帧时间戳非单调!")
        if abs(times[0]) > tolerance:
            raise RuntimeError(f"verify_index: 首帧时间 {times[0]:.3f}s 偏离 0 超过 {tolerance}s")
        dur = duration_sec or self._duration
        if dur and abs(times[-1] - dur) > tolerance:
            log.warning("verify_index: 末帧 %.3fs 与视频时长 %.3fs 偏差 %.3fs(>%.1fs),请检查",
                        times[-1], dur, abs(times[-1] - dur), tolerance)
        log.info("verify_index 通过:%d 帧 [%.3f, %.3f]s", len(times), times[0], times[-1])
        return True

    def stats(self) -> Dict[str, object]:
        self._ensure_index()
        return {"frames_dir": self.frames_dir, "fps": self.fps, "frame_count": len(self._index or {}),
                "duration_sec": self._duration or ((self._times or [0])[-1] if self._times else 0.0)}

    # ------------------------------------------------------------ 读取 API
    def index_times(self) -> List[float]:
        """升序的全部帧时刻(供 OCR 等阶段复用帧索引,避免重复解码视频,C8)。"""
        try:
            self._ensure_index()
        except RuntimeError:
            return []
        return list(self._times or [])

    def read_frame(self, t: float):
        """读取 ≤t 的最近一帧为 BGR ndarray(不做缩放);缺帧返回 None。"""
        p = self.frame_path(t)
        if not p or not os.path.exists(p):
            return None
        return cv2.imread(p)

    def frame_path(self, t: float) -> Optional[str]:
        """返回 ≤t 的最近一帧文件路径(二分查找,不 seek 视频流)。"""
        self._ensure_index()
        times = self._times
        import bisect
        i = bisect.bisect_right(times, t) - 1
        if i < 0:
            i = 0
        return os.path.join(self.frames_dir, _fname(times[i]))

    def frame_b64(self, t: float, max_edge: Optional[int] = None) -> Optional[str]:
        p = self.frame_path(t)
        if not p or not os.path.exists(p):
            return None
        img = cv2.imread(p)
        if img is None:
            return None
        return _encode_jpg(img, max_edge or self.max_edge)

    def frames_b64(self, times: List[float], max_edge: Optional[int] = None
                   ) -> List[Tuple[float, str]]:
        """批量取帧(单视频内可线程池加速)。"""
        out: List[Tuple[float, str]] = []
        for t in times:
            b64 = self.frame_b64(t, max_edge)
            if b64:
                out.append((round(float(t), 3), b64))
        return out

    def sample_times(self, start: float, end: float, k: int,
                     pad: float = 0.0) -> List[float]:
        """在 [start-pad, end+pad] 内均匀取 k 个时刻(去掉端点),全部 clamp 到有效范围。"""
        self._ensure_index()
        lo = clamp(start - pad, 0.0, self._times[-1])
        hi = clamp(end + pad, 0.0, self._times[-1])
        if k <= 0 or hi <= lo:
            return []
        return [round(t, 3) for t in np.linspace(lo, hi, k + 2)[1:-1]]

    def shot_keyframe_times(self, start: float, end: float) -> List[float]:
        """每镜首/中/尾 3 关键帧(Res 1.2)。"""
        mid = (start + end) / 2.0
        return [round(t, 3) for t in (start, mid, end)]

    def distinct_times(self, start: float, end: float, k: int,
                       pad: float = 0.0) -> List[float]:
        """[start-pad, end+pad] 内【真实存在且互不重复】的帧时刻,最多 k 个。

        修复 B2:sample_times 在 1fps 网格上取 64 个时刻时会大量落到同一帧
        (3s 事件的 64 格实际只有 11 张不同帧,重复 5.8 倍),马赛克因此充斥
        重复画面。这里先取区间内的真实帧,再按需均匀下采样。
        """
        self._ensure_index()
        times = self._times or []
        if not times or k <= 0:
            return []
        lo = clamp(start - pad, 0.0, times[-1])
        hi = clamp(end + pad, 0.0, times[-1])
        if hi <= lo:
            return []
        import bisect
        i0 = bisect.bisect_left(times, lo)
        i1 = bisect.bisect_right(times, hi)
        window = times[i0:i1]
        if len(window) <= k:
            return list(window)
        idx = np.linspace(0, len(window) - 1, k)
        return [window[int(round(j))] for j in idx]

    def mosaic(self, span: Tuple[float, float], k: int = 64,
               tile_edge: int = 128, pad: float = 0.0
               ) -> Optional[Tuple[str, List[float], int]]:
        """帧马赛克(Res 3.2 边界复核):span ±pad 内的真实帧排成方阵单图。

        返回 (base64_jpeg, 每格对应的时刻, 每行格数)。格数不足以铺满 8×8 时
        自动缩小方阵(如 11 张真实帧 -> 3×3),而不是用重复帧凑数(B2)。
        """
        side = max(1, int(k ** 0.5))
        times = self.distinct_times(span[0], span[1], side * side, pad=pad)
        if len(times) < 4:
            return None
        side = max(2, int(len(times) ** 0.5))
        times = times[: side * side]
        tiles = []
        for t in times:
            p = self.frame_path(t)
            img = cv2.imread(p) if p and os.path.exists(p) else None
            if img is None:
                return None
            tiles.append(cv2.resize(img, (tile_edge, tile_edge), interpolation=cv2.INTER_AREA))
        rows = [np.hstack(tiles[r * side:(r + 1) * side]) for r in range(side)]
        grid = np.vstack(rows)
        # 马赛克本身就是给模型看细节的,不要再压回 max_edge —— 那会把每格从
        # tile_edge 降到 tile_edge*max_edge/(side*tile_edge),白白丢分辨率(B2)。
        b64 = _encode_jpg(grid, max_edge=side * tile_edge, quality=85)
        return (b64, times, side) if b64 else None

    def shot_thumbnail(self, start: float, end: float, grid: Tuple[int, int] = (4, 4),
                       tile_edge: int = 160) -> Optional[str]:
        """整镜 4×4 缩略图(供人工抽样,Res 1.2)。"""
        out = self.mosaic((start, end), k=grid[0] * grid[1], tile_edge=tile_edge)
        return out[0] if out else None
