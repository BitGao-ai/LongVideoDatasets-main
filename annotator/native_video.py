# -*- coding: utf-8 -*-
"""原生视频输入(native_video 模式)——仅 Qwen 可用。

思路:30min–2h 无法一次喂整段,故按窗口用 ffmpeg 切出短片段,再用 DashScope
原生 SDK(MultiModalConversation)把【真实视频片段】直接送 Qwen-VL(而非抽帧)。

依赖:
  - 系统安装 ffmpeg
  - pip install dashscope
Kimi(Moonshot)无原生视频输入,native_video 模式不支持,请用 frames 模式。
"""

import logging
import os
import shutil
import subprocess
from typing import Optional

log = logging.getLogger("annotator.native")


def ensure_available() -> None:
    """检查 ffmpeg 与 dashscope 是否就绪,缺失则给出明确报错。"""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("native_video 模式需要 ffmpeg 切分视频;请先安装 ffmpeg")
    try:
        import dashscope  # noqa: F401
    except ImportError as e:
        raise RuntimeError("native_video 模式需要 dashscope SDK;请 pip install dashscope") from e


def cut_clip(video_path: str, start: float, end: float, out_path: str,
             max_edge: int = 720) -> str:
    """用 ffmpeg 切出 [start, end] 片段,下采样、去音轨以缩小体积。"""
    dur = max(0.1, float(end) - float(start))
    vf = f"scale='min({max_edge},iw)':-2"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{float(start):.3f}", "-i", video_path, "-t", f"{dur:.3f}",
        "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-movflags", "+faststart", out_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"ffmpeg 切片失败: {proc.stderr.decode('utf-8', 'ignore')[:300]}")
    return out_path


def describe_clip(clip_path: str, system_text: str, user_text: str,
                  model: str, api_key: str, fps: float = 2.0) -> str:
    """把视频片段直接送 Qwen-VL,返回模型文本(应为 JSON 字符串,由调用方解析)。"""
    import dashscope
    from dashscope import MultiModalConversation

    abspath = os.path.abspath(clip_path)
    messages = [
        {"role": "system", "content": [{"text": system_text}]},
        {"role": "user", "content": [{"video": f"file://{abspath}", "fps": fps},
                                     {"text": user_text}]},
    ]
    resp = MultiModalConversation.call(model=model, messages=messages, api_key=api_key)
    if resp is None or getattr(resp, "status_code", 200) != 200:
        code = getattr(resp, "code", None)
        msg = getattr(resp, "message", str(resp))
        raise RuntimeError(f"DashScope 视频调用失败: {code} {msg}")

    content = resp.output.choices[0].message.content
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content if isinstance(c, dict))
    return content or ""
