#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""长视频 LLM 自动标注 CLI(v2,Res 全案)。

输入输出(视频 ID = 文件名去扩展名):
  输入:structure/<id>.json(preprocess.py 产出,经 --in 指定目录)
        meta/<id>.json(经 --meta-dir,缺失时自动生成)
  输出:annotations/<id>.json(经 --out-dir 指定,默认 annotations/)

用法:
  # API 模式(默认,云端大批量处理)
  export DASHSCOPE_API_KEY=sk-xxx        # provider=qwen
  # 或 export MOONSHOT_API_KEY=sk-xxx    # provider=kimi
  # 答案核验用另一家:export MOONSHOT_API_KEY=sk-xxx(配 --verify-provider kimi)

  # 批量模式:处理 structure/ 下所有预处理结果(meta 缺失时自动生成)
  python run_annotate.py --in structure/ --provider qwen \
      --meta-dir meta --out-dir annotations --videos-dir videos
  # 标注产物:annotations/<id>.json;缓存与阶段状态:annotations/.cache/<id>/、
  # structure/<id>.manifest.json;LLM 调用审计:reports/llm_trace_<id>.jsonl

  # 单视频模式(与旧版一致)
  python run_annotate.py \
      --meta       examples/meta_template.json \
      --structure  structure/doc_00001.json \
      --out        annotations/doc_00001.json \
      --provider   qwen

v2 变化:
- 详述默认必产(--no-describe 关闭);
- 事件引擎:挖掘→8×8 边界复核→去重→关键性分级;
- 锚定出题 + 能力矩阵 + 双模型答案核验(--verify-provider);
- 五通道抗捷径(含单帧剔除修复,--keep-single-frame 可保留);
- 断点续跑:manifest + 窗口缓存,中断重跑不重复烧钱。
"""

import argparse
import datetime
import glob
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from annotator.config import RunConfig
from annotator.pipeline import annotate_video
from annotator.utils import duration_bucket, quiet_http_loggers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
quiet_http_loggers()
log = logging.getLogger("run_annotate")

VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".ts")


def _collect(inp: str) -> list:
    files = sorted(glob.glob(os.path.join(inp, "*.json"))) if os.path.isdir(inp) \
        else sorted(glob.glob(inp))
    return [f for f in files if not f.endswith(".manifest.json")]


def _find_video(video_id: str, videos_dir: str) -> str:
    for ext in VIDEO_EXTS:
        p = os.path.join(videos_dir, f"{video_id}{ext}")
        if os.path.exists(p):
            return p
    return ""


def _probe_video(video: str):
    """读取 (duration, fps, width, height);失败返回 0。"""
    import cv2
    duration = fps = 0.0
    width = height = 0
    cap = cv2.VideoCapture(video)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps > 0 and n > 0:
            duration = n / fps
        cap.release()
    return duration, fps, width, height


def build_meta_dict(video_id: str, video_path: str, duration: float, lang: str,
                    fps: float = 0.0, width: int = 0, height: int = 0) -> dict:
    """按 schema 生成 meta 骨架(修复 A6)。

    此前写入 genre="unknown" / duration_bucket="unknown"(两个 enum 都没有该项)、
    fps=None(schema 要求 number 且必填)、resolution=None(要求 string),
    自动生成的每一份 meta 都是 schema 非法的,而校验只 warning 不阻断。
    """
    return {
        "video_id": video_id,
        "meta": {
            # genre 是受控枚举,自动生成时留占位值 "unknown"(schema 已显式允许),
            # 人工必须改成真实题材后才可发布。
            "genre": "unknown",
            "title": {"zh": None, "en": None},
            "duration_sec": round(float(duration or 0.0), 3),
            "duration_bucket": duration_bucket(duration or 0.0),
            "source": "local",
            "license": "UNSPECIFIED",       # 必须人工确认后再发布
            "url": None,
            "release_date": datetime.date.today().isoformat(),
            "primary_lang": lang if lang in ("zh", "en") else "zh",
            "has_subtitle": False,
            "subtitle_langs": [lang] if lang in ("zh", "en") else [],
        },
        "media": {
            "video_path": video_path,
            "audio_path": None,
            "fps": round(float(fps or 0.0), 3),
            "resolution": f"{width}x{height}" if width and height else "unknown",
        },
    }


def _build_meta(video_id: str, meta_dir: str, videos_dir: str,
                structure_path: str, lang: str) -> str:
    """meta 缺失时按项目 schema 模板自动生成(已存在则复用)。"""
    meta_path = os.path.join(meta_dir, f"{video_id}.json")
    if os.path.exists(meta_path):
        return meta_path
    video = _find_video(video_id, videos_dir)
    if not video:
        raise RuntimeError(
            f"meta 与视频均缺失,无法自动生成 meta: {video_id}(videos_dir={videos_dir})")
    duration, fps, width, height = _probe_video(video)
    try:
        with open(structure_path, encoding="utf-8") as f:
            subs = json.load(f).get("structure", {}).get("subtitles", [])
        if subs and subs[0].get("lang"):
            lang = subs[0]["lang"]
    except Exception:  # noqa: BLE001 —— 语言识别失败不阻断
        pass
    meta = build_meta_dict(video_id, video, duration, lang, fps, width, height)
    os.makedirs(meta_dir, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.warning("已自动生成 meta,发布前必须人工补全 genre / title / license: %s", meta_path)
    return meta_path


def main():
    p = argparse.ArgumentParser(description="长视频 LLM 自动标注 v2(Qwen / Kimi)")
    # 批量模式
    p.add_argument("--in", dest="inp", default="",
                   help="批量模式:structure 目录或 glob(preprocess.py 产出),逐个标注")
    p.add_argument("--meta-dir", default="meta", help="批量模式:meta 目录(缺失自动生成,默认 meta)")
    p.add_argument("--out-dir", default="annotations",
                   help="批量模式:标注结果输出目录,每视频写 <id>.json(默认 annotations)")
    p.add_argument("--videos-dir", default="videos", help="批量模式:自动生成 meta 时按该目录找视频")
    p.add_argument("--lang", default="zh", choices=["zh", "en"],
                   help="批量模式:自动生成 meta 时的语言(优先取 structure 字幕语言)")
    # 单视频模式(与旧版一致)
    p.add_argument("--meta", default="", help="单视频模式:视频元信息 JSON(video_id/meta/media)")
    p.add_argument("--structure", default="", help="单视频模式:预处理结构 JSON(preprocess.py 产出)")
    p.add_argument("--out", default="", help="单视频模式:标注结果输出路径")
    p.add_argument("--schema", default="schema/annotation.schema.json", help="校验用 schema")
    p.add_argument("--provider", default="qwen", choices=["qwen", "kimi", "local"],
                    help="标注供应商;local=本地 vLLM/SGLang 端点(须配 --local-base-url/--local-model)")
    p.add_argument("--verify-provider", default="", choices=["qwen", "kimi", "local"],
                    help="答案核验/复筛用另一家供应商(空=自动选;provider=local 时自动用本地模型)")
    p.add_argument("--window-sec", type=float, default=180.0)
    p.add_argument("--frames", type=int, default=10, help="每窗口采样帧数(场景标注)")
    p.add_argument("--qa", type=int, default=0,
                   help="每视频出题数(0=按时长自动:ceil(hours×24),下限 10)")
    p.add_argument("--no-describe", action="store_true", help="关闭全片详述(默认必产)")
    p.add_argument("--describe-mode", default="frames", choices=["frames", "native_video"],
                   help="frames=抽帧(Qwen/Kimi 通用);native_video=原生视频片段(仅 Qwen,需 ffmpeg+dashscope)")
    p.add_argument("--native-fps", type=float, default=2.0, help="native_video 模式下的视频抽帧率")
    p.add_argument("--frame-rate", type=float, default=1.0, help="帧索引抽帧率(1fps 默认)")
    p.add_argument("--local-base-url", default="", help="本地小模型端点(vLLM):provider=local 时的主端点,兼做帧级描述/音频判别")
    p.add_argument("--local-model", default="", help="本地视觉模型名(如 Qwen/Qwen3-VL-8B-Instruct)")
    p.add_argument("--max-windows", type=int, default=0, help=">0 时只处理前 N 窗(调试)")
    p.add_argument("--keep-shortcut", action="store_true", help="不剔除命中捷径的题(仅标记)")
    p.add_argument("--keep-single-frame", action="store_true",
                   help="opt-in:保留单帧可答的视觉题(默认剔除,D1 修复)")
    p.add_argument("--no-verify", action="store_true", help="关闭双模型答案核验")
    p.add_argument("--workers", type=int, default=4, help="单视频内的阶段并发线程数")
    p.add_argument("--video-workers", type=int, default=1,
                   help="批量模式:同时标注几个视频(默认 1 串行);总在途请求 ≈ "
                        "video-workers × workers,按供应商 QPS 上限调")
    args = p.parse_args()

    cfg = RunConfig(
        provider=args.provider,
        verify_provider=args.verify_provider,
        window_sec=args.window_sec,
        frames_per_window=args.frames,
        qa_override=args.qa,
        describe=not args.no_describe,
        describe_mode=args.describe_mode,
        native_fps=args.native_fps,
        frame_rate=args.frame_rate,
        local_base_url=args.local_base_url,
        local_vision_model=args.local_model,
        max_windows=args.max_windows,
        drop_shortcut=not args.keep_shortcut,
        keep_single_frame=args.keep_single_frame,
        verify_answers=not args.no_verify,
        workers=args.workers,
    )

    if args.inp:
        # ---------------- 批量模式 ----------------
        files = _collect(args.inp)
        if not files:
            raise SystemExit(f"未找到 structure 文件: {args.inp}")
        log.info("批量标注 %d 个文件(provider=%s)", len(files), args.provider)
        log.info("输入 structure:%s;输出 annotations:%s/<id>.json",
                 os.path.abspath(args.inp) if os.path.isdir(args.inp) else args.inp,
                 os.path.abspath(args.out_dir))
        os.makedirs(args.out_dir, exist_ok=True)

        def _annotate_one(item):
            """标注一个视频;失败返回该 structure 路径,成功返回 None。"""
            i, f = item
            try:
                with open(f, encoding="utf-8") as fh:
                    vid = json.load(fh).get("video_id") or \
                        os.path.splitext(os.path.basename(f))[0]
            except Exception as e:  # noqa: BLE001
                log.error("[%d/%d] 读取 structure 失败: %s (%s)", i, len(files), f, e)
                return f
            try:
                meta_path = _build_meta(vid, args.meta_dir, args.videos_dir, f, args.lang)
                out = os.path.join(args.out_dir, f"{vid}.json")
                log.info("[%d/%d] 标注: %s", i, len(files), vid)
                annotate_video(meta_path, f, out, cfg, schema_path=args.schema)
            except Exception as e:  # noqa: BLE001
                log.error("[%d/%d] 标注失败: %s (%s)", i, len(files), vid, e)
                return f
            return None

        tasks = list(enumerate(files, 1))
        video_workers = max(1, min(args.video_workers, len(tasks)))
        if video_workers == 1:
            results = [_annotate_one(t) for t in tasks]
        else:
            # 每个视频自带独立的 manifest / 缓存目录 / trace 文件,彼此不共享可变状态。
            # 但总并发 = video_workers × workers,超过供应商 QPS 只会换来一片 429。
            log.info("跨视频并发 %d;单视频内阶段并发 %d,合计约 %d 个在途请求",
                     video_workers, cfg.workers, video_workers * cfg.workers)
            with ThreadPoolExecutor(max_workers=video_workers) as pool:
                results = list(pool.map(_annotate_one, tasks))
        failed = [f for f in results if f]

        if failed:
            log.error("批次完成,失败 %d 个: %s", len(failed), ", ".join(failed))
            sys.exit(1)
        log.info("批次全部完成,共 %d 部;标注产物见 %s/", len(files), os.path.abspath(args.out_dir))
    else:
        # ---------------- 单视频模式 ----------------
        if not (args.meta and args.structure and args.out):
            p.error("请提供 --in(批量)或 --meta/--structure/--out(单视频)")
        annotate_video(args.meta, args.structure, args.out, cfg, schema_path=args.schema)


if __name__ == "__main__":
    main()