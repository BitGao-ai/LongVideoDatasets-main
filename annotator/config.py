# -*- coding: utf-8 -*-
"""供应商与运行配置。

统一走 OpenAI 兼容接口:Qwen 用 DashScope 兼容端点,Kimi 用 Moonshot 端点。
模型名可按你实际开通的版本改(下方为撰写时的可用名;例如用户说的
"Qwen3.7-plus" 并非真实模型 id,请替换成 qwen-vl-max / qwen2.5-vl-* 等)。
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    vision_model: str   # 多模态(图文)模型:场景理解 / 单帧过滤用
    text_model: str     # 纯文本模型:全局聚合 / 出题 / 盲答过滤用


PROVIDERS = {
    # 阿里云 DashScope(百炼)——OpenAI 兼容模式
    "qwen": ProviderConfig(
        name="qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        vision_model="qwen-vl-max",       # 备选: qwen-vl-max-latest / qwen2.5-vl-72b-instruct
        text_model="qwen-plus",           # 备选: qwen-max
    ),
    # Moonshot Kimi —— OpenAI 兼容
    "kimi": ProviderConfig(
        name="kimi",
        base_url="https://api.moonshot.cn/v1",
        api_key_env="MOONSHOT_API_KEY",
        vision_model="moonshot-v1-32k-vision-preview",  # 备选: kimi-latest
        text_model="moonshot-v1-32k",
    ),
}


@dataclass
class RunConfig:
    provider: str = "qwen"

    # 分块与抽帧(针对 30min–2h 长视频)
    window_sec: float = 180.0          # 场景窗口时长(镜头合并上限)
    frames_per_window: int = 10        # 每个窗口送入多模态模型的采样帧数
    max_image_edge: int = 768          # 图像长边下采样,控制 token/成本
    max_windows: int = 0               # >0 时只处理前 N 个窗口(调试用),0=全部

    # 出题
    qa_per_video: int = 12
    cross_scene_ratio: float = 0.4     # 要求 ≥ 该比例的题为 cross_scene / whole_video

    # 全片详细描述(可选,opt-in)
    describe: bool = False             # 标注流水线是否附带详述阶段
    describe_mode: str = "frames"      # "frames"(抽帧,Qwen/Kimi 通用)| "native_video"(原生视频,仅 Qwen)
    describe_frames_per_window: int = 0  # frames 模式:0=复用 frames_per_window;详述通常可给更多帧
    describe_agg_group: int = 15       # 汇总时的分组大小(段数超阈值走两级汇总)
    native_fps: float = 2.0            # native_video 模式:DashScope 对视频片段的抽帧率
    native_clip_dir: str = ""          # native_video 模式:切片临时目录;空=系统临时目录且用完即删

    # 抗捷径过滤
    drop_shortcut: bool = True         # 命中盲答 / 字幕捷径是否直接剔除

    # 通用
    temperature: float = 0.2
    max_retries: int = 4
    request_timeout: int = 120

    def provider_cfg(self) -> ProviderConfig:
        if self.provider not in PROVIDERS:
            raise ValueError(f"未知 provider: {self.provider};可选 {list(PROVIDERS)}")
        return PROVIDERS[self.provider]
