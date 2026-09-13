# -*- coding: utf-8 -*-
"""供应商与运行配置。

统一走 OpenAI 兼容接口:Qwen 用 DashScope 兼容端点,Kimi 用 Moonshot 端点。
模型名可按你实际开通的版本改(下方为撰写时的可用名)。

两级模型策略(成本控制,见 Res.txt §0):
- 本地小模型(local_*)做重活:1fps 帧级短描述 / 音频事件类型判别;
- API 大模型做精活:镜头/片段级详述、事件边界复核、出题、答案核验。
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------- 内置默认密钥
# 来源: /Users/bityangzi/python/lingxi/默认业务空间-apiKey-6375308.xlsx
#   - tokenplan sheet (A1): sk-sp-... 为百炼 Token Plan 公共端点密钥
#   - 默认业务空间 sheet: sk-ws-... 为企业版 MaaS 私有端点密钥
# 优先级: 环境变量 > 内置默认值 > .env 文件(见下方 _load_dotenv)
# 这样 clone 后不配置环境变量也能直接跑;线上/CI 通过环境变量覆盖即可。
DEFAULT_DASHSCOPE_API_KEY = "sk-sp-H.YPEML.ugTC.MEUCIBiqFI-s5pSQCK87NMBeuXP936Rp1NMWUhgQ3Md8qmk1AiEAsxwK5Un2x1IwY_hHXoT_zweTPmqZ90IAQHOZ3CLV-Fo"
DEFAULT_WORKSPACE_API_KEY = "sk-ws-H.EIPRIMY.gNWQ.MEUCIQDUpIXzjdRdGjY3OMADt7XI-Gy86ZYi25A0rTM6IcHV1AIgEwe9qWBdZQB2rEmzV5G_QPGPr4kDpWL94YO_qvVoyZ4"
DEFAULT_WORKSPACE_BASE_URL = "https://llm-7dfb5cb9gwefuqdz.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_WORKSPACE_ID = "llm-7dfb5cb9gwefuqdz"
DEFAULT_OPENAI_COMPATIBLE_URL = DEFAULT_WORKSPACE_BASE_URL  # 别名,见 Excel 描述

# 环境变量 -> 默认值的映射,供 LLMClient fallback
DEFAULT_API_KEYS: Dict[str, str] = {
    "DASHSCOPE_API_KEY": DEFAULT_DASHSCOPE_API_KEY,
    "MAAS_API_KEY": DEFAULT_WORKSPACE_API_KEY,
    "MOONSHOT_API_KEY": "",  # 未提供,可按需在 .env 中配置
}

# Workspace 额外信息(仅作文档/调试用,不直接参与鉴权)
WORKSPACE_INFO = {
    "workspace_name": "默认业务空间",
    "workspace_id": DEFAULT_WORKSPACE_ID,
    "apiHost": "llm-7dfb5cb9gwefuqdz.cn-beijing.maas.aliyuncs.com",
    "dashScope": "https://llm-7dfb5cb9gwefuqdz.cn-beijing.maas.aliyuncs.com/api/v1",
}


def _load_dotenv(dotenv_path: str = ".env") -> None:
    """极简 .env 加载器(不依赖 python-dotenv)。

    仅在环境变量缺失时注入,已有的环境变量优先级更高,符合 12-Factor。
    """
    if os.path.exists(dotenv_path):
        try:
            with open(dotenv_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass
    # 也尝试从项目根目录加载(当 annotator 作为包被导入时 CWD 可能不同)
    try:
        import pathlib
        root_env = pathlib.Path(__file__).resolve().parents[1] / ".env"
        if str(root_env) != dotenv_path and root_env.exists():
            with open(root_env, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass


# 模块导入时自动尝试加载 .env(失败静默,不影响无 .env 环境)
_load_dotenv()


def get_api_key(env_name: str) -> str:
    """按优先级返回 API Key: 环境变量 > 内置默认值 > 空字符串。

    - 高内聚:所有 key 解析收口于此,调用方无需散落 os.environ.get
    - 支持 MAAS_API_KEY 未配置时复用 DASHSCOPE_API_KEY(兼容历史配置)
    """
    if not env_name:
        return ""
    val = os.environ.get(env_name, "").strip()
    if val:
        return val
    # maas 家族允许复用 DASHSCOPE_API_KEY 中的 ws 密钥
    if env_name == "MAAS_API_KEY":
        alt = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if alt.startswith("sk-ws-"):
            return alt
        # 也尝试从内置默认值兜底
        if DEFAULT_WORKSPACE_API_KEY:
            return DEFAULT_WORKSPACE_API_KEY
    return DEFAULT_API_KEYS.get(env_name, "").strip()


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    vision_model: str   # 多模态(图文)模型:场景理解 / 单帧过滤用
    text_model: str     # 纯文本模型:全局聚合 / 出题 / 盲答过滤用


PROVIDERS = {
    # 阿里云 DashScope(百炼)——OpenAI 兼容模式(公共端点)
    # 默认 key 来源: Excel tokenplan sheet (sk-sp-...)
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
    # 企业版 MaaS 私有端点(默认业务空间)
    # 默认 key 来源: Excel 默认业务空间 sheet (sk-ws-...)
    # base_url: https://llm-7dfb5cb9gwefuqdz.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
    "maas": ProviderConfig(
        name="maas",
        base_url=DEFAULT_WORKSPACE_BASE_URL,
        api_key_env="MAAS_API_KEY",
        vision_model="qwen-vl-max",
        text_model="qwen-plus",
    ),
    # 别名: qwen-maas / workspace 均指向同一私有端点,兼容不同文档命名
    "qwen-maas": ProviderConfig(
        name="qwen-maas",
        base_url=DEFAULT_WORKSPACE_BASE_URL,
        api_key_env="DASHSCOPE_API_KEY",  # 允许复用 DASHSCOPE_API_KEY 注入 sk-ws-*
        vision_model="qwen-vl-max",
        text_model="qwen-plus",
    ),
    "workspace": ProviderConfig(
        name="workspace",
        base_url=DEFAULT_WORKSPACE_BASE_URL,
        api_key_env="MAAS_API_KEY",
        vision_model="qwen-vl-max",
        text_model="qwen-plus",
    ),
}

def resolve_base_url(provider_name: str, api_key: str = "") -> str:
    """根据 provider 与 key 前缀智能选择 base_url(独立、通用)。

    - maas/workspace/qwen-maas 固定返回私有端点(优先 MAAS_BASE_URL 覆盖)
    - qwen 若传入 sk-ws-* 前缀的 key 自动切换到私有端点(兼容用户把 ws 密钥配到 DASHSCOPE_API_KEY)
      此时优先使用 MAAS_BASE_URL / 私有端点,而非 DASHSCOPE_BASE_URL
    - 支持通过环境变量显式覆盖: DASHSCOPE_BASE_URL (仅 qwen) / MAAS_BASE_URL (私有端点族)
    - 否则返回 PROVIDERS 中配置的默认 base_url
    """
    # 私有端点族固定走 MAAS 私有端点, 不受 DASHSCOPE_BASE_URL 影响
    if provider_name in ("maas", "workspace", "qwen-maas"):
        return os.environ.get("MAAS_BASE_URL", "").strip() or DEFAULT_WORKSPACE_BASE_URL
    # qwen + ws 前缀自动切换到私有端点(高内聚:配置层统一处理,调用方无需手动选 maas)
    if provider_name == "qwen" and api_key.strip().startswith("sk-ws-"):
        return os.environ.get("MAAS_BASE_URL", "").strip() or DEFAULT_WORKSPACE_BASE_URL
    # 通用覆盖:仅对 qwen 生效,方便更换模型/端点(配置通用且独立)
    if provider_name == "qwen":
        env_override = os.environ.get("DASHSCOPE_BASE_URL", "").strip()
        if env_override:
            return env_override
    pc = PROVIDERS.get(provider_name)
    return pc.base_url if pc else DEFAULT_WORKSPACE_BASE_URL


# 本地部署模型(vLLM / SGLang / Ollama,OpenAI 兼容端点)。
LOCAL_PROVIDER_NAME = "local"


@dataclass
class RunConfig:
    # ---------------------------------------------------------------- 供应商
    provider: str = "qwen"              # 主标注供应商
    verify_provider: str = ""           # 答案核验/复筛供应商;空 = 自动选与 provider 不同的另一家

    # ---------------------------------------------------------- 分块与抽帧
    window_sec: float = 180.0           # 场景窗口时长(镜头合并上限)
    frames_per_window: int = 10         # 每个窗口送入多模态模型的采样帧数(场景标注)
    max_image_edge: int = 768           # 图像长边下采样,控制 token/成本
    max_windows: int = 0                # >0 时只处理前 N 个窗口(调试用),0=全部

    # ----------------------------------------------------- M0 帧索引化(Res 1.1/1.2)
    frame_rate: float = 1.0             # 帧索引抽帧率(1fps;越密越贵)
    frame_index_dir: str = ""           # 帧索引根目录;空 = 默认 <structure 所在目录>/frames
    force_frame_index: bool = False     # 已存在帧索引时是否强制重建

    # -------------------------------------------- M1 密集描述(Res 2,默认必产)
    describe: bool = True               # 标注流水线默认附带详述阶段(f1: 默认必产)
    describe_mode: str = "frames"       # "frames"(抽帧,Qwen/Kimi 通用)| "native_video"(原生视频,仅 Qwen)
    describe_frames_per_window: int = 0  # frames 模式:0=复用 frames_per_window
    describe_agg_group: int = 15        # 汇总时的分组大小(段数超阈值走两级汇总)
    native_fps: float = 2.0             # native_video 模式:DashScope 对视频片段的抽帧率
    native_clip_dir: str = ""           # native_video 模式:切片临时目录;空=系统临时目录且用完即删
    # 质量门(Res 2.2/2.4)
    shot_desc_min_chars: int = 80       # 镜头级描述:中文长度下限
    shot_desc_min_words: int = 60       # 镜头级描述:英文词数下限
    segment_min_zh: int = 400           # 片段级描述:中文长度下限
    segment_min_en: int = 350           # 片段级描述:英文词数下限
    element_checklist_required: int = 6  # 要素清单 ≥ 6/8 才接受
    dense_anchor_min: int = 2           # 每段详述至少含的时间锚点数("约 MM:SS")
    dense_max_retries: int = 2          # 质量门未过的重试次数(含反馈)
    full_desc_min_zh: int = 1000        # 全片详述中文长度门(2h 视频应 2000+)
    strict_lang_check: bool = False     # 语言对齐回译校验(更严,成本更高)

    # --------------------------------------------- 本地小模型(两级策略)
    local_base_url: str = ""            # 如 http://127.0.0.1:8000/v1 (vLLM / SGLang)
    local_api_key: str = "EMPTY"
    local_vision_model: str = ""        # 如 Qwen/Qwen2.5-VL-8B-Instruct;空 = 禁用本地模型
    frame_caption_cloud_fps: float = 0.05  # 无本地模型时,云端帧描述抽帧率(1/20s,控成本)

    # ------------------------------------------------- M2 事件引擎(Res 3)
    event_sliding_window: float = 60.0  # 事件挖掘滑窗(步进 50% 叠加)
    event_min_width: float = 3.0        # 事件最小宽度(秒),key 事件必须 ≥ 该值
    event_density_min: float = 45.0     # 目标事件密度(个/小时)
    mosaic_pad_sec: float = 4.0         # 边界复核:span ±pad 取帧
    mosaic_frames: int = 64             # 8×8 马赛克帧数
    boundary_max_rounds: int = 2        # Refined/Split 后二次复核上限轮数
    dedup_similarity: float = 0.85      # 去重:embedding 相似度阈值
    dedup_time_overlap: float = 0.5     # 去重:时间重叠比例阈值

    # ------------------------------------------------- M4 QA 引擎(Res 5)
    qa_per_hour: int = 24               # 题目数量按时长:ceil(duration_hour × 24)
    min_qa_per_video: int = 10          # 下限
    qa_override: int = 0                # >0 时固定出题数(调试/兼容旧 --qa)
    qa_batch_size: int = 20             # 超长视频分批出题,每批上限
    mcq_min_ratio: float = 0.5          # task_type 分布:mcq 50%~70%
    mcq_max_ratio: float = 0.7
    temporal_min_ratio: float = 0.15    # temporal_grounding 15%~25%
    temporal_max_ratio: float = 0.25
    min_qa_per_capability_layer: int = 2  # L1~L5 每层 ≥2 题
    cross_scene_min_ratio: float = 0.5  # cross_scene/whole_video ≥50%
    min_question_len: int = 25          # 题干平均字数 ≥25(LongVideoBench 参考)
    option_len_variance: float = 0.35   # 选项长度方差上限(正确项非最长)
    verify_answers: bool = True         # 双模型(VIDEO)答案核验开关
    verify_max_regen: int = 2           # 核验不一致打回重写上限
    drop_verify_mismatch: bool = True   # 重写后仍不一致 → 剔除(而非只打 quality_flag)

    # ------------------------------------------- M5 抗捷径 v2(Res 6)
    drop_shortcut: bool = True          # 命中捷径通道是否直接剔除
    keep_single_frame: bool = False     # opt-in:保留单帧可答的视觉题(默认剔除,修复 D1)
    synopsis_leak_check: bool = True    # 梗概泄漏检测通道
    language_prior_check: bool = True   # 语言先验通道
    temporal_shortcut_iou: float = 0.6  # temporal 字幕定位捷径:区间 IoU 阈值
    open_shortcut_sim: float = 0.85     # open/summary 盲答相似度阈值

    # ----------------------------------------- M7 QC v2 发布门槛(Res 8)
    alpha_threshold: float = 0.8        # MCQ 一致性门槛
    gold_threshold: float = 0.9         # 黄金题正确率门槛
    miou_threshold: float = 0.5         # temporal 验证者 mIoU 合格线
    miou_pass_ratio: float = 0.9        # mIoU 样本合格率 ≥90%
    open_sim_threshold: float = 0.5     # open/summary 单条作答"算一致"的相似度线
    open_agree_threshold: float = 0.6   # open/summary 一致率 ≥60%
    bilingual_min_ratio: float = 0.95   # 双语完备率 ≥95%
    min_qa_after_filter: int = 8        # 五通道过滤后每视频最少题数
    qc_human_spot: float = 0.1          # open/summary 人工抽测比例

    # ------------------------------------------------------------ 工程
    temperature: float = 0.2
    max_retries: int = 4                # LLM 调用退避重试次数
    request_timeout: int = 120
    max_window_retries: int = 3         # 窗口标注失败重试次数(指数退避),之后进 gaps
    workers: int = 4                    # 事件级并发的线程数
    trace_dir: str = "reports"          # LLM 调用审计 trace 输出目录(Res 10.4)

    # ------------------------------------------------------------ 方法
    def _local_provider_cfg(self) -> ProviderConfig:
        if not self.local_base_url or not self.local_vision_model:
            raise ValueError(
                "provider=local 需要提供本地端点:请设置 --local-base-url(如 "
                "http://127.0.0.1:8000/v1)与 --local-model(如 Qwen/Qwen3-VL-8B-Instruct)"
            )
        return ProviderConfig(
            name=LOCAL_PROVIDER_NAME,
            base_url=self.local_base_url,
            api_key_env="",              # 本地端点免 key
            vision_model=self.local_vision_model,
            text_model=self.local_vision_model,
        )

    def _resolve_provider(self, name: str) -> ProviderConfig:
        """通用解析:返回 ProviderConfig,如需则按 key 前缀/环境变量动态覆盖 base_url。

        高内聚:所有 provider 的 base_url 动态逻辑收口于此,调用方无需关心端点细节。
        低耦合:不修改全局 PROVIDERS,返回副本仅影响本次调用。
        """
        pc = PROVIDERS[name]
        api_key = get_api_key(pc.api_key_env)
        resolved_url = resolve_base_url(pc.name, api_key)
        if resolved_url != pc.base_url:
            return ProviderConfig(
                name=pc.name,
                base_url=resolved_url,
                api_key_env=pc.api_key_env,
                vision_model=pc.vision_model,
                text_model=pc.text_model,
            )
        return pc

    def provider_cfg(self) -> ProviderConfig:
        if self.provider == LOCAL_PROVIDER_NAME:
            return self._local_provider_cfg()
        if self.provider not in PROVIDERS:
            raise ValueError(f"未知 provider: {self.provider};可选 {list(PROVIDERS)}+{LOCAL_PROVIDER_NAME}")
        return self._resolve_provider(self.provider)

    def verify_provider_cfg(self) -> Optional[ProviderConfig]:
        if self.verify_provider:
            if self.verify_provider not in (*PROVIDERS, LOCAL_PROVIDER_NAME):
                raise ValueError(f"未知 verify_provider: {self.verify_provider}")
            if self.verify_provider == LOCAL_PROVIDER_NAME:
                if self.provider != LOCAL_PROVIDER_NAME:
                    raise ValueError("verify_provider=local 要求 provider 也为 local")
                return self._local_provider_cfg()
            if self.verify_provider == self.provider:
                raise ValueError("verify_provider 必须与 provider 不同(否则无交叉核验意义)")
            return self._resolve_provider(self.verify_provider)
        if self.provider == LOCAL_PROVIDER_NAME:
            return self._local_provider_cfg()
        for name in PROVIDERS:
            if name != self.provider:
                return self._resolve_provider(name)
        return None

    def qa_quota(self, duration_sec: float) -> int:
        if self.qa_override > 0:
            return self.qa_override
        hours = max(0.0, duration_sec) / 3600.0
        return max(self.min_qa_per_video, int(-(-hours * self.qa_per_hour // 1)))
