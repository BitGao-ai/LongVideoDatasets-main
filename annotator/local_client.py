# -*- coding: utf-8 -*-
"""本地小模型客户端(Res 0.D 两级模型策略:本地小模型做重活)。

通过 OpenAI 兼容端点(vLLM / SGLang / Ollama)调用本地视觉小模型,
承担两类“重活”:
  1. 1fps 帧级短描述(≤30 字,Res 2.1 帧级轨);
  2. 音频事件类型判别(音乐/音效/掌声/噪声,Res 1.3)。

未配置(local_base_url/local_vision_model 为空)时视为 disabled,
调用方应降级:帧描述用云端模型降频采样、音频事件类型走规则判别。
"""

import base64
import logging
from typing import Dict, List, Optional

from .config import RunConfig
from .utils import quiet_http_loggers

log = logging.getLogger("annotator.local")


class LocalLLMClient:
    """本地视觉小模型(OpenAI 兼容)。所有方法在 disabled 时返回 None/[]。"""

    def __init__(self, cfg: RunConfig):
        self.enabled = bool(cfg.local_base_url and cfg.local_vision_model)
        self.model = cfg.local_vision_model
        self._client = None
        if self.enabled:
            try:
                from openai import OpenAI
                self._client = OpenAI(base_url=cfg.local_base_url, api_key=cfg.local_api_key)
                quiet_http_loggers()
                log.info("本地模型已启用: %s @ %s", self.model, cfg.local_base_url)
            except Exception as e:  # noqa: BLE001
                log.warning("本地模型初始化失败,将降级为云端/规则: %s", e)
                self.enabled = False

    # ------------------------------------------------------------------ caption
    def caption_frame(self, image_b64: str, ctx: str = "", max_tokens: int = 48) -> Optional[str]:
        """单帧短描述(≤30 字),失败返回 None(调用方降级)。"""
        if not self.enabled or not self._client:
            return None
        try:
            content = [{"type": "text", "text": ctx or "用不超过30个中文字描述这张画面。"}]
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
            resp = self._client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": content}],
                temperature=0.1, max_tokens=max_tokens)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            log.warning("本地模型单帧描述失败: %s", e)
            return None

    def caption_frames(self, store, times: List[float],
                       ctx: str = "") -> List[Dict[str, object]]:
        """对一批时刻逐帧出短描述(本地模型免费,循环即可;返回 [{t, text}])。"""
        out: List[Dict[str, object]] = []
        if not self.enabled:
            return out
        for i, t in enumerate(times, 1):
            b64 = store.frame_b64(t)
            if not b64:
                continue
            text = self.caption_frame(b64, ctx)
            if text:
                out.append({"t": round(float(t), 3), "text": text})
            if i % 240 == 0:
                log.info("帧描述进度: %d/%d", i, len(times))
        log.info("本地帧描述完成:%d/%d 帧", len(out), len(times))
        return out

    # ------------------------------------------------------------------ audio
    def classify_audio(self, rms: float, rms_ratio: float, dur: float,
                       asr_context: str = "") -> Dict[str, object]:
        """音频事件类型判别(Res 1.3):{type, desc}。

        特征:rms(均方根能量)、rms_ratio(相对语音段能量比)、dur(段长)。
        本地模型不可用时返回空 dict,调用方走规则判别。
        """
        if not self.enabled or not self._client:
            return {}
        try:
            feat = {"rms": round(rms, 4), "rms_ratio": round(rms_ratio, 2), "dur_s": round(dur, 2)}
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": (
                    f"这是视频中一段非语音音频段的声学特征: {feat}。\n"
                    f"前后文 ASR(可空): {asr_context[:200] or '(无)'}\n"
                    "请判定类型,只输出 JSON: {\"type\": \"music\"|\"sfx\"|\"applause\"|\"noise\"|\"other\", "
                    "\"desc\": \"不超过20字描述\"}")}],
                temperature=0.1, max_tokens=80)
            from .llm_client import extract_json
            return extract_json(resp.choices[0].message.content) or {}
        except Exception as e:  # noqa: BLE001
            log.debug("本地音频分类失败: %s", e)
            return {}
