# -*- coding: utf-8 -*-
"""统一的多模态 LLM 客户端(OpenAI 兼容,Qwen / Kimi 通用)。

- complete(): 一次对话补全,支持图文(vision)与纯文本;带指数退避重试;
  want_json 时返回已解析的 dict。
- 纯文本调用尝试用 response_format=json_object 强约束;多模态调用不用该参数
  (部分 vision 模型不支持),改用提示词 + 稳健 JSON 抽取。
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import RunConfig

log = logging.getLogger("annotator.llm")


# --------- 消息内容构造 ---------
def text_part(t: str) -> Dict[str, Any]:
    return {"type": "text", "text": t}


def image_part(b64_jpeg: str) -> Dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"}}


# --------- 稳健 JSON 抽取 ---------
def extract_json(text: Optional[str]) -> Any:
    if not text:
        raise ValueError("模型返回为空")
    # 去掉 ```json ... ``` 围栏
    m = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.S)
    raw = m.group(1) if m else text
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 兜底:截取首个 { 到末个 }(或 [ ... ])
        for lb, rb in (("{", "}"), ("[", "]")):
            s, e = raw.find(lb), raw.rfind(rb)
            if s != -1 and e != -1 and e > s:
                try:
                    return json.loads(raw[s:e + 1])
                except json.JSONDecodeError:
                    continue
        raise


class LLMClient:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.pc = cfg.provider_cfg()
        import os
        key = os.environ.get(self.pc.api_key_env)
        if not key:
            raise RuntimeError(
                f"缺少 API Key:请设置环境变量 {self.pc.api_key_env}(provider={self.pc.name})"
            )
        self.client = OpenAI(api_key=key, base_url=self.pc.base_url, timeout=cfg.request_timeout)

    def _model(self, vision: bool) -> str:
        return self.pc.vision_model if vision else self.pc.text_model

    def complete(
        self,
        messages: List[Dict[str, Any]],
        vision: bool = False,
        temperature: Optional[float] = None,
        want_json: bool = True,
    ) -> Any:
        temperature = self.cfg.temperature if temperature is None else temperature
        use_rf = want_json and not vision   # response_format 仅用于纯文本模型
        last_err: Optional[Exception] = None

        for attempt in range(self.cfg.max_retries):
            try:
                kwargs: Dict[str, Any] = dict(
                    model=self._model(vision), messages=messages, temperature=temperature
                )
                if use_rf:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content
                return extract_json(content) if want_json else content
            except Exception as e:  # noqa: BLE001 —— 模板层统一兜底
                last_err = e
                # 若因 response_format 不支持而失败,则关闭该参数再试
                if use_rf and "response_format" in str(e).lower():
                    use_rf = False
                    log.warning("模型不支持 response_format,已关闭后重试")
                    continue
                wait = min(2 ** attempt, 30)
                log.warning("LLM 调用失败(第 %d/%d 次): %s;%ds 后重试",
                            attempt + 1, self.cfg.max_retries, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"LLM 调用多次失败: {last_err}")
