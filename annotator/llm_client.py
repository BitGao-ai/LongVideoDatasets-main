# -*- coding: utf-8 -*-
"""统一的多模态 LLM 客户端(OpenAI 兼容,Qwen / Kimi / 本地 vLLM 通用)。

- complete(): 一次对话补全,支持图文(vision)与纯文本;want_json 时返回已解析的 dict。
  重试区分错误类型:429 听供应商的 Retry-After、5xx/网络错走指数退避 + 抖动、
  4xx 参数类错误立即放弃(重试无意义)。
- 纯文本调用尝试用 response_format=json_object 强约束;多模态调用不用该参数
  (部分 vision 模型不支持),改用提示词 + 稳健 JSON 抽取。
- trace(Res 10.4):每次 LLM 调用记录 时间戳/模型/输入哈希/重试/token 用量/输出摘要
  -> reports/llm_trace_<vid>.jsonl,可复现、可审计;累计用量见 client.usage。
- embed(): 供应商 embedding 服务(仅 qwen 等支持;失败返回 None,调用方回退)。
"""

import datetime
import json
import logging
import os
import random
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .config import LOCAL_PROVIDER_NAME, RunConfig
from .utils import quiet_http_loggers, sha256_of

log = logging.getLogger("annotator.llm")

#: trace 是多线程共享的追加写文件(事件复核 / 抗捷径过滤都并发)
_TRACE_LOCK = threading.Lock()

#: 这些状态码重试没有意义(参数非法 / 鉴权 / 模型不存在 / 请求体过大 / 内容被拒)。
#: 此前所有异常一视同仁走退避重试,一次必然失败的调用要被内层重试 max_retries 次、
#: 再被阶段级 retry_exp_backoff 重试 3 次,白等两分钟才报出真正的原因。
#: 429(限流)与 5xx 不在此列,仍然重试。
_FATAL_STATUS = frozenset({400, 401, 403, 404, 413, 415, 422})

#: 退避上限;供应商显式给了 Retry-After 时可以等更久(它比我们更清楚窗口)。
_MAX_BACKOFF_SEC = 30.0
_MAX_RETRY_AFTER_SEC = 60.0


def _status_of(err: Exception) -> Optional[int]:
    status = getattr(err, "status_code", None)
    return status if isinstance(status, int) else None


def _is_fatal(err: Exception) -> bool:
    return _status_of(err) in _FATAL_STATUS


def _retry_after_seconds(err: Exception) -> Optional[float]:
    """读取供应商返回的 Retry-After(秒);缺失或不可解析时返回 None。"""
    headers = getattr(getattr(err, "response", None), "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return None
    raw = headers.get("retry-after")
    try:
        return max(0.0, float(raw)) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _backoff_seconds(attempt: int, err: Exception) -> float:
    """限流优先听供应商的;否则指数退避 + 抖动。

    抖动不可省:workers 个线程同时撞上 429 时,无抖动的退避会让它们一起醒来、
    再一起被拒,把一次限流放大成一轮又一轮的齐步重试。
    """
    after = _retry_after_seconds(err)
    if after is not None:
        return min(after, _MAX_RETRY_AFTER_SEC)
    base = min(2.0 ** attempt, _MAX_BACKOFF_SEC)
    return base * (0.5 + random.random() * 0.5)


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
        self.is_local = (self.pc.name == LOCAL_PROVIDER_NAME)
        key = os.environ.get(self.pc.api_key_env, "").strip() if self.pc.api_key_env else ""
        if not key:
            if self.is_local:
                key = cfg.local_api_key or "EMPTY"   # 本地端点(vLLM 等)免 key
            else:
                raise RuntimeError(
                    f"缺少 API Key:请设置环境变量 {self.pc.api_key_env}(provider={self.pc.name})"
                )
        self.client = OpenAI(api_key=key, base_url=self.pc.base_url, timeout=cfg.request_timeout)
        quiet_http_loggers()
        self.trace_path: Optional[str] = None
        #: 累计 token 用量(供 manifest 记账;没有它就无法核算单视频成本)
        self.usage: Dict[str, int] = {"calls": 0, "prompt_tokens": 0,
                                      "completion_tokens": 0, "total_tokens": 0}
        self._usage_lock = threading.Lock()

    def _model(self, vision: bool) -> str:
        return self.pc.vision_model if vision else self.pc.text_model

    # ------------------------------------------------------------------ trace
    def set_trace(self, trace_path: str) -> None:
        self.trace_path = trace_path
        os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)

    @staticmethod
    def _trace_digest(messages: List[Dict]) -> Tuple[str, int]:
        """输入摘要 = (文本部分的哈希, 图片张数)。

        此前直接 sha256_of(messages),会把全部 base64 图片 json.dumps 一遍再哈希:
        一个 10 帧窗口约 1.3MB,每次调用(成功与失败)都算一次,1 小时视频 1000+
        次调用就是 GB 级纯浪费的临时分配(C6)。图片只记张数,不进哈希。
        """
        texts: List[str] = []
        n_images = 0
        for m in messages or []:
            content = m.get("content")
            if isinstance(content, str):
                texts.append(content)
                continue
            for part in content or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    n_images += 1
                elif part.get("type") == "text":
                    texts.append(str(part.get("text") or ""))
        return sha256_of(texts)[:32], n_images

    def _trace(self, model: str, messages: List[Dict], retries: int, ok: bool,
               summary: str, error: str = "",
               usage: Optional[Dict[str, int]] = None) -> None:
        if not self.trace_path:
            return
        input_hash, n_images = self._trace_digest(messages)
        record = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "input_hash": input_hash,
            "images": n_images,
            "retries": retries,
            "ok": ok,
            "usage": usage or {},
            "out_summary": str(summary)[:200],
            "error": str(error)[:300],
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        try:
            # 事件复核 / 过滤阶段是多线程的,加锁避免交错写坏 jsonl
            with _TRACE_LOCK:
                with open(self.trace_path, "a", encoding="utf-8") as f:
                    f.write(line)
        except OSError as e:
            log.debug("trace 写入失败: %s", e)

    # ------------------------------------------------------------------ embed
    def embed(self, texts: List[str], model: str = "text-embedding-v3") -> Optional[List[List[float]]]:
        """文本 embedding(去重/相似度用)。供应商不支持时抛错,由调用方回退 n-gram。"""
        if not texts:
            return []
        resp = self.client.embeddings.create(model=model, input=[str(t) for t in texts])
        rows = sorted(resp.data, key=lambda d: d.index)
        return [r.embedding for r in rows]

    # ------------------------------------------------------------------ usage
    def _record_usage(self, resp: Any) -> Dict[str, int]:
        """累加本次调用的 token 用量;供应商不返回 usage 时按 0 计。"""
        u = getattr(resp, "usage", None)
        one = {k: int(getattr(u, k, 0) or 0)
               for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        with self._usage_lock:
            self.usage["calls"] += 1
            for k, v in one.items():
                self.usage[k] += v
        return one

    # ------------------------------------------------------------------ chat
    def complete(
        self,
        messages: List[Dict[str, Any]],
        vision: bool = False,
        temperature: Optional[float] = None,
        want_json: bool = True,
    ) -> Any:
        temperature = self.cfg.temperature if temperature is None else temperature
        # response_format 仅用于云端纯文本模型;本地(vLLM 等)多不支持,直接关闭
        use_rf = want_json and not vision and not self.is_local
        model = self._model(vision)
        last_err: Optional[Exception] = None
        retries = 0

        for attempt in range(self.cfg.max_retries):
            try:
                kwargs: Dict[str, Any] = dict(
                    model=model, messages=messages, temperature=temperature
                )
                if use_rf:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content
                result = extract_json(content) if want_json else content
                self._trace(model, messages, retries, True, result,
                            usage=self._record_usage(resp))
                return result
            except Exception as e:  # noqa: BLE001 —— 模板层统一兜底
                last_err = e
                retries = attempt + 1
                # 若因 response_format 不支持而失败,则关闭该参数再试
                if use_rf and "response_format" in str(e).lower():
                    use_rf = False
                    log.warning("模型不支持 response_format,已关闭后重试")
                    continue
                if _is_fatal(e):
                    log.error("LLM 调用返回 %s,重试无意义,立即放弃: %s", _status_of(e), e)
                    break
                wait = _backoff_seconds(attempt, e)
                log.warning("LLM 调用失败(第 %d/%d 次): %s;%.1fs 后重试",
                            attempt + 1, self.cfg.max_retries, e, wait)
                time.sleep(wait)
        self._trace(model, messages, retries, False, "", str(last_err))
        raise RuntimeError(f"LLM 调用多次失败: {last_err}")

    def complete_text(self, system: str, user: str, temperature: Optional[float] = None) -> str:
        """便捷:纯文本调用,返回原始字符串。"""
        return self.complete(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            vision=False, temperature=temperature, want_json=False,
        )
