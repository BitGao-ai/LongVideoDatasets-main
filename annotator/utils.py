# -*- coding: utf-8 -*-
"""公共工具:内容哈希 / 文本度量 / 相似度 / 退避重试 / 覆盖率工具。

纯标准库 + numpy,被各阶段模块共享,保证幂等与可审计。
"""

import hashlib
import json
import logging
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("annotator.utils")

# ---------------------------------------------------------------- 日志控制
def quiet_http_loggers() -> None:
    """抑制 httpx/httpx2/httpcore(openai SDK 底层 HTTP 客户端)的 INFO 访问日志。

    openai SDK 每次创建客户端会注册新 logger(httpx / httpx1 / httpx2 ...;
    新版 openai 依赖的分叉包名即为 httpx2),故对已知前缀与已注册 logger
    全部降级,避免每次 LLM 调用刷屏。
    """
    prefixes = ("httpx", "httpcore")
    for name in list(logging.root.manager.loggerDict):
        if name.startswith(prefixes):
            logging.getLogger(name).setLevel(logging.WARNING)
    for name in prefixes:
        logging.getLogger(name).setLevel(logging.WARNING)

# ---------------------------------------------------------------- 内容哈希
def sha256_of(obj: Any) -> str:
    """对任意 JSON 化对象做内容哈希(用于 manifest 幂等 / 缓存 key)。"""
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    """对文件做内容哈希(大文件分块读)。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 文本度量
_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")


def zh_len(s: Optional[str]) -> int:
    """中文“字数”:去空白后的字符数(中文/英文/数字各算 1)。"""
    if not s:
        return 0
    return len(re.sub(r"\s+", "", s))


def en_len(s: Optional[str]) -> int:
    """英文“词数”。"""
    if not s:
        return 0
    return len(str(s).split())


def lang_len(s: Optional[str], lang: str) -> int:
    return zh_len(s) if lang == "zh" else en_len(s)


_ANCHOR_RE = re.compile(r"(约|大约|around|approximately|~|near)?\s*(\d{1,2}):(\d{2})")


def count_time_anchors(text: Optional[str]) -> int:
    """统计文本中的时间锚点("约 MM:SS"),用于详述质量门(Res 2.2)。"""
    if not text:
        return 0
    return len(_ANCHOR_RE.findall(str(text)))


#: 序数/时序消歧词。中英分开写:中文没有词边界概念,英文必须加 \b,否则
#: "strengthen" 里的 then、"against" 里的 again 都会让模糊题干通过校验(B16)。
_ORDINAL_ZH_RE = re.compile(
    r"(第[一二三四五六七八九十百]+次?|最后一?次?|最初|最先|首次|再一次|又一次|"
    r"初次|初始|先前|此前|之后|随后|后来|开场|结尾|片尾|片头)"
)
_ORDINAL_EN_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|last|final|finally|again|initial|initially|"
    r"earlier|previously|later|afterwards?|beginning|ending)\b",
    re.IGNORECASE,
)


def has_referring_ctx(text: Optional[str]) -> bool:
    """题干“指代上下文”校验(Res 5.1):含时间/序数消歧/命名短语即达标。

    - 时间描述:"第 12 分钟" / "约 12:30"
    - 序数消歧:"第二次 / 最后一次" / "the first time"
    - 事件名/地点/人物锁定短语(带引号引用)
    """
    t = str(text or "")
    if not t:
        return False
    if _ANCHOR_RE.search(t) or _ORDINAL_ZH_RE.search(t) or _ORDINAL_EN_RE.search(t):
        return True
    # 带引号的锁定短语:"争吵" / "in the office"
    if re.search(r"[“\"『「]([^”\"』」]{2,})[”\"』」]", t):
        return True
    return False


# ---------------------------------------------------------------- 相似度
def _ngrams(text: str, n: int = 3) -> set:
    chars = re.sub(r"\s+", "", str(text or ""))
    if len(chars) <= n:
        return {chars}
    return {chars[i:i + n] for i in range(len(chars) - n + 1)}


def ngram_similarity(a: Optional[str], b: Optional[str], n: int = 3) -> float:
    """字符 n-gram Jaccard 相似度(纯文本兜底,不依赖 embedding 服务)。"""
    ga, gb = _ngrams(a, n), _ngrams(b, n)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def interval_overlap_ratio(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """b 与 a 的时间重叠比例 = 交 / min(各自长度)(用于去重与捷径判定)。"""
    sa, ea = a
    sb, eb = b
    inter = max(0.0, min(ea, eb) - max(sa, sb))
    denom = min(ea - sa, eb - sb)
    if denom <= 0:
        return 0.0
    return inter / denom


def miou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """区间 mIoU(对称):交 / 并。temporal_grounding 验证用(Res 8.2)。"""
    sa, ea = a
    sb, eb = b
    inter = max(0.0, min(ea, eb) - max(sa, sb))
    union = max(ea, eb) - min(sa, sb)
    if union <= 0:
        return 0.0
    return inter / union


#: 非对称补边时,两侧各自至少分到的余量比例(其余按内容哈希分配)。
_PAD_MIN_FRAC = 0.2


def asym_pad(seed: Any, width: float, base_pad: float,
             min_frac: float = _PAD_MIN_FRAC) -> Tuple[float, float]:
    """把马赛克的时间余量非对称地拆到区间两侧,返回 (左侧余量, 右侧余量)。

    对称补边会让待判定区间恒定落在视野正中,模型只要答"正中间那段"就能拿到
    高 IoU —— 边界复核的 verified 率与 temporal 的 mIoU 门都会因此系统性虚高。
    这里按 seed 的内容哈希分配左右比例:同一输入每次得到同一视野(可复现、
    可缓存),不同输入的偏移方向不同,位置先验失效。视野总宽度与对称补边一致,
    不额外增加取帧与 token 成本。
    """
    total = 2.0 * max(float(base_pad), float(width))
    frac = min_frac + (1.0 - 2.0 * min_frac) * (int(sha256_of(seed)[:8], 16) % 1000) / 999.0
    left = total * frac
    return round(left, 3), round(total - left, 3)


def text_similarity(client: Any, a: Optional[str], b: Optional[str],
                    threshold_fallback: bool = True) -> float:
    """文本相似度:优先 embedding(供应商支持),失败回退 n-gram Jaccard。

    client 为 LLMClient 实例;embedding 不可用时返回 n-gram 相似度。
    """
    if not a or not b:
        return 0.0
    try:
        emb = client.embed([a, b])
        if emb and len(emb) == 2:
            return cosine(emb[0], emb[1])
    except Exception as e:  # noqa: BLE001 —— 回退不阻断
        log.debug("embedding 不可用,回退 n-gram: %s", e)
    if threshold_fallback:
        return ngram_similarity(a, b)
    return 0.0


# ---------------------------------------------------------------- 退避重试
def retry_exp_backoff(fn: Callable[[], Any], retries: int = 3, base: float = 2.0,
                      max_wait: float = 30.0, logger: Optional[logging.Logger] = None) -> Tuple[bool, Any, Optional[Exception]]:
    """指数退避重试:返回 (ok, result_or_None, error_or_None)。"""
    lg = logger or log
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return True, fn(), None
        except Exception as e:  # noqa: BLE001 —— 调用方负责语义化
            last_err = e
            if attempt >= retries:
                break
            wait = min(base ** attempt, max_wait)
            lg.warning("操作失败(第 %d/%d 次): %s;%.1fs 后重试", attempt, retries, e, wait)
            time.sleep(wait)
    return False, None, last_err


# ---------------------------------------------------------------- 时间轴
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def coverage_gaps(windows: Sequence[Tuple[float, float]], duration: float,
                  eps: float = 1.0) -> List[Dict[str, float]]:
    """覆盖校验(Res 3.6):union(windows ∪ gaps) 应覆盖 [0, duration]。

    返回未覆盖区间列表(全片未覆盖时会暴露在 gaps 中,任何 gaps 都导致验收不过)。
    """
    if not windows:
        return [{"start": 0.0, "end": duration}] if duration > eps else []
    merged: List[Tuple[float, float]] = []
    for s, e in sorted(windows):
        s, e = clamp(s, 0.0, duration), clamp(e, 0.0, duration)
        if e <= s:
            continue
        if merged and s <= merged[-1][1] + eps:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    gaps: List[Dict[str, float]] = []
    cursor = 0.0
    for s, e in merged:
        if s - cursor > eps:
            gaps.append({"start": round(cursor, 3), "end": round(s, 3)})
        cursor = max(cursor, e)
    if duration - cursor > eps:
        gaps.append({"start": round(cursor, 3), "end": round(duration, 3)})
    return gaps


def duration_bucket(sec: float) -> str:
    if sec < 300:
        return "<5m"
    if sec < 900:
        return "5-15m"
    if sec < 2400:
        return "15-40m"
    if sec <= 7200:
        return "40m-2h"
    return ">2h"
