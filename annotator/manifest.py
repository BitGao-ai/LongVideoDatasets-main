# -*- coding: utf-8 -*-
"""阶段 manifest(Res 1.5 / 10.2):断点续跑与幂等的基石。

- 每个阶段产物 state = pending | done | failed | skipped;
- 记录阶段输入内容哈希(sha256,非路径),哈希不变 → 直接复用输出 = 天然缓存;
- 下游阶段上游变更时自动失效(链式重算);
- 失败重试(指数退避)后落 failed,由 orchestrator 决定封板/阻塞。

manifest 文件: <structure_dir>/<video_id>.manifest.json
"""

import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional

from .utils import sha256_of

log = logging.getLogger("annotator.manifest")

STAGE_STRUCTURE = "structure"
STAGE_DESCRIBE = "describe"
STAGE_EVENTS = "events"
STAGE_AGGREGATE = "aggregate"
STAGE_QA = "qa"
STAGE_VERIFY = "verify"
STAGE_FILTER = "filter"

ALL_STAGES = (STAGE_STRUCTURE, STAGE_DESCRIBE, STAGE_EVENTS,
              STAGE_AGGREGATE, STAGE_QA, STAGE_VERIFY, STAGE_FILTER)

# 阶段依赖:key 依赖的 stage 失效时,key 必须重跑
_STAGE_DEPS = {
    STAGE_DESCRIBE: (STAGE_STRUCTURE,),
    STAGE_EVENTS: (STAGE_STRUCTURE, STAGE_DESCRIBE),
    STAGE_AGGREGATE: (STAGE_STRUCTURE, STAGE_EVENTS),
    STAGE_QA: (STAGE_STRUCTURE, STAGE_EVENTS, STAGE_AGGREGATE),
    STAGE_VERIFY: (STAGE_QA,),
    STAGE_FILTER: (STAGE_QA, STAGE_VERIFY),
}

STATE_PENDING = "pending"
STATE_DONE = "done"
STATE_FAILED = "failed"
STATE_SKIPPED = "skipped"


class Manifest:
    def __init__(self, video_id: str, manifest_path: str):
        self.video_id = video_id
        self.path = manifest_path
        self.data: Dict[str, Any] = {
            "video_id": video_id,
            "schema_version": "2.0",
            "created_date": datetime.date.today().isoformat(),
            "updated_date": None,
            "stages": {},
            "gaps": [],
            "stats": {},
            "human_todo": [],
            "frames": {},
        }

    # ------------------------------------------------------------------ io
    @classmethod
    def load_or_create(cls, video_id: str, manifest_path: str) -> "Manifest":
        m = cls(video_id, manifest_path)
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    m.data = json.load(f)
                if m.data.get("video_id") != video_id:
                    raise ValueError(f"manifest video_id 不一致: {m.data.get('video_id')} != {video_id}")
            except Exception as e:  # noqa: BLE001
                log.warning("manifest 读取失败(%s),重建: %s", manifest_path, e)
                m = cls(video_id, manifest_path)
        return m

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.data["updated_date"] = datetime.datetime.now().isoformat(timespec="seconds")
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ stage
    def set_stage(self, stage: str, state: str, input_hash: str = "",
                  output: Optional[str] = None, error: str = "") -> None:
        self.data.setdefault("stages", {})[stage] = {
            "state": state,
            "input_hash": input_hash,
            "output": output,
            "error": error,
            "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        self.save()

    def stage_state(self, stage: str) -> str:
        return self.data.get("stages", {}).get(stage, {}).get("state", STATE_PENDING)

    def stage_input_hash(self, stage: str) -> str:
        return self.data.get("stages", {}).get(stage, {}).get("input_hash", "")

    def stage_output(self, stage: str) -> Optional[str]:
        return self.data.get("stages", {}).get(stage, {}).get("output")

    def stage_done(self, stage: str) -> bool:
        return self.stage_state(stage) == STATE_DONE

    def invalidate_downstream(self, stage: str) -> None:
        """标记下游阶段为 pending(链式失效),供输入变更时触发重算。"""
        queue = [stage]
        seen = set()
        while queue:
            cur = queue.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            for st, deps in _STAGE_DEPS.items():
                if cur in deps:
                    self.data.setdefault("stages", {}).pop(st, None)
                    queue.append(st)
        self.save()

    def needs_run(self, stage: str, input_hash: str) -> bool:
        """该阶段是否需要运行:未 done 或输入哈希已变化。"""
        return not (self.stage_done(stage) and self.stage_input_hash(stage) == input_hash)

    # ------------------------------------------------------------------ data
    def add_gap(self, start: float, end: float, reason: str) -> None:
        self.data.setdefault("gaps", []).append(
            {"start": round(float(start), 3), "end": round(float(end), 3), "reason": reason})
        self.save()

    def clear_gaps(self) -> None:
        """清空 gaps。gaps 是 S1 的产物,重跑该阶段前必须先清,否则跨次运行累积重复区间。"""
        self.data["gaps"] = []
        self.save()

    def set_stats(self, **kw: Any) -> None:
        self.data.setdefault("stats", {}).update(kw)
        self.save()

    def add_human_todo(self, item: Dict[str, Any]) -> None:
        item = dict(item)
        item["created_date"] = datetime.date.today().isoformat()
        self.data.setdefault("human_todo", []).append(item)
        self.save()

    def add_human_todos(self, items: List[Dict[str, Any]]) -> None:
        for it in items:
            self.add_human_todo(it)

    def set_frames(self, **kw: Any) -> None:
        self.data["frames"] = kw
        self.save()

    def to_annotation_meta_extra(self) -> Dict[str, Any]:
        """供 annotation_meta 落盘的附加块。"""
        stages = {}
        for k, v in (self.data.get("stages") or {}).items():
            # input_hash 可能是 None(旧 manifest / 手工编辑),此前直接切片会
            # TypeError,而且发生在落盘最后一步,前面所有 LLM 开销全部作废(D5)
            h = v.get("input_hash") or ""
            stages[k] = {"state": v.get("state"), "input_hash": str(h)[:12]}
        return {
            "gaps": self.data.get("gaps", []),
            "stats": self.data.get("stats", {}),
            "human_todo": self.data.get("human_todo", []),
            "frames_index": self.data.get("frames", {}),
            "stages": stages,
            "manifest_version": "2.0",
        }


def compute_structure_hash(structure: Dict, meta: Dict) -> str:
    """structure 阶段输入哈希:预处理结构 + meta 内容(哈希不变 → 复用窗口标注)。"""
    return sha256_of({
        "structure": {k: structure.get(k) for k in ("shots", "subtitles", "ocr", "audio_events")},
        "meta": meta,
    })


class StageCache:
    """窗口级结果缓存(断点续跑到“窗口粒度”,Res 10.2/C)。

    键 = sha256(阶段 + 窗口输入内容哈希);进程中断后重启,已完成的窗口
    直接从磁盘取回,不重复烧钱。目录: <out 目录>/.cache/<video_id>/
    """

    def __init__(self, cache_dir: str):
        self.root = os.path.abspath(cache_dir)
        os.makedirs(self.root, exist_ok=True)

    def _path(self, stage: str, key: str) -> str:
        # key 为任意 JSON 化对象(dict/tuple/str),统一内容哈希防碰撞(修复:此前
        # 对 dict 直接切 [:32] 抛 TypeError,导致所有 cache.get/put 崩溃)
        return os.path.join(self.root, f"{stage}_{sha256_of(key)[:32]}.json")

    def get(self, stage: str, key: str) -> Optional[Any]:
        p = self._path(stage, key)
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f).get("result")
        except Exception as e:  # noqa: BLE001
            log.warning("缓存读取失败,视为未缓存(%s): %s", p, e)
            return None

    def put(self, stage: str, key: str, result: Any) -> None:
        try:
            with open(self._path(stage, key), "w", encoding="utf-8") as f:
                json.dump({"stage": stage, "key": key, "result": result},
                          f, ensure_ascii=False)
        except OSError as e:
            log.warning("缓存写入失败(不影响主流程): %s", e)
