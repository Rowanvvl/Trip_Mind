"""请求轨迹（trace）。

一次问答一行记录，回答四个问题：
1. 走了哪条分支、为什么（路由判定理由）；
2. 召回了什么、精排分数多少；
3. 调了哪些外部工具、耗时多少、有没有降级；
4. 护栏与 critic 给了什么结论、最终引用了什么。

用途有三：排障、评测集自动积累、Ch19 的轨迹评估（实际步骤 vs 理想路径）。
存储：先写进程内环形缓冲（永远可用），`TRACE_ENABLED=true` 且有 Mongo 时异步落库。
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from common.config import lm_config, trip_config
from common.logging.logger import logger
from utils.format_utils import now_iso

_MAX_TRACES = 500


@dataclass
class Trace:
    trace_id: str
    session_id: str = ""
    query: str = ""
    started_at: str = field(default_factory=now_iso)
    started_ts: float = field(default_factory=time.time)

    route: Dict[str, Any] = field(default_factory=dict)          # 路由判定与理由
    retrieval: Dict[str, Any] = field(default_factory=dict)      # 各路线召回数、覆盖类型
    rerank: Dict[str, Any] = field(default_factory=dict)         # 精排分数
    tools: List[Dict[str, Any]] = field(default_factory=list)    # 外部工具调用
    guards: List[Dict[str, Any]] = field(default_factory=list)   # 护栏命中
    critique: Dict[str, Any] = field(default_factory=dict)       # critic 结论
    goal: Dict[str, Any] = field(default_factory=dict)           # 目标验收
    nodes: List[Dict[str, Any]] = field(default_factory=list)    # 节点耗时
    tokens: Dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0})
    citations: List[Dict[str, Any]] = field(default_factory=list)
    degraded: bool = False
    error: Optional[str] = None
    finished_ts: Optional[float] = None

    @property
    def latency_ms(self) -> int:
        end = self.finished_ts or time.time()
        return int((end - self.started_ts) * 1000)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "query": self.query,
            "started_at": self.started_at,
            "latency_ms": self.latency_ms,
            "route": self.route,
            "retrieval": self.retrieval,
            "rerank": self.rerank,
            "tools": self.tools,
            "guards": self.guards,
            "critique": self.critique,
            "goal": self.goal,
            "nodes": self.nodes,
            "tokens": self.tokens,
            "citations": self.citations,
            "degraded": self.degraded,
            "error": self.error,
        }


_buffer: Deque[Trace] = deque(maxlen=_MAX_TRACES)
_index: Dict[str, Trace] = {}


def start_trace(trace_id: str, session_id: str = "", query: str = "") -> Trace:
    trace = Trace(trace_id=trace_id, session_id=session_id, query=query)
    _buffer.append(trace)
    _index[trace_id] = trace
    if len(_index) > _MAX_TRACES:
        # 与 buffer 同步裁剪，避免索引无限增长
        _index.pop(next(iter(_index)), None)
    return trace


def get_trace(trace_id: str) -> Optional[Trace]:
    return _index.get(trace_id)


def _ensure(trace_id: str) -> Optional[Trace]:
    trace = _index.get(trace_id)
    if trace is None:
        logger.debug(f"[trace] 未找到 {trace_id}，已自动新建")
        trace = start_trace(trace_id)
    return trace


def record_node_latency(trace_id: str, node: str, cost_ms: int) -> None:
    trace = _ensure(trace_id)
    if trace is None:
        return
    trace.nodes.append({"node": node, "cost_ms": cost_ms})


def record_route(trace_id: str, **fields: Any) -> None:
    trace = _ensure(trace_id)
    if trace:
        trace.route.update(fields)


def record_retrieval(trace_id: str, **fields: Any) -> None:
    trace = _ensure(trace_id)
    if trace:
        trace.retrieval.update(fields)


def record_rerank(trace_id: str, **fields: Any) -> None:
    trace = _ensure(trace_id)
    if trace:
        trace.rerank.update(fields)


def record_tool(trace_id: str, tool: str, ok: bool, cost_ms: int, **extra: Any) -> None:
    trace = _ensure(trace_id)
    if trace is None:
        return
    trace.tools.append({"tool": tool, "ok": ok, "cost_ms": cost_ms, **extra})
    if not ok:
        trace.degraded = True


def record_guard(trace_id: str, layer: str, verdict: str, reason: str = "") -> None:
    trace = _ensure(trace_id)
    if trace:
        trace.guards.append({"layer": layer, "verdict": verdict, "reason": reason})


def record_critique(trace_id: str, passed: bool, issues: list, retried: bool = False) -> None:
    trace = _ensure(trace_id)
    if trace:
        trace.critique = {"passed": passed, "issues": issues, "retried": retried}


def record_goal(trace_id: str, achieved: bool, **fields: Any) -> None:
    trace = _ensure(trace_id)
    if trace:
        trace.goal = {"achieved": achieved, **fields}


def record_tokens(trace_id: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
    trace = _ensure(trace_id)
    if trace is None:
        return
    trace.tokens["prompt"] += int(prompt_tokens or 0)
    trace.tokens["completion"] += int(completion_tokens or 0)


def record_citations(trace_id: str, citations: List[Dict[str, Any]]) -> None:
    trace = _ensure(trace_id)
    if trace:
        trace.citations = citations or []


def finish_trace(trace_id: str, error: Optional[str] = None) -> Optional[dict]:
    trace = _index.get(trace_id)
    if trace is None:
        return None
    trace.finished_ts = time.time()
    trace.error = error
    payload = trace.to_dict()

    if trip_config.trace_enabled and not lm_config.enabled:
        # 没配模型时仍然记录，但降级标记为 True，避免把空答案当正常样本
        payload["degraded"] = True

    if trip_config.trace_enabled:
        _persist(payload)
    return payload


def _persist(payload: dict) -> None:
    """落 Mongo。失败只记日志，绝不影响请求。"""
    try:
        from utils.clients.mongo_history_utils import save_trace

        save_trace(payload)
    except Exception as exc:  # 依赖未装 / 库未启动都走这里
        logger.debug(f"[trace] 落库跳过：{exc}")


def recent_traces(limit: int = 20) -> List[dict]:
    return [t.to_dict() for t in list(_buffer)[-limit:]][::-1]


def reset() -> None:
    _buffer.clear()
    _index.clear()
