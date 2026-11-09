"""查询图状态。

一条请求的完整生命周期都在这一个 dict 里流转。字段命名遵循三条规则：
1. 与 prompt 里的占位符同名（`slots` / `context` / `facts`），减少映射代码；
2. 「输入」与「中间产物」分开，便于打 trace 时只看中间产物；
3. 两个重试计数器**分开**：`critic_retry` 管重写，`goal_retry` 管补检索。
   合用一个会出现「critic 用完额度导致 goal 无法补救」。
"""
from __future__ import annotations

import copy
import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class QueryGraphState(TypedDict, total=False):
    # ---------- 标识 ----------
    session_id: str
    task_id: str
    user_id: str
    trace_id: str

    # ---------- 输入 ----------
    original_query: str
    is_stream: bool
    history: List[dict]

    # ---------- 理解层 ----------
    guard_result: dict          # {allowed, action, category, reply, flags}
    rewritten_query: str
    city: str
    cities: List[str]
    content_type: str
    entities: List[str]
    intent: str
    tool_flags: Dict[str, bool]
    need_clarify: bool
    clarify_question: str
    slots: Dict[str, Any]
    profile: Dict[str, Any]

    # ---------- 检索层 ----------
    dense_chunks: List[dict]
    sparse_chunks: List[dict]
    hyde_chunks: List[dict]
    web_docs: List[dict]
    rrf_chunks: List[dict]
    reranked_docs: List[dict]
    # 四路召回并行执行，三个分支都会写这个键。LastValue 通道一个超步只收一次写入，
    # 不加 reducer 就会抛 InvalidUpdateError（并行分支不要共写一个 key）。
    retrieval_degraded: Annotated[bool, operator.or_]

    # ---------- 证据评估 ----------
    evidence_types: List[str]       # 已覆盖的 content_type
    evidence_coverage: int          # 覆盖类型数
    missing_types: List[str]
    sub_queries: List[dict]
    sub_query_used: bool

    # ---------- 事实层 ----------
    route_facts: dict
    weather_facts: dict
    budget_facts: dict
    facts: dict                     # 三路合并结果

    # ---------- 上下文与生成 ----------
    context: str
    citations: List[dict]
    prompt: str
    answer: str
    answer_raw: str                 # critic 前的原稿
    critique: dict
    critic_retry: int

    # ---------- 目标验收 ----------
    goal_spec: dict
    goal_result: dict
    goal_retry: int

    # ---------- 输出 ----------
    images: List[dict]
    # 事实层三路（路线 / 天气 / 预算）并行，其中两路会写它，同样需要 reducer。
    # OR 的语义也正确：任一环节降级，整条链路即降级，且不会被后续节点改回去。
    degraded: Annotated[bool, operator.or_]
    error: str
    debug: Dict[str, Any]


QUERY_DEFAULT_STATE: QueryGraphState = {
    "session_id": "",
    "task_id": "",
    "user_id": "",
    "trace_id": "",
    "original_query": "",
    "is_stream": False,
    "history": [],
    "guard_result": {},
    "rewritten_query": "",
    "city": "",
    "cities": [],
    "content_type": "综合",
    "entities": [],
    "intent": "",
    "tool_flags": {"route": False, "weather": False, "budget": False},
    "need_clarify": False,
    "clarify_question": "",
    "slots": {},
    "profile": {},
    "dense_chunks": [],
    "sparse_chunks": [],
    "hyde_chunks": [],
    "web_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "retrieval_degraded": False,
    "evidence_types": [],
    "evidence_coverage": 0,
    "missing_types": [],
    "sub_queries": [],
    "sub_query_used": False,
    "route_facts": {},
    "weather_facts": {},
    "budget_facts": {},
    "facts": {},
    "context": "",
    "citations": [],
    "prompt": "",
    "answer": "",
    "answer_raw": "",
    "critique": {},
    "critic_retry": 0,
    "goal_spec": {},
    "goal_result": {},
    "goal_retry": 0,
    "images": [],
    "degraded": False,
    "error": "",
    "debug": {},
}


def create_query_state(**overrides) -> QueryGraphState:
    """建状态。深拷贝默认值，避免多个请求共享同一个 list。"""
    state = copy.deepcopy(QUERY_DEFAULT_STATE)
    state.update(overrides)
    return state


def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state


def routing_gate(state: QueryGraphState) -> str:
    """入口分支的唯一判定处，route 函数与节点都调它，避免两处判断不一致。"""
    guard = state.get("guard_result") or {}
    if guard and not guard.get("allowed", True):
        return "reject"
    if state.get("need_clarify"):
        return "clarify"
    return "search"
