"""行程图状态。

与查询图分开是刻意的：行程的中间产物（按天候选、片区簇、路段）结构完全不同，
塞进 QueryGraphState 会让两边都变得难读。

`validate_errors` 是**结构化的**（每条带 rule / detail / day），不是一句自然语言：
只有结构化才能被 `node_plan_validate` 回退时精准定位到某一天。
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, TypedDict


class PlanGraphState(TypedDict, total=False):
    # ---------- 标识 ----------
    session_id: str
    task_id: str
    trace_id: str
    plan_id: str

    # ---------- 输入约束 ----------
    raw_query: str
    city: str
    days: int
    start_date: str
    budget: float
    travelers: int
    pace: str
    companions: List[str]
    transport: str
    must_go: List[str]
    exclude: List[str]
    constraints: Dict[str, Any]

    # ---------- 检索与排布 ----------
    daily_candidates: Dict[int, List[dict]]
    candidate_pool: List[dict]
    clustered_days: List[dict]      # [{day, area, members[]}]
    route_legs: List[dict]          # [{from, to, distance_m, duration_s, mode, estimated}]
    weather_days: List[dict]
    daily_time_budget: List[dict]   # [{day, total_min, items[]}]

    # ---------- 预算 ----------
    budget_estimate: Dict[str, Any]

    # ---------- 生成与校验 ----------
    plan_json: Dict[str, Any]
    validate_errors: List[dict]
    validate_retry: int
    markdown: str

    # ---------- 输出 ----------
    degraded: bool
    error: str
    debug: Dict[str, Any]


PLAN_DEFAULT_STATE: PlanGraphState = {
    "session_id": "",
    "task_id": "",
    "trace_id": "",
    "plan_id": "",
    "raw_query": "",
    "city": "",
    "days": 3,
    "start_date": "",
    "budget": 0.0,
    "travelers": 2,
    "pace": "适中",
    "companions": [],
    "transport": "",
    "must_go": [],
    "exclude": [],
    "constraints": {},
    "daily_candidates": {},
    "candidate_pool": [],
    "clustered_days": [],
    "route_legs": [],
    "weather_days": [],
    "daily_time_budget": [],
    "budget_estimate": {},
    "plan_json": {},
    "validate_errors": [],
    "validate_retry": 0,
    "markdown": "",
    "degraded": False,
    "error": "",
    "debug": {},
}


def create_plan_state(**overrides) -> PlanGraphState:
    state = copy.deepcopy(PLAN_DEFAULT_STATE)
    state.update(overrides)
    return state


def copy_plan_state(state: PlanGraphState, **overrides) -> PlanGraphState:
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state


def has_blocking_errors(state: PlanGraphState) -> bool:
    """只有「硬约束」违反才回退重排；软提醒不影响出稿。"""
    blocking = {"cross_area", "closed_day", "overtime", "budget_over"}
    return any((err.get("rule") in blocking) for err in (state.get("validate_errors") or []))
