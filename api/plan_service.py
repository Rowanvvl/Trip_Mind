"""行程服务。

与问答分开成独立服务的原因：行程是**重操作**（检索 + 聚类 + 生成 + 校验 + 可能的回退重排），
耗时通常是问答的 3~5 倍。混在 `/query` 里会让问答的超时配置失去意义。

`POST /plan` 支持同步返回；需要看进度时前端轮询 `/import/status/{task_id}`
（沿用同一套任务状态机，graph="plan"）。
"""
from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from common.config import trip_config
from common.logging.logger import logger
from utils.guard.param_guard import ParamError, validate_budget, validate_city, validate_days, validate_pace
from utils.task_utils import TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, create_task, set_task_error, set_task_status
from utils.trace_utils import finish_trace, start_trace

router = APIRouter(prefix="/plan", tags=["plan"])


class PlanRequest(BaseModel):
    session_id: str
    query: str = Field(default="", description="自然语言需求，可留空")
    city: Optional[str] = None
    days: int = 3
    start_date: Optional[str] = None
    budget: Optional[float] = None
    travelers: int = 2
    pace: Optional[str] = None
    must_go: List[str] = []
    exclude: List[str] = []


class PlanResponse(BaseModel):
    plan_id: str
    city: str
    days: int
    markdown: str = ""
    plan_json: dict = {}
    budget: dict = {}
    warnings: List[dict] = []
    errors: List[dict] = []
    degraded: bool = False


@router.post("", response_model=PlanResponse)
async def create_plan(request: PlanRequest) -> PlanResponse:
    """生成行程。"""
    from processor.plan_processor.main_graph import plan_app
    from processor.plan_processor.state import create_plan_state

    try:
        city = validate_city(request.city, allow_none=True)
        days = validate_days(request.days, upper=trip_config.plan_max_days)
        budget = validate_budget(request.budget) or 0.0
        pace = validate_pace(request.pace)
    except ParamError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not city and not request.query:
        raise HTTPException(status_code=400, detail="需要城市或自然语言描述")

    trace_id = uuid.uuid4().hex[:16]
    task_id = uuid.uuid4().hex[:16]
    plan_id = f"plan_{uuid.uuid4().hex[:12]}"

    state = create_plan_state(
        session_id=request.session_id,
        task_id=task_id,
        trace_id=trace_id,
        plan_id=plan_id,
        raw_query=request.query,
        city=city or "",
        days=days,
        start_date=request.start_date or "",
        budget=budget,
        travelers=request.travelers,
        pace=pace,
        must_go=request.must_go,
        exclude=request.exclude,
    )

    start_trace(trace_id, session_id=request.session_id, query=request.query or f"{city}{days}天")
    create_task(task_id, graph="plan")

    try:
        result = plan_app.invoke(state)
    except Exception as exc:
        logger.exception("[plan] 生成失败")
        set_task_error(task_id, str(exc))
        finish_trace(trace_id, error=str(exc))
        raise HTTPException(status_code=500, detail="行程生成失败，请稍后重试") from exc

    if result.get("error"):
        set_task_error(task_id, result["error"])
        raise HTTPException(status_code=400, detail=result["error"])

    set_task_status(task_id, TASK_STATUS_COMPLETED)
    finish_trace(trace_id)

    # 落库（失败不影响返回）
    try:
        from utils.clients import mongo_history_utils

        mongo_history_utils.save_itinerary(
            {
                "plan_id": result.get("plan_id") or plan_id,
                "session_id": request.session_id,
                "city": result.get("city"),
                "days": result.get("days"),
                "plan_json": result.get("plan_json"),
                "markdown": result.get("markdown"),
                "budget_estimate": result.get("budget_estimate"),
                "trace_id": trace_id,
            }
        )
    except Exception:
        logger.debug("[plan] 行程落库跳过")

    debug = result.get("debug") or {}
    return PlanResponse(
        plan_id=result.get("plan_id") or plan_id,
        city=result.get("city") or (city or ""),
        days=int(result.get("days") or days),
        markdown=result.get("markdown") or "",
        plan_json=result.get("plan_json") or {},
        budget=result.get("budget_estimate") or {},
        warnings=(debug.get("validate") or {}).get("warnings") or [],
        errors=result.get("validate_errors") or [],
        degraded=bool(result.get("degraded")),
    )


@router.get("/{plan_id}")
async def get_plan(plan_id: str) -> dict:
    """按 plan_id 取历史行程。"""
    from utils.clients import mongo_history_utils

    plan = mongo_history_utils.get_itinerary(plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="行程不存在")
    return plan
