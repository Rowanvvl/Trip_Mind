"""问答服务。

两条路径：
- `POST /query`（`is_stream=false`）：同步返回完整结果，适合脚本与评测；
- `POST /query`（`is_stream=true`）：建 SSE 队列 + 后台跑图，立刻返回 `session_id`，
  前端再连 `GET /stream/{session_id}`。**先建队列再返回**是必须的——
  否则前端连上来时队列还不存在，`sse_generator` 会直接报 no_queue。

结果落 Mongo（消息 + 槽位快照），trace 落 Mongo（`TRACE_ENABLED`），
所以「为什么这么回答」在 `/trace/{trace_id}` 里可以完整回看。
"""
from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from common.config import trip_config
from common.logging.logger import logger
from utils.format_utils import now_iso
from utils.guard.param_guard import ParamError, sanitize_text, validate_session_id, validate_user_id
from utils.memory.profile_utils import empty_profile
from utils.sse_utils import SSEEvent, close_session, create_sse_queue, push_to_session, sse_generator
from utils.task_utils import TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, create_task, set_task_error, set_task_status
from utils.trace_utils import finish_trace, start_trace

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    session_id: str = Field(..., description="会话 ID，前端生成")
    query: str = Field(..., description="用户提问")
    user_id: Optional[str] = None
    is_stream: bool = False


class QueryResponse(BaseModel):
    session_id: str
    trace_id: str
    task_id: str
    answer: str = ""
    citations: List[dict] = []
    facts: dict = {}
    images: List[dict] = []
    city: str = ""
    slots: dict = {}
    degraded: bool = False
    critique: dict = {}
    goal: dict = {}
    streaming: bool = False


class FeedbackRequest(BaseModel):
    trace_id: str
    session_id: str = ""
    helpful: bool = True
    comment: str = ""


def _build_state(session_id: str, query: str, user_id: str | None, is_stream: bool) -> dict:
    """组装查询图初始状态。历史与偏好在这里读一次，节点内部不再访问 Mongo。"""
    from processor.query_processor.state import create_query_state
    from utils.clients import mongo_history_utils

    trace_id = uuid.uuid4().hex[:16]
    task_id = uuid.uuid4().hex[:16]
    history = mongo_history_utils.get_history(session_id, trip_config.history_keep_turns)
    profile = mongo_history_utils.get_profile(user_id) if user_id else empty_profile().copy()

    return create_query_state(
        session_id=session_id,
        task_id=task_id,
        user_id=user_id or "",
        trace_id=trace_id,
        original_query=query,
        is_stream=is_stream,
        history=history,
        profile=profile or {},
    )


def _run_query(state: dict) -> dict:
    """跑一次查询图，并完成落库、事件推送、trace 收尾。

    这是唯一一处把「图执行」与「副作用」绑在一起的地方，
    同步与流式两条路径共用，避免两边行为不一致。
    """
    from processor.query_processor.main_graph import query_app
    from utils.clients import mongo_history_utils

    trace_id = state["trace_id"]
    task_id = state["task_id"]
    session_id = state["session_id"]

    start_trace(trace_id, session_id=session_id, query=state.get("original_query", ""))
    create_task(task_id, graph="query")

    # 节点进度按 task_id 推送，前端连接按 session_id，这里把两者指向同一个队列
    if state.get("is_stream"):
        from utils import sse_utils

        sse_utils.alias_queue(task_id, session_id)

    try:
        result = query_app.invoke(state)
    except Exception as exc:
        logger.exception("[query] 图执行异常")
        set_task_error(task_id, str(exc))
        finish_trace(trace_id, error=str(exc))
        if state.get("is_stream"):
            push_to_session(session_id, SSEEvent.ERROR, {"message": "服务异常，请稍后再试", "code": "graph_error"})
            close_session(session_id)
        raise

    answer = result.get("answer") or ""
    citations = result.get("citations") or []
    facts = result.get("facts") or {}

    # 消息落库（含槽位快照，便于回放"为什么这轮变聪明了"）
    mongo_history_utils.append_message(
        session_id=session_id,
        role="user",
        text=state.get("original_query", ""),
        city=result.get("city", ""),
        slots=result.get("slots") or {},
        trace_id=trace_id,
    )
    mongo_history_utils.append_message(
        session_id=session_id,
        role="assistant",
        text=answer,
        citations=citations,
        city=result.get("city", ""),
        slots=result.get("slots") or {},
        facts={"degraded": facts.get("degraded", False)},
        images=[img.get("url") for img in result.get("images") or []],
        trace_id=trace_id,
    )
    mongo_history_utils.upsert_session(
        session_id,
        city=result.get("city", ""),
        last_query=sanitize_text(state.get("original_query", ""), 120),
        updated_at=now_iso(),
    )

    set_task_status(task_id, TASK_STATUS_COMPLETED)
    finish_trace(trace_id)

    if state.get("is_stream"):
        push_to_session(session_id, SSEEvent.CITATIONS, {"items": citations})
        push_to_session(session_id, SSEEvent.FACTS, _facts_payload(facts))
        push_to_session(
            session_id,
            SSEEvent.FINAL,
            {
                "answer": answer,
                "trace_id": trace_id,
                "task_id": task_id,
                "city": result.get("city", ""),
                "degraded": bool(result.get("degraded")),
            },
        )
        close_session(session_id)
        from utils import sse_utils

        sse_utils.drop_alias(task_id)

    return result


def _facts_payload(facts: dict) -> dict:
    """SSE 只下发前端要展示的部分，不把整个 facts 里的大对象推过去。"""
    return {
        "route": (facts.get("route") or {}).get("summaries") or [],
        "weather": (facts.get("weather") or {}).get("summary") or "",
        "budget": (facts.get("budget") or {}).get("lines") or [],
        "degraded": bool(facts.get("degraded")),
    }


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, background: BackgroundTasks):
    """问答入口。流式时立刻返回 session_id，由 /stream/{session_id} 推送。"""
    try:
        session_id = validate_session_id(request.session_id)
        user_id = validate_user_id(request.user_id)
    except ParamError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    if request.is_stream:
        create_sse_queue(session_id)
        state = _build_state(session_id, request.query, user_id, True)
        background.add_task(_run_query, state)
        return QueryResponse(
            session_id=session_id,
            trace_id=state["trace_id"],
            task_id=state["task_id"],
            streaming=True,
        )

    state = _build_state(session_id, request.query, user_id, False)
    try:
        result = _run_query(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="处理失败，请稍后重试") from exc

    return QueryResponse(
        session_id=session_id,
        trace_id=state["trace_id"],
        task_id=state["task_id"],
        answer=result.get("answer") or "",
        citations=result.get("citations") or [],
        facts=result.get("facts") or {},
        images=result.get("images") or [],
        city=result.get("city") or "",
        slots=result.get("slots") or {},
        degraded=bool(result.get("degraded")),
        critique=result.get("critique") or {},
        goal=result.get("goal_result") or {},
    )


@router.get("/stream/{session_id}")
async def stream(session_id: str, request: Request) -> StreamingResponse:
    """SSE 事件流。事件类型见 `utils/sse_utils.SSEEvent`。"""
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/history/{session_id}")
async def history(session_id: str, limit: int = Query(default=20, ge=1, le=100)) -> dict:
    """历史消息（时间正序）。"""
    from utils.clients import mongo_history_utils

    return {"session_id": session_id, "messages": mongo_history_utils.get_history(session_id, limit)}


@router.delete("/history/{session_id}")
async def clear_history(session_id: str) -> dict:
    from utils.clients import mongo_history_utils

    deleted = mongo_history_utils.clear_history(session_id)
    return {"session_id": session_id, "deleted": deleted}


@router.get("/trace/{trace_id}")
async def trace(trace_id: str) -> dict:
    """请求轨迹：路由判定、召回、工具、护栏、critic、token。"""
    from utils.clients import mongo_history_utils
    from utils.trace_utils import get_trace as get_memory_trace

    memory_trace = get_memory_trace(trace_id)
    if memory_trace:
        return memory_trace.to_dict()
    stored = mongo_history_utils.get_trace(trace_id)
    if not stored:
        raise HTTPException(status_code=404, detail="trace 不存在")
    return stored


@router.post("/feedback")
async def feedback(request: FeedbackRequest) -> dict:
    """用户反馈。这是评测集最便宜的来源：一条差评就是一条待分析样本。"""
    from utils.clients import mongo_history_utils

    ok = mongo_history_utils.save_feedback(
        {
            "trace_id": request.trace_id,
            "session_id": request.session_id,
            "helpful": request.helpful,
            "comment": sanitize_text(request.comment, 500),
        }
    )
    return {"ok": ok}
