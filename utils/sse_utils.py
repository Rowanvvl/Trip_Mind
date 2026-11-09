"""SSE 推送。

一条连接 = 一个 session 队列。事件类型固定，前端按 event 名分派：

| event | 时机 | data |
|---|---|---|
| `ready` | 连接建立 | `{}` |
| `progress` | 图节点开始/结束 | `{node, label, percent, status}` |
| `delta` | 生成流式增量 | `{text}` |
| `citations` | 引用确定后 | `{items: [...]}` |
| `facts` | 外部事实汇总 | `{route, weather, budget, degraded}` |
| `images` | 答案结束后挂图 | `{items: [...]}` |
| `final` | 全部完成 | `{answer, ...}` |
| `error` | 异常 | `{message}` |
| `__close__` | 关闭信号 | `{}` |

设计取舍：**facts / images / citations 是独立事件，不塞进 final**。
前端可以先渲染答案，再补卡片；图片失败也不会拖住首字。
"""
from __future__ import annotations

import asyncio
import json
import queue
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import Request

from common.logging.logger import logger


class SSEEvent:
    READY = "ready"
    PROGRESS = "progress"
    DELTA = "delta"
    CITATIONS = "citations"
    FACTS = "facts"
    IMAGES = "images"
    FINAL = "final"
    ERROR = "error"
    CLOSE = "__close__"


_session_stream: Dict[str, "queue.Queue"] = {}


def create_sse_queue(session_id: str) -> "queue.Queue":
    q: queue.Queue = queue.Queue()
    _session_stream[session_id] = q
    logger.debug(f"[SSE] 创建队列 session={session_id}")
    return q


def get_sse_queue(session_id: str) -> Optional["queue.Queue"]:
    return _session_stream.get(session_id)


def alias_queue(alias: str, session_id: str) -> bool:
    """让一个别名指向同一队列。

    用途：`node_log` 按 `task_id` 推节点进度，而前端连接按 `session_id` 建队列。
    不做别名会出现「答案能流式出，但进度条一直不动」这种很难查的现象。
    """
    stream_queue = get_sse_queue(session_id)
    if stream_queue is None or alias == session_id:
        return False
    _session_stream[alias] = stream_queue
    return True


def drop_alias(alias: str) -> None:
    """只删别名，不动真实队列（队列由 session 持有）。"""
    _session_stream.pop(alias, None)


def remove_sse_queue(session_id: str) -> None:
    _session_stream.pop(session_id, None)
    logger.debug(f"[SSE] 移除队列 session={session_id}")


def _pack(event: str, data: Dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def push_to_session(session_id: str, event: str, data: Dict[str, Any]) -> bool:
    """推送事件。返回是否成功（没有队列时返回 False，调用方不必处理）。"""
    stream_queue = get_sse_queue(session_id)
    if stream_queue is None:
        logger.debug(f"[SSE] 无队列，丢弃事件 {event} session={session_id}")
        return False
    stream_queue.put({"event": event, "data": data})
    return True


def push_delta(session_id: str, text: str) -> bool:
    return push_to_session(session_id, SSEEvent.DELTA, {"text": text})


def push_progress(session_id: str, node: str, label: str, percent: int, status: str = "processing") -> bool:
    return push_to_session(
        session_id,
        SSEEvent.PROGRESS,
        {"node": node, "label": label, "percent": percent, "status": status},
    )


def push_error(session_id: str, message: str, code: str = "internal_error") -> bool:
    return push_to_session(session_id, SSEEvent.ERROR, {"message": message, "code": code})


async def sse_generator(session_id: str, request: Request) -> AsyncGenerator[str, None]:
    """FastAPI StreamingResponse 用的生成器。"""
    stream_queue = get_sse_queue(session_id)
    if stream_queue is None:
        yield _pack(SSEEvent.ERROR, {"message": "会话队列不存在", "code": "no_queue"})
        return

    loop = asyncio.get_running_loop()
    yield _pack(SSEEvent.READY, {"session_id": session_id})
    try:
        while True:
            if await request.is_disconnected():
                logger.info(f"[SSE] 客户端断开 session={session_id}")
                break
            try:
                # 阻塞取队列放到线程池，避免卡住事件循环
                msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
            except queue.Empty:
                # 心跳，防止中间代理断连
                yield ": keep-alive\n\n"
                continue

            event = msg.get("event", SSEEvent.DELTA)
            data = msg.get("data", {})
            yield _pack(event, data)

            if event == SSEEvent.CLOSE:
                break
    finally:
        remove_sse_queue(session_id)


def close_session(session_id: str) -> None:
    """通知前端结束并清理队列。"""
    push_to_session(session_id, SSEEvent.CLOSE, {})
    remove_sse_queue(session_id)


def active_sessions() -> list[str]:
    return list(_session_stream.keys())
