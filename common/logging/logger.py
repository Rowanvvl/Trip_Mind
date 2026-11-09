"""日志工具。

基于 loguru：控制台 + 按天切割的文件双输出，贯穿全项目。
对外暴露两个装饰器：

- `node_log(node_name)`：LangGraph 节点专用。除打印耗时外，还会
  1) 把节点名推给 SSE（`TASK_STATUS_PROCESSING` 用），前端进度条依赖它；
  2) 若 state 里有 `trace_id`，把该节点耗时记进 trace。
- `step_log(step_name)`：普通步骤耗时，不碰任务状态。
"""
from __future__ import annotations

import inspect
import os
import sys
import time
from functools import wraps
from typing import Any, Callable, Mapping

from loguru import logger as _base_logger

from utils.path_util import LOG_DIR

LOG_CONSOLE_ENABLE = os.getenv("LOG_CONSOLE_ENABLE", "true").lower() == "true"
LOG_CONSOLE_LEVEL = os.getenv("LOG_CONSOLE_LEVEL", "INFO").upper()
LOG_FILE_ENABLE = os.getenv("LOG_FILE_ENABLE", "true").lower() == "true"
LOG_FILE_LEVEL = os.getenv("LOG_FILE_LEVEL", "INFO").upper()
LOG_FILE_RETENTION = os.getenv("LOG_FILE_RETENTION", "7 days")

LOG_FILE_PATH = LOG_DIR / "app_{time:YYYYMMDD}.log"

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name: <24}</cyan>:<cyan>{function: <18}</cyan>:<cyan>{line: <4}</cyan> - "
    "<level>{message}</level>"
)


def init_logger():
    """初始化全局日志：移除默认 handler，按配置挂控制台与文件。"""
    _base_logger.remove()

    if LOG_CONSOLE_ENABLE:
        _base_logger.add(
            sink=sys.stdout,
            level=LOG_CONSOLE_LEVEL,
            format=LOG_FORMAT,
            colorize=True,
            enqueue=True,
        )

    if LOG_FILE_ENABLE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        _base_logger.add(
            sink=str(LOG_FILE_PATH),
            level=LOG_FILE_LEVEL,
            format=LOG_FORMAT,
            rotation="00:00",
            retention=LOG_FILE_RETENTION,
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=False,
        )
    return _base_logger


def _fix_log_position(record: dict) -> None:
    """把日志的 name/function/line 指向业务代码真实调用位置。"""
    for frame in inspect.stack():
        filename = frame.filename
        if "_logger.py" in filename or filename.endswith("logger.py"):
            continue
        record.update(
            name=os.path.basename(filename),
            function=frame.function,
            line=frame.lineno,
        )
        break


logger = init_logger().patch(_fix_log_position)


def _trace_key(state: Mapping[str, Any] | None) -> str:
    if isinstance(state, Mapping):
        return str(state.get("trace_id") or state.get("session_id") or state.get("task_id") or "-")
    return "-"


def node_log(node_name: str) -> Callable:
    """节点级日志：耗时 + 任务进度 + trace 记时。"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            trace_id = _trace_key(state)
            started = time.perf_counter()
            logger.info(f"[{node_name}] 节点开始 trace={trace_id}")

            # 延迟导入，避免 logger 与 task/sse 之间形成导入环
            try:
                from utils.task_utils import push_node_running

                task_id = state.get("task_id") if isinstance(state, Mapping) else None
                if task_id:
                    push_node_running(task_id, node_name)
            except Exception:  # 进度推送失败不能影响主流程
                logger.debug(f"[{node_name}] 进度推送跳过")

            try:
                result = func(state, *args, **kwargs)
            except Exception:
                logger.exception(f"[{node_name}] 节点异常 trace={trace_id}")
                raise

            cost_ms = int((time.perf_counter() - started) * 1000)
            logger.info(f"[{node_name}] 节点完成 trace={trace_id} 耗时={cost_ms}ms")

            try:
                from utils.task_utils import push_node_done

                task_id = state.get("task_id") if isinstance(state, Mapping) else None
                if task_id:
                    push_node_done(task_id, node_name)
            except Exception:
                logger.debug(f"[{node_name}] 完成事件推送跳过")

            try:
                from utils.trace_utils import record_node_latency

                trace_id_value = state.get("trace_id") if isinstance(state, Mapping) else None
                if trace_id_value:
                    record_node_latency(trace_id_value, node_name, cost_ms)
            except Exception:
                logger.debug(f"[{node_name}] trace 记时跳过")

            return result

        return wrapper

    return decorator


def step_log(step_name: str) -> Callable:
    """普通步骤耗时日志，不涉及任务状态。"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            logger.info(f"[{step_name}] 步骤开始")
            try:
                result = func(*args, **kwargs)
            except Exception:
                logger.exception(f"[{step_name}] 步骤异常")
                raise
            cost_ms = int((time.perf_counter() - started) * 1000)
            logger.info(f"[{step_name}] 步骤完成 耗时={cost_ms}ms")
            return result

        return wrapper

    return decorator
