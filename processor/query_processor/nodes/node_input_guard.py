"""输入护栏节点。

放在图的最前面，**先于问题理解**：域外问题连改写和检索都不用付钱。
规则判定（毫秒级）为主，只有「过短 / 拿不准」时才请模型复核一道。
"""
from __future__ import annotations

from common.config import lm_config, trip_config
from common.logging.logger import logger, node_log
from processor.query_processor.state import QueryGraphState
from utils.guard.input_guard import check_input
from utils.load_prompt import load_prompt, render_safe
from utils.lm.lm_utils import chat_json
from utils.trace_utils import record_guard, record_route


@node_log("node_input_guard")
def node_input_guard(state: QueryGraphState) -> dict:
    """三层判定：规则 → （必要时）模型复核。

    输出：`guard_result`、`original_query`（已清理）
    """
    query = state.get("original_query") or ""
    trace_id = state.get("trace_id", "")

    if not trip_config.guard_enabled:
        return {"guard_result": {"allowed": True, "action": "pass", "sanitized_query": query}}

    result = check_input(query)
    record_guard(trace_id, "input", result.action, result.reason)

    # 规则判不了且模型可用时，让模型判一次范围（只判范围，不回答问题）
    if result.need_llm_check and lm_config.enabled:
        prompt = render_safe("guard_input", "", query=result.sanitized_query)
        data, llm_result = chat_json(prompt, tier="light", fallback={"in_scope": True})
        if llm_result.ok and data.get("in_scope") is False:
            category = data.get("category", "other")
            result.allowed = False
            result.action = "reject"
            result.category = category
            result.reply = data.get("reply") or result.reply
            record_guard(trace_id, "input_llm", "reject", category)
        else:
            record_guard(trace_id, "input_llm", "pass", "")

    if not result.allowed:
        logger.info(f"[input_guard] 拦截 category={result.category} reason={result.reason}")

    record_route(trace_id, guard_action=result.action, guard_category=result.category)

    return {
        "guard_result": result.to_dict(),
        "original_query": result.sanitized_query or query,
    }


def after_input_guard(state: QueryGraphState) -> str:
    """拦截则走拒答，否则进入问题理解。"""
    guard = state.get("guard_result") or {}
    return "reject" if not guard.get("allowed", True) else "understand"
