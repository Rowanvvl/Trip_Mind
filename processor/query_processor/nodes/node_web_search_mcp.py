"""网络检索（兜底）。

用途只有一个：知识库不可能收录的**时变信息**——临时闭馆、票价调整、突发天气。
语料能回答的问题不应该走这里，因为网络结果没有元数据、无法过滤城市、质量不可控。

当前实现是显式的空实现：没有配置搜索 provider 时返回空并标记降级，
**不假装搜到了**。接入时只要实现 `_web_search()`，图不用改。
"""
from __future__ import annotations

from typing import List

from common.config import trip_config
from common.logging.logger import logger, node_log
from processor.query_processor.state import QueryGraphState
from utils.tools.tool_registry import call
from utils.trace_utils import record_retrieval, record_tool


@node_log("node_web_search_mcp")
def node_web_search_mcp(state: QueryGraphState) -> dict:
    """网络兜底检索。

    输出：`web_docs`、`retrieval_degraded`
    """
    query = state.get("rewritten_query") or state.get("original_query") or ""

    # 只在明确需要时变信息时才打网络，省钱也省延迟
    if not (state.get("tool_flags") or {}).get("weather"):
        return {"web_docs": []}

    result = call("web_search", _web_search, query)
    record_tool(state.get("trace_id", ""), "web_search", result.ok, result.cost_ms, error=result.error)

    docs = result.data if result.ok else []
    record_retrieval(state.get("trace_id", ""), web={"count": len(docs), "ok": result.ok})

    return {
        "web_docs": docs or [],
        "retrieval_degraded": bool(state.get("retrieval_degraded")) or not result.ok,
    }


def _web_search(query: str) -> List[dict] | None:
    """接入点。

    TODO(M3)：接一个搜索 API（或自建 MCP server），返回统一结构：
    `[{"title": "", "url": "", "snippet": "", "published_at": ""}]`
    接好后 `node_context_build` 会把它们当作普通资料处理，编号连续。
    """
    logger.debug(f"[web_search] 未配置搜索 provider，跳过：{query}")
    return None
