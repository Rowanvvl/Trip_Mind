"""语义召回（dense）。

带 city / content_type 标量过滤，空结果自动放宽。这一路的强项是
「带老人适合去哪」这种没有专名的问法；弱项是专名精确匹配（交给 sparse）。
"""
from __future__ import annotations

from common.config import trip_config
from common.logging.logger import node_log
from processor.query_processor.nodes._retrieval import normalize_hits, preview, search_with_relax
from processor.query_processor.state import QueryGraphState
from utils.clients import milvus_utils
from utils.lm import embedding_utils
from utils.trace_utils import record_retrieval


@node_log("node_search_dense")
def node_search_dense(state: QueryGraphState) -> dict:
    """向量召回。

    输出：`dense_chunks`、`retrieval_degraded`
    """
    query = state.get("rewritten_query") or state.get("original_query") or ""
    city = state.get("city")
    content_type = state.get("content_type")

    dense, _, degraded = embedding_utils.encode_query(query)
    if not dense:
        return {"dense_chunks": [], "retrieval_degraded": True}

    def _search(expr: str, limit: int):
        return milvus_utils.search_dense(dense, expr, limit)

    result = search_with_relax(_search, city, content_type, label="dense")
    chunks = normalize_hits(result["docs"], "dense")

    record_retrieval(
        state.get("trace_id", ""),
        dense={"count": len(chunks), "level": result["level"], "top": preview(chunks)},
    )

    return {
        "dense_chunks": chunks,
        "retrieval_degraded": bool(state.get("retrieval_degraded")) or degraded,
    }
