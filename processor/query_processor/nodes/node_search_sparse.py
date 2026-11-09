"""关键词召回（sparse）。

这一路存在的唯一理由是**专名**：蜈支洲岛、陕历博、袁家界、曾厝垵。
这些词在语义空间里与「海岛」「博物馆」几乎等价，但用户问的就是这一个，
dense 会把「西岛」「厦门博物馆」一起召回来，sparse 不会。
"""
from __future__ import annotations

from common.logging.logger import node_log
from processor.query_processor.nodes._retrieval import normalize_hits, preview, search_with_relax
from processor.query_processor.state import QueryGraphState
from utils.clients import milvus_utils
from utils.lm import embedding_utils
from utils.trace_utils import record_retrieval


@node_log("node_search_sparse")
def node_search_sparse(state: QueryGraphState) -> dict:
    """稀疏召回。

    输出：`sparse_chunks`
    """
    query = state.get("rewritten_query") or state.get("original_query") or ""
    city = state.get("city")
    content_type = state.get("content_type")

    _, sparse, degraded = embedding_utils.encode_query(query)
    if not sparse:
        return {"sparse_chunks": []}

    def _search(expr: str, limit: int):
        return milvus_utils.search_sparse(sparse, expr, limit)

    result = search_with_relax(_search, city, content_type, label="sparse")
    chunks = normalize_hits(result["docs"], "sparse")

    record_retrieval(
        state.get("trace_id", ""),
        sparse={"count": len(chunks), "level": result["level"], "top": preview(chunks)},
    )

    return {
        "sparse_chunks": chunks,
        "retrieval_degraded": bool(state.get("retrieval_degraded")) or degraded,
    }
