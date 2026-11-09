"""精排。

RRF 之后的 Top-20 里经常混着"同城不同事"的切片。重排把 20 压到 6，
这 6 条才是真正进 prompt 的内容。

阈值行为需要说明：全部低于阈值时**保留最高分那条**而不是返回空。
返回空会让下游说「知识库未收录」，但实际是有相关内容的，只是分数低。
把「证据不足」的判断权交给 `node_retrieve_assess`，而不是在这里丢掉。
"""
from __future__ import annotations

from common.config import trip_config
from common.logging.logger import node_log
from processor.query_processor.nodes._retrieval import preview
from processor.query_processor.state import QueryGraphState
from utils.lm.reranker_utils import rerank
from utils.trace_utils import record_rerank


@node_log("node_rerank")
def node_rerank(state: QueryGraphState) -> dict:
    """重排并截断到 RERANK_TOP_K。

    输出：`reranked_docs`
    """
    query = state.get("rewritten_query") or state.get("original_query") or ""
    candidates = state.get("rrf_chunks") or []

    if not candidates:
        return {"reranked_docs": []}

    docs = rerank(query, candidates, top_k=trip_config.rerank_top_k)

    record_rerank(
        state.get("trace_id", ""),
        rerank={
            "in": len(candidates),
            "out": len(docs),
            "backend": docs[0].get("rerank_backend") if docs else "",
            "scores": [d.get("rerank_score") for d in docs],
            "top": preview(docs),
        },
    )

    return {"reranked_docs": docs}
