"""RRF 融合。

公式 `Σ 1/(k + rank)`，只看排名不看分数——这正是它的价值：
dense 的余弦相似度与 sparse 的 IP 分数量纲不同，直接加权需要调参且不稳定；
按排名融合不需要任何校准，且对某一路偶发的高分噪声免疫。

k=60 是原论文的经验值：k 越大越平缓，靠前的名次优势越小。
本项目保留 `RRF_K` 可配，但默认不要动——调它几乎没有收益。
"""
from __future__ import annotations

from typing import Dict, List

from common.config import trip_config
from common.logging.logger import node_log
from processor.query_processor.nodes._retrieval import dedup_chunks, preview
from processor.query_processor.state import QueryGraphState
from utils.trace_utils import record_rerank


@node_log("node_rrf")
def node_rrf(state: QueryGraphState) -> dict:
    """多路召回融合。

    输出：`rrf_chunks`
    """
    sources = {
        "dense": state.get("dense_chunks") or [],
        "sparse": state.get("sparse_chunks") or [],
        "hyde": state.get("hyde_chunks") or [],
        "web": state.get("web_docs") or [],
        # 子问题补检索的结果会追加到这里，二次融合时一并参与
        "previous": state.get("rrf_chunks") or [] if state.get("sub_query_used") else [],
    }

    k = trip_config.rrf_k
    scores: Dict[str, float] = {}
    registry: Dict[str, dict] = {}
    hit_sources: Dict[str, List[str]] = {}

    for name, docs in sources.items():
        seen_in_this_source = set()
        for rank, doc in enumerate(docs, start=1):
            key = doc.get("chunk_id") or doc.get("text", "")[:64]
            if not key or key in seen_in_this_source:
                continue
            seen_in_this_source.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            registry.setdefault(key, doc)
            hit_sources.setdefault(key, []).append(name)

    merged: List[dict] = []
    for key, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        doc = dict(registry[key])
        doc["rrf_score"] = round(score, 6)
        doc["hit_sources"] = hit_sources.get(key, [])
        merged.append(doc)

    merged = dedup_chunks(merged)

    record_rerank(
        state.get("trace_id", ""),
        rrf={
            "inputs": {name: len(docs) for name, docs in sources.items()},
            "merged": len(merged),
            "top": preview(merged),
        },
    )

    return {"rrf_chunks": merged}
