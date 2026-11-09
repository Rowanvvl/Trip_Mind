"""向量生成：一次前向出 dense + sparse。

批处理而非逐条：BGE-M3 在 CPU 上单条编码的启动开销远大于批处理差异，
70 篇语料约 400~600 个切片，批处理能把导入时间从十几分钟压到一两分钟。
"""
from __future__ import annotations

from typing import List

from common.config import milvus_config
from common.logging.logger import logger, node_log
from processor.import_processor.state import ImportGraphState
from utils.lm import embedding_utils


@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> dict:
    """给每个切片写 dense_vector / sparse_vector。

    输出：`chunks`（已带向量）、`embeddings_ok`、`embedding_backend`、`stage`、`stats.embedding`
    """
    chunks: List[dict] = [dict(c) for c in (state.get("chunks") or [])]
    if not chunks:
        return {"error": "无切片可编码", "final_status": "failed", "stage": "bge_embedding"}

    texts = [c.get("text", "") for c in chunks]
    batch = embedding_utils.encode(texts)

    if len(batch.dense) != len(chunks):
        return {
            "error": f"向量数量与切片数量不一致：{len(batch.dense)} vs {len(chunks)}",
            "final_status": "failed",
            "stage": "bge_embedding",
        }

    if batch.degraded:
        logger.warning(f"[bge_embedding] 使用降级向量后端：{batch.backend}")

    dim_mismatch = 0
    for chunk, dense, sparse in zip(chunks, batch.dense, batch.sparse):
        if len(dense) != milvus_config.dense_dim:
            dim_mismatch += 1
        chunk[milvus_config.dense_field] = dense
        chunk[milvus_config.sparse_field] = sparse

    if dim_mismatch:
        return {
            "error": f"向量维度与 Milvus schema 不一致（{len(batch.dense[0])} != {milvus_config.dense_dim}），"
                     f"请调整 BGE_DENSE_DIM 或重建集合",
            "final_status": "failed",
            "stage": "bge_embedding",
        }

    return {
        "chunks": chunks,
        "embeddings_ok": True,
        "embedding_backend": batch.backend,
        "stage": "bge_embedding",
        "stats": {
            **(state.get("stats") or {}),
            "embedding": {
                "count": len(chunks),
                "backend": batch.backend,
                "degraded": batch.degraded,
                "sparse_avg_terms": round(sum(len(s) for s in batch.sparse) / max(len(batch.sparse), 1), 1),
            },
        },
    }
