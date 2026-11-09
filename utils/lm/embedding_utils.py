"""向量编码：BGE-M3 一次前向同时产出 dense 与 sparse。

三级降级（按可用性自动选择，**始终如实标注用了哪一级**）：
1. 本地 BGE-M3（`FlagEmbedding`）——正式方案；
2. OpenAI 兼容 embedding 接口（只有 dense，sparse 用 jieba 补）——临时方案；
3. 字符 bigram 哈希向量（dense）+ 词频（sparse）——**仅用于骨架跑通与单测**，
   语义能力很弱，`degraded=True`，答案里不能当作正式检索质量。

为什么要留第 3 级：没有模型权重时如果直接抛异常，整条链路连「跑一遍看看」都做不到；
留一个弱兜底，至少能验证切分、过滤、融合、展示这些环节。
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

from common.config import embedding_config
from common.logging.logger import logger
from utils.normalize_sparse_vector import from_bge_output, from_text

_model = None
_model_failed = False


@dataclass
class EmbeddingBatch:
    dense: List[List[float]]
    sparse: List[Dict[int, float]]
    degraded: bool = False
    backend: str = "bge-m3"


def _load_model():
    """懒加载 BGE-M3。加载失败只记一次日志，之后的调用直接走降级。"""
    global _model, _model_failed
    if _model is not None or _model_failed:
        return _model
    try:
        from FlagEmbedding import BGEM3FlagModel

        _model = BGEM3FlagModel(
            embedding_config.model_path,
            use_fp16=embedding_config.use_fp16,
            device=embedding_config.device,
        )
        logger.info(f"[embedding] BGE-M3 加载完成：{embedding_config.model_path}")
    except Exception as exc:
        _model_failed = True
        logger.warning(f"[embedding] BGE-M3 不可用（{exc}），降级为哈希向量")
    return _model


def available() -> bool:
    return _load_model() is not None


def _hash_dense(text: str, dim: int | None = None) -> List[float]:
    """字符 bigram 哈希向量。只为让链路跑通，不代表语义能力。"""
    dim = dim or embedding_config.dense_dim
    vec = [0.0] * dim
    clean = (text or "").strip()
    if not clean:
        return vec
    grams = [clean[i : i + 2] for i in range(max(len(clean) - 1, 1))]
    counts: Dict[int, float] = {}
    for gram in grams:
        bucket = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16) % dim
        counts[bucket] = counts.get(bucket, 0.0) + 1.0
    for bucket, count in counts.items():
        vec[bucket] = 1.0 + math.log(count)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


def _api_dense(texts: Sequence[str]) -> List[List[float]] | None:
    """OpenAI 兼容 embedding 接口，只有 dense。"""
    try:
        from utils.lm.lm_utils import get_client

        client = get_client()
        if client is None:
            return None
        resp = client.embeddings.create(model=embedding_config.api_model, input=list(texts))
        return [item.embedding for item in resp.data]
    except Exception as exc:
        logger.debug(f"[embedding] 接口向量不可用：{exc}")
        return None


def encode(texts: Sequence[str], batch_size: int | None = None) -> EmbeddingBatch:
    """批量编码。返回的 `degraded` 表示是否用了弱兜底。"""
    items = [t or "" for t in texts]
    if not items:
        return EmbeddingBatch(dense=[], sparse=[], backend="empty")

    model = _load_model()
    if model is not None:
        batch_size = batch_size or embedding_config.batch_size
        dense_all: List[List[float]] = []
        sparse_all: List[Dict[int, float]] = []
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            output = model.encode(batch, max_length=embedding_config.max_length, return_dense=True, return_sparse=True)
            dense_all.extend([list(map(float, v)) for v in output["dense_vecs"]])
            for weights in output["lexical_weights"]:
                sparse_all.append(from_bge_output({int(k): float(v) for k, v in weights.items()}))
        return EmbeddingBatch(dense=dense_all, sparse=sparse_all, backend="bge-m3")

    api_vectors = _api_dense(items)
    if api_vectors:
        sparse_all = [from_text(t) for t in items]
        return EmbeddingBatch(dense=api_vectors, sparse=sparse_all, degraded=True, backend="api-embedding")

    return EmbeddingBatch(
        dense=[_hash_dense(t) for t in items],
        sparse=[from_text(t) for t in items],
        degraded=True,
        backend="hash-fallback",
    )


def encode_one(text: str) -> EmbeddingBatch:
    return encode([text])


def encode_query(text: str) -> tuple[List[float], Dict[int, float], bool]:
    """查询侧常用形态：`(dense, sparse, degraded)`。"""
    batch = encode([text])
    dense = batch.dense[0] if batch.dense else []
    sparse = batch.sparse[0] if batch.sparse else {}
    return dense, sparse, batch.degraded


def health() -> dict:
    return {
        "model_path": embedding_config.model_path,
        "loaded": _model is not None,
        "load_failed": _model_failed,
        "dense_dim": embedding_config.dense_dim,
        "device": embedding_config.device,
    }
