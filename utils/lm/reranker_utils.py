"""重排：bge-reranker 精排。

重排不是"可选优化"，而是把 20 条召回压到 6 条的唯一可靠手段：
RRF 只看排名，不看相关性高低；语义召回的 Top-1 经常是"提了同一座城市但答非所问"的切片。

无模型时的降级：词项覆盖率打分（query 的实词在切片中出现的比例 + 专名加权）。
它比向量弱，但**排序方向是对的**，且完全可解释，便于排查。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from common.config import reranker_config, trip_config
from common.logging.logger import logger
from utils.normalize_sparse_vector import tokenize

_model = None
_model_failed = False


def _load_model():
    global _model, _model_failed
    if _model is not None or _model_failed:
        return _model
    try:
        from FlagEmbedding import FlagReranker

        _model = FlagReranker(
            reranker_config.model_path,
            use_fp16=reranker_config.use_fp16,
            device=reranker_config.device,
        )
        logger.info(f"[rerank] 模型加载完成：{reranker_config.model_path}")
    except Exception as exc:
        _model_failed = True
        logger.warning(f"[rerank] 模型不可用（{exc}），降级为词项覆盖打分")
    return _model


def available() -> bool:
    return _load_model() is not None


def _lexical_score(query: str, text: str) -> float:
    """词项覆盖率：query 的实词有多少出现在 text 里，长词加权。"""
    q_tokens = [t for t in tokenize(query) if len(t) >= 2]
    if not q_tokens:
        return 0.0
    hits = 0.0
    total = 0.0
    for token in q_tokens:
        weight = 1.0 + min(len(token), 5) / 5.0
        total += weight
        if token in text:
            hits += weight
    return round(hits / total, 4) if total else 0.0


def rerank(
    query: str,
    documents: Sequence[Dict],
    top_k: int | None = None,
    text_key: str = "text",
    threshold: float | None = None,
) -> List[Dict]:
    """精排并截断。

    每个文档会写入 `rerank_score` 与 `rerank_backend`，
    下游 `node_context_build` 与 trace 都用它，不要在别处重算分数。

    `threshold` 只对真实模型生效；降级到词项覆盖时不做阈值过滤（原因见下）。
    """
    docs = [dict(d) for d in documents if d]
    if not docs:
        return []

    top_k = top_k or trip_config.rerank_top_k
    threshold = reranker_config.score_threshold if threshold is None else threshold

    model = _load_model()
    if model is not None:
        pairs = [[query, str(d.get(text_key, ""))[: reranker_config.max_length]] for d in docs]
        try:
            scores = model.compute_score(pairs, normalize=True)
            if isinstance(scores, float):
                scores = [scores]
        except Exception:
            logger.exception("[rerank] 打分失败，降级为词项覆盖")
            scores = [_lexical_score(query, str(d.get(text_key, ""))) for d in docs]
        backend = "bge-reranker"
    else:
        scores = [_lexical_score(query, str(d.get(text_key, ""))) for d in docs]
        backend = "lexical-fallback"

    for doc, score in zip(docs, scores):
        doc["rerank_score"] = round(float(score), 4)
        doc["rerank_backend"] = backend

    ranked = sorted(docs, key=lambda d: d["rerank_score"], reverse=True)

    if backend == "bge-reranker":
        kept = [d for d in ranked if d["rerank_score"] >= threshold]
        # 全部低于阈值时不要返回空——返回最高分的那条，让上游自己决定"证据不足"
        if not kept:
            kept = ranked[:1]
    else:
        # 降级打分用的是词项覆盖率，取值范围和 cross-encoder 的 sigmoid 分完全不是一回事，
        # 拿模型阈值来卡它会把 20 条砍成 1 条。降级时不做阈值过滤：
        # 排不出高低就保留 RRF 原序（sorted 是稳定排序），宁可多给几条噪音，也不能漏掉证据。
        kept = ranked
    return kept[:top_k]


def health() -> dict:
    return {
        "model_path": reranker_config.model_path,
        "loaded": _model is not None,
        "load_failed": _model_failed,
        "threshold": reranker_config.score_threshold,
    }
