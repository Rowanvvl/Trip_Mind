"""稀疏向量构造。

BGE-M3 会直接给出 `{token_id: weight}`，本地模型可用时优先走它；
没加载模型时退化成 jieba 分词 + 词频权重（效果差一些，但能保证检索链路可跑通）。

Milvus 的 sparse 字段要求 key 为整数，所以 jieba 分支用哈希把词映射到 int32 空间。
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Dict, Iterable

_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "有", "我", "你", "他", "它", "这", "那",
    "怎么", "什么", "哪些", "如何", "请问", "一下", "可以", "需要", "还是", "以及",
    "吗", "呢", "吧", "啊", "个", "去", "要", "就", "都", "很",
}


def _token_id(token: str) -> int:
    """稳定哈希到 int32 正区间，跨进程一致。"""
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def normalize_sparse(raw: Dict[int, float]) -> Dict[int, float]:
    """L2 归一化，去掉非正值，保证不同长度文本的分数可比。"""
    cleaned = {int(k): float(v) for k, v in (raw or {}).items() if float(v) > 0}
    if not cleaned:
        return {}
    norm = math.sqrt(sum(v * v for v in cleaned.values())) or 1.0
    return {k: round(v / norm, 6) for k, v in cleaned.items()}


def from_bge_output(token_weights: Dict[int, float]) -> Dict[int, float]:
    """BGE-M3 的 lexical_weights 直接用。"""
    return normalize_sparse(token_weights)


def tokenize(text: str) -> list[str]:
    """jieba 精确模式 + 过滤停用词与单字。"""
    import jieba

    words = [w.strip() for w in jieba.lcut(text or "")]
    return [w for w in words if len(w) > 1 and w not in _STOPWORDS and not w.isdigit()]


def from_text(text: str, boost_terms: Iterable[str] = ()) -> Dict[int, float]:
    """词频 + 亚线性缩放构造稀疏向量。

    `boost_terms` 用于把专名（蜈支洲岛、陕历博）的权重抬起来，
    这与 BM25 的 IDF 直觉一致：越少见的词越能定位文档。
    """
    tokens = tokenize(text)
    if not tokens:
        return {}
    counter = Counter(tokens)
    boost = {t for t in boost_terms if t}
    weights: Dict[int, float] = {}
    for token, freq in counter.items():
        weight = 1.0 + math.log(freq)
        if token in boost:
            weight *= 2.0
        tid = _token_id(token)
        weights[tid] = weights.get(tid, 0.0) + weight
    return normalize_sparse(weights)


def merge_sparse(*vectors: Dict[int, float]) -> Dict[int, float]:
    """合并多段文本的稀疏向量（切片正文 + 小节标题）。"""
    merged: Dict[int, float] = {}
    for vec in vectors:
        for k, v in (vec or {}).items():
            merged[int(k)] = merged.get(int(k), 0.0) + float(v)
    return normalize_sparse(merged)
