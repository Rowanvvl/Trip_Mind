"""检索节点的公共逻辑（内部模块，不对图暴露）。

三个检索节点（dense / sparse / hyde）与子问题检索都要做同样三件事：
1. 依据 city + content_type 构造 Milvus 过滤；
2. 过滤命中为空时**自动放宽**（先去掉类型，再去掉城市）；
3. 结果归一化成统一字段。

放宽这一步是必须的：用户没说城市时（「5 天带老人怎么排」），
硬加城市过滤会直接返回空，答案变成「知识库未收录」——这是 RAG 最典型的漏召。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from common.config import trip_config
from common.logging.logger import logger
from utils.escape_milvus_string_utils import ExprBuildError, build_filter
from utils.geo.city_gazetteer import city_scope, normalize_city


def build_expr(city: str | None, content_type: str | None, with_city: bool = True) -> str:
    """构造过滤表达式，任何非法取值都退化成空过滤而不是抛给 Milvus。"""
    try:
        scope = city_scope(city) if (city and with_city) else {"cities": []}
        cities = scope.get("cities") or []
        types = [content_type] if content_type and content_type != "综合" else []
        if len(cities) > 1:
            # 区域类目的地（云南）要匹配省与子城市
            from utils.escape_milvus_string_utils import and_expr, eq_expr, in_expr

            return and_expr(in_expr("city", cities), in_expr("content_type", types))
        return build_filter(city=cities[0] if cities else None, content_types=types)
    except ExprBuildError as exc:
        logger.warning(f"[retrieval] 过滤条件非法，已退化为无过滤：{exc}")
        return ""


def search_with_relax(
    search_fn: Callable[..., List[dict]],
    city: str | None,
    content_type: str | None,
    limit: int | None = None,
    label: str = "search",
) -> Dict[str, list | str | bool]:
    """按「城市+类型 → 仅城市 → 无过滤」三级回退检索。

    返回 `{docs, expr, relaxed, level}`，`relaxed=True` 说明发生过放宽，
    trace 里要能看到——否则「为什么召回了别的城市」无从解释。
    """
    limit = limit or trip_config.retrieve_top_k
    attempts = [
        ("city+type", build_expr(city, content_type, True)),
        ("city", build_expr(city, None, True)),
        ("none", ""),
    ]

    for level, expr in attempts:
        docs = search_fn(expr, limit)
        if docs:
            return {"docs": docs, "expr": expr, "relaxed": level != "city+type", "level": level}

    logger.info(f"[{label}] 三级回退后仍无召回 city={city} type={content_type}")
    return {"docs": [], "expr": "", "relaxed": True, "level": "empty"}


def normalize_hits(docs: List[dict], source: str) -> List[dict]:
    """统一字段名，避免下游节点到处写 `doc.get("text") or doc.get("content")`。"""
    out = []
    for doc in docs or []:
        item = {
            "chunk_id": doc.get("chunk_id", ""),
            "doc_id": doc.get("doc_id", 0),
            "doc_title": doc.get("doc_title", ""),
            "section": doc.get("section", ""),
            "city": normalize_city(doc.get("city", "")),
            "content_type": doc.get("content_type", ""),
            "district": doc.get("district", ""),
            "entities": doc.get("entities", []),
            "source_path": doc.get("source_path", ""),
            "updated_at": doc.get("updated_at", ""),
            "text": doc.get("text", ""),
            "retrieval_source": source,
        }
        for key in ("dense_score", "sparse_score", "hybrid_score", "rerank_score"):
            if key in doc:
                item[key] = doc[key]
        out.append(item)
    return out


def dedup_chunks(chunks: List[dict]) -> List[dict]:
    """按 chunk_id 去重，保留分数最高的那条。"""
    best: Dict[str, dict] = {}
    for chunk in chunks:
        key = chunk.get("chunk_id") or chunk.get("text", "")[:64]
        current = best.get(key)
        if current is None:
            best[key] = chunk
            continue
        if _score(chunk) > _score(current):
            best[key] = chunk
    return list(best.values())


def _score(chunk: dict) -> float:
    for key in ("rerank_score", "hybrid_score", "dense_score", "sparse_score"):
        if key in chunk:
            return float(chunk[key] or 0)
    return 0.0


def preview(chunks: List[dict], limit: int = 3) -> List[dict]:
    """trace 里只存预览，不存全文，避免 trace 文档过大。"""
    return [
        {
            "chunk_id": c.get("chunk_id"),
            "doc_title": c.get("doc_title"),
            "section": c.get("section"),
            "score": _score(c),
        }
        for c in chunks[:limit]
    ]
