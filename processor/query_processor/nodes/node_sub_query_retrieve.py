"""子问题补检索。

针对缺失的证据类型，拆出 1~3 个子问题分别检索，把结果并入候选池。
每路只用 `SUB_QUERY_TOP_K`（默认 5）的小预算——补检索的目的是"补类型"，
不是再捞一批同类文档。

**只执行一次**：`sub_query_used` 置 true 后，即使覆盖仍不足也不再重试，
由生成阶段在答案里说明「这部分缺可靠依据」，比无限循环检索诚实。
"""
from __future__ import annotations

from typing import List

from common.config import lm_config, trip_config
from common.logging.logger import logger, node_log
from processor.query_processor.nodes._retrieval import (
    dedup_chunks,
    normalize_hits,
    search_with_relax,
)
from processor.query_processor.state import QueryGraphState
from utils.clients import milvus_utils
from utils.load_prompt import render_safe
from utils.lm import embedding_utils
from utils.lm.lm_utils import chat_json, record_usage
from utils.trace_utils import record_retrieval


@node_log("node_sub_query_retrieve")
def node_sub_query_retrieve(state: QueryGraphState) -> dict:
    """按缺失类型补检索。

    输出：`rrf_chunks`（追加）、`sub_queries`、`sub_query_used`
    """
    query = state.get("rewritten_query") or state.get("original_query") or ""
    city = state.get("city") or ""
    covered = state.get("evidence_types") or []
    missing = state.get("missing_types") or []

    sub_queries = _split(query, city, covered, missing)
    if not sub_queries:
        return {"sub_query_used": True, "sub_queries": []}

    collected: List[dict] = list(state.get("rrf_chunks") or [])
    for item in sub_queries:
        sub_query = item.get("query", "")
        content_type = item.get("content_type", "")
        if not sub_query:
            continue
        dense, _, _ = embedding_utils.encode_query(sub_query)
        if not dense:
            continue
        found = search_with_relax(
            lambda expr, limit, d=dense: milvus_utils.search_dense(d, expr, limit),
            city,
            content_type,
            limit=trip_config.sub_query_top_k,
            label="sub_query",
        )
        hits = normalize_hits(found["docs"], "sub_query")
        for hit in hits:
            hit["sub_query"] = sub_query
        collected.extend(hits)
        logger.debug(f"[sub_query] {content_type} → {len(hits)} 条：{sub_query}")

    merged = dedup_chunks(collected)
    record_retrieval(
        state.get("trace_id", ""),
        sub_query={
            "queries": sub_queries,
            "added": len(merged) - len(state.get("rrf_chunks") or []),
            "total": len(merged),
        },
    )

    return {
        "rrf_chunks": merged,
        "sub_queries": sub_queries,
        "sub_query_used": True,
    }


def _split(query: str, city: str, covered: List[str], missing: List[str]) -> List[dict]:
    """拆子问题。模型可用时用提示词，否则按缺失类型拼模板句。"""
    if not missing:
        return []

    if lm_config.enabled:
        prompt = render_safe(
            "sub_query_split",
            "",
            city=city or "（未指定）",
            query=query,
            covered="、".join(covered) or "无",
            missing="、".join(missing),
        )
        data, result = chat_json(prompt, tier="light", fallback={"sub_queries": []})
        record_usage("", result)
        items = data.get("sub_queries") or []
        if items:
            return items[:3]

    # 模板兜底：直接按缺失类型问，足够触发对应类型的召回
    templates = {
        "交通指南": f"{city}从机场/高铁站到市区和主要景点怎么走",
        "线路推荐": f"{city}的经典线路和片区是怎么划分的",
        "酒店信息": f"{city}住宿选在哪个区域更合适",
        "美食推荐": f"{city}本地特色吃什么、去哪吃",
        "景点攻略": f"{city}最值得去的景点和取舍",
    }
    return [
        {"query": templates.get(ct, f"{city} {ct}"), "content_type": ct, "reason": "template"}
        for ct in missing[:3]
        if ct in templates
    ]
