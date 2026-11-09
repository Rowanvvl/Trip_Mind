"""HyDE 召回：先让模型写一段"像攻略原文"的答案，再用它去检索。

为什么有效：用户问「带老人去哪合适」时，向量空间里这句话与语料中的判断句
（「带老人不要排爬升大的点」）距离很远；而模型生成的假答案与真实语料的
**文体和用词**更接近，检索距离立刻拉近。

代价是多一次生成调用，所以只在 dense 召回偏少时才值得跑。
本节点与 dense 并行，用便宜模型。
"""
from __future__ import annotations

from common.config import lm_config
from common.logging.logger import node_log
from processor.query_processor.nodes._retrieval import normalize_hits, preview, search_with_relax
from processor.query_processor.state import QueryGraphState
from utils.clients import milvus_utils
from utils.load_prompt import render_safe
from utils.lm import embedding_utils
from utils.lm.lm_utils import chat, record_usage
from utils.trace_utils import record_retrieval


@node_log("node_search_hyde")
def node_search_hyde(state: QueryGraphState) -> dict:
    """生成假设答案后检索。

    输出：`hyde_chunks`
    """
    if not lm_config.enabled:
        return {"hyde_chunks": []}

    query = state.get("rewritten_query") or state.get("original_query") or ""
    city = state.get("city") or ""
    content_type = state.get("content_type") or "综合"

    prompt = render_safe("hyde_travel", "", city=city or "（未指定）", content_type=content_type, query=query)
    result = chat([{"role": "user", "content": prompt}], tier="light", temperature=0.5)
    record_usage(state.get("trace_id", ""), result)
    if not result.ok:
        return {"hyde_chunks": []}

    dense, _, _ = embedding_utils.encode_query(result.text)

    def _search(expr: str, limit: int):
        return milvus_utils.search_dense(dense, expr, limit)

    found = search_with_relax(_search, state.get("city"), content_type, label="hyde")
    chunks = normalize_hits(found["docs"], "hyde")

    record_retrieval(
        state.get("trace_id", ""),
        hyde={"count": len(chunks), "level": found["level"], "top": preview(chunks)},
    )
    return {"hyde_chunks": chunks}
