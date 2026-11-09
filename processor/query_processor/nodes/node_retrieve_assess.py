"""证据充分性评估。

**规则判定，不靠模型感觉。** 判据是「覆盖了几类证据」而不是「分数高不高」：
分数高只说明片段沾题，覆盖类型才说明能不能撑起一个综合结论。

典型失败：「三亚 5 天带老人怎么排」只召回 6 条景点攻略，全都在讲哪个岛好看，
没有交通、没有住宿。这种答案写出来必然纸上谈兵，而重排分数可能是 0.9。

判定后只有两个出口：补检索（`node_sub_query_retrieve`）或直接组装上下文。
补检索**只做一次**（`sub_query_used`），避免无限循环。
"""
from __future__ import annotations

from common.config import CONTENT_TYPES, INTENT_REQUIRED_TYPES, trip_config
from common.logging.logger import logger, node_log
from processor.query_processor.state import QueryGraphState

# 综合类之外的默认要求：线路类问题必须同时有线路与交通
_DEFAULT_REQUIRED = {
    "景点攻略": ["景点攻略"],
    "线路推荐": ["线路推荐", "交通指南"],
    "酒店信息": ["酒店信息"],
    "美食推荐": ["美食推荐"],
    "交通指南": ["交通指南"],
    "综合": list(CONTENT_TYPES),
}


@node_log("node_retrieve_assess")
def node_retrieve_assess(state: QueryGraphState) -> dict:
    """判断证据够不够。

    输出：`evidence_types`、`evidence_coverage`、`missing_types`、`debug.assess`
    """
    docs = state.get("reranked_docs") or []
    content_type = state.get("content_type") or "综合"
    goal_spec = state.get("goal_spec") or {}

    covered = []
    for doc in docs:
        ct = doc.get("content_type")
        if ct and ct not in covered:
            covered.append(ct)

    required = goal_spec.get("must_cover") or _DEFAULT_REQUIRED.get(content_type) or list(CONTENT_TYPES)
    required = [ct for ct in required if ct in CONTENT_TYPES] or list(CONTENT_TYPES)

    missing = [ct for ct in required if ct not in covered]
    coverage = len([ct for ct in covered if ct in required])

    # 目标门槛与「要求类型数」取较小值：只问住宿的问题不该被要求覆盖五类
    threshold = min(trip_config.evidence_min_coverage, len(required))
    insufficient = coverage < threshold

    # 补检索只用一次；已经补过就直接放行，让生成阶段说明局限
    need_more = insufficient and not state.get("sub_query_used") and bool(docs)

    if need_more:
        logger.info(f"[retrieve_assess] 证据不足 coverage={coverage}/{threshold} missing={missing}")

    return {
        "evidence_types": covered,
        "evidence_coverage": coverage,
        "missing_types": missing,
        "debug": {
            **(state.get("debug") or {}),
            "assess": {
                "required": required,
                "covered": covered,
                "coverage": coverage,
                "threshold": threshold,
                "need_more": need_more,
            },
        },
    }


def after_assess(state: QueryGraphState) -> str:
    """证据不足且有缺口 → 补检索；否则组装上下文。"""
    assess = (state.get("debug") or {}).get("assess") or {}
    return "retrieve_more" if assess.get("need_more") else "context"
