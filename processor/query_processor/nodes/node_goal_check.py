"""目标验收。

「成功」必须有定义，否则每次都只能说"跑了没报错"。这里的定义来自
`goal_spec`（由 `node_query_understand` 产出）：
- `min_citations`：引用条数下限（综合类默认 3）；
- `must_cover`：必须覆盖的内容类型；
- `needs_fresh_facts`：是否必须带实时数据。

不达标时允许**补检索一次**（`goal_retry`），仍不达标就放行——
但要在 `goal_result` 里记下缺口，trace 与评测集都用它。

注意：`goal_retry` 与 `critic_retry` 是两个独立计数器。合用一个会出现
"critic 用完额度导致 goal 无法补救"这种难查的耦合问题。
"""
from __future__ import annotations

from typing import List

from common.config import trip_config
from common.logging.logger import logger, node_log
from processor.query_processor.state import QueryGraphState
from utils.trace_utils import record_goal


@node_log("node_goal_check")
def node_goal_check(state: QueryGraphState) -> dict:
    """核对验收标准。

    输出：`goal_result`、`goal_retry`
    """
    spec = state.get("goal_spec") or {}
    citations = state.get("citations") or []
    evidence_types = state.get("evidence_types") or []
    facts = state.get("facts") or {}
    trace_id = state.get("trace_id", "")

    min_citations = int(spec.get("min_citations") or 1)
    must_cover = spec.get("must_cover") or []
    needs_fresh = bool(spec.get("needs_fresh_facts"))

    gaps: List[str] = []
    if len(citations) < min_citations:
        gaps.append(f"引用不足：{len(citations)}/{min_citations}")
    missing_types = [ct for ct in must_cover if ct not in evidence_types]
    if missing_types:
        gaps.append(f"缺证据类型：{'、'.join(missing_types)}")
    if needs_fresh and facts.get("degraded"):
        gaps.append("需要实时数据但未取到")

    achieved = not gaps
    can_retry = (state.get("goal_retry") or 0) < trip_config.goal_max_retry and any(
        g.startswith("引用不足") or g.startswith("缺证据类型") for g in gaps
    )

    if gaps:
        logger.info(f"[goal_check] 未达标：{gaps}")

    record_goal(trace_id, achieved, gaps=gaps, citations=len(citations), evidence=evidence_types)

    payload: dict = {"goal_result": {"achieved": achieved, "gaps": gaps, "retryable": can_retry}}
    if can_retry:
        payload["goal_retry"] = (state.get("goal_retry") or 0) + 1
    return payload


def after_goal(state: QueryGraphState) -> str:
    """可补救 → 回检索补一轮；否则收尾挂图。"""
    result = state.get("goal_result") or {}
    if result.get("retryable") and not result.get("achieved"):
        return "retry"
    return "images"
