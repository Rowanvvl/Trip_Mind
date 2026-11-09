"""预算事实。

用规则测算，模型只负责把结果讲成人话（在生成节点里），绝不参与计算。
缺天数或人数时用槽位默认值，并把假设写进 facts，答案必须带上这些假设。
"""
from __future__ import annotations

from common.logging.logger import node_log
from processor.query_processor.state import QueryGraphState
from utils.tools import budget_utils


@node_log("node_budget")
def node_budget(state: QueryGraphState) -> dict:
    """测算花费区间。

    输出：`budget_facts`、`degraded`（缺关键条件时）
    """
    slots = state.get("slots") or {}
    city = state.get("city") or slots.get("city") or ""
    days = slots.get("days")
    travelers = int(slots.get("travelers") or 2)

    if not city or not days:
        return {
            "budget_facts": {
                "degraded": True,
                "reason": "缺少城市或天数，无法测算",
                "missing": [k for k, v in (("city", city), ("days", days)) if not v],
            },
            "degraded": True,
        }

    estimate = budget_utils.estimate(
        city=city,
        days=int(days),
        travelers=travelers,
        level=slots.get("budget_level"),
    )
    comparison = budget_utils.compare(estimate, slots.get("budget"))
    estimate["comparison"] = comparison
    estimate["lines"] = budget_utils.format_lines(estimate)

    # 门票若能从未被用在测算里，这里补一条说明，避免用户以为门票已含具体景点
    if not estimate["breakdown"]["tickets"][0]:
        estimate["assumptions"].append("门票为 0 说明未识别到收费景点，需按实际行程补充")

    return {"budget_facts": estimate}
