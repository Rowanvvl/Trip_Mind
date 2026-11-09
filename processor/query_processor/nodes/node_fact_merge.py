"""事实汇聚。

三路外部调用各自可能失败或跳过，这里做四件事：
1. 统一结构（`route / weather / budget` 三个键）；
2. 记录 `degraded`——任一路失败就为 True，答案里要能看出"这块没查到"；
3. 生成给 prompt 用的 `text`：把结构化事实压成几行中文，
   **不带原始 JSON**（模型读 JSON 容易漏字段，读中文更容易用对）；
4. 时间戳保留到秒，`answer_critic` 用它判断事实是否过期。
"""
from __future__ import annotations

from typing import Any, Dict

from common.config import trip_config
from common.logging.logger import logger, node_log
from processor.query_processor.state import QueryGraphState
from utils.format_utils import age_hours, money_range


@node_log("node_fact_merge")
def node_fact_merge(state: QueryGraphState) -> dict:
    """合并三路事实。

    输出：`facts`、`degraded`
    """
    route = state.get("route_facts") or {}
    weather = state.get("weather_facts") or {}
    budget = state.get("budget_facts") or {}

    lines = []

    if route and not route.get("degraded"):
        summaries = route.get("summaries") or []
        if summaries:
            lines.append(f"路线：{route.get('origin')} → {route.get('destination')}，{'；'.join(summaries)}〔高德〕")
    elif route:
        lines.append(f"路线：未取到数据（{route.get('error') or route.get('reason') or '接口不可用'}）")

    if weather and not weather.get("degraded"):
        lines.append(f"天气：{weather.get('summary')}〔天气〕")
        if weather.get("rain_days"):
            lines.append(f"预报有雨的日期：{'、'.join(weather['rain_days'])}")
        lines.append(f"（{weather.get('capability_note')}）")
    elif weather:
        lines.append("天气：未取到数据")

    if budget and not budget.get("degraded"):
        low, high = budget.get("total", [0, 0])
        lines.append(f"预算：{money_range(low, high)}（{budget.get('days')} 天 {budget.get('travelers')} 人，"
                     f"{budget.get('level')}档）〔估算〕")
        for assumption in (budget.get("assumptions") or [])[:4]:
            lines.append(f"  口径：{assumption}")
        comparison = budget.get("comparison") or {}
        if comparison.get("comment"):
            lines.append(f"  预算比对：{comparison['comment']}")
    elif budget:
        lines.append(f"预算：无法测算（{budget.get('reason')}）")

    degraded = bool(route.get("degraded") or weather.get("degraded") or budget.get("degraded"))

    # 过期检查：超过 FACT_MAX_AGE_H 的事实仍传，但要标明，让生成层避免用它做"明天"判断
    stale = []
    for name, payload in (("route", route), ("weather", weather)):
        age = age_hours((payload or {}).get("fetched_at"))
        if age is not None and age > trip_config.fact_max_age_h:
            stale.append({"name": name, "age_hours": age})
    if stale:
        lines.append(f"注意：以下事实已超过 {trip_config.fact_max_age_h} 小时，不要用于当下的时效判断："
                     + "、".join(f"{s['name']}({s['age_hours']}h)" for s in stale))

    facts: Dict[str, Any] = {
        "route": route,
        "weather": weather,
        "budget": budget,
        "degraded": degraded,
        "stale": stale,
        "text": "\n".join(lines) if lines else "（本次未调用外部实时数据）",
    }

    if degraded:
        logger.info("[fact_merge] 存在降级的事实来源，答案需说明缺失部分")

    return {
        "facts": facts,
        "degraded": bool(state.get("degraded")) or degraded,
    }
