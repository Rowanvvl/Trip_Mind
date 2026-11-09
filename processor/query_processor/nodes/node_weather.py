"""天气事实。

**必须带着能力边界回答**：预报只有今天起 3 天，问第四天以后的要明说查不到。
这个节点只负责取数，不负责措辞；措辞由 `answer_out.prompt` 里的规则约束
（`weather_utils.capability_note()` 提供的原句会被塞进 facts）。
"""
from __future__ import annotations

from common.config import weather_config
from common.logging.logger import node_log
from processor.query_processor.state import QueryGraphState
from utils.format_utils import now
from utils.tools import weather_utils
from utils.trace_utils import record_tool


@node_log("node_weather")
def node_weather(state: QueryGraphState) -> dict:
    """取实时与预报天气。

    输出：`weather_facts`、`degraded`（失败时）
    """
    city = state.get("city") or (state.get("slots") or {}).get("city") or ""
    if not city:
        return {"weather_facts": {"degraded": True, "reason": "未识别城市"}, "degraded": True}

    data = weather_utils.get_weather(city)
    data["today"] = now().strftime("%Y-%m-%d")
    data["capability_note"] = weather_utils.capability_note()
    data["summary"] = weather_utils.describe(data)
    data["rain_days"] = weather_utils.rain_days(data.get("forecast") or [])
    data["outdoor_risk_ratio"] = weather_utils.outdoor_risk_ratio(data.get("forecast") or [])
    data["max_age_hours"] = weather_config.max_age_hours

    record_tool(
        state.get("trace_id", ""),
        "amap_weather",
        not data.get("degraded"),
        0,
        city=city,
        days=len(data.get("forecast") or []),
        rain_days=data["rain_days"],
    )

    return {
        "weather_facts": data,
        "degraded": bool(state.get("degraded")) or bool(data.get("degraded")),
    }
