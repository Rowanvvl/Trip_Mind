"""路线事实：按出行方式调地图，给出可对比的方案。

真正的价值不是"给一条路线"，而是**把语料的经验判断与真实距离对上**：
知识库说「东线西线得分天」，地图给出「亚龙湾到天涯海角 42 公里、62 分钟」，
用户立刻就理解了为什么不能一天跑完。

出行方式判定：
- 用户明确说了（自驾/地铁/步行/骑行）→ 只给这一种；
- 没说 → 给「公共交通 + 打车/自驾」两种对比，让用户按预算和时间选。
"""
from __future__ import annotations

from common.config import amap_config
from common.logging.logger import node_log
from processor.query_processor.state import QueryGraphState
from utils.format_utils import human_distance, human_duration
from utils.geo.city_gazetteer import find_attractions_in_text
from utils.tools import amap_utils
from utils.tools.tool_registry import is_open
from utils.trace_utils import record_tool

_MODE_MAP = {
    "自驾": "driving",
    "打车": "driving",
    "公共交通": "transit",
    "地铁": "transit",
    "公交": "transit",
    "步行": "walking",
    "骑行": "bicycling",
}


@node_log("node_route_plan")
def node_route_plan(state: QueryGraphState) -> dict:
    """调用高德路径规划。

    输出：`route_facts`、`degraded`（失败时）
    """
    query = state.get("rewritten_query") or state.get("original_query") or ""
    city = state.get("city") or ""
    slots = state.get("slots") or {}
    trace_id = state.get("trace_id", "")

    places = find_attractions_in_text(query, city)
    if len(places) < 2:
        # 只有单个地点时也给"从城市中心出发"的方案，比"没有数据"有用
        if len(places) == 1 and city:
            origin, destination = city, places[0].name
        else:
            return {"route_facts": {"degraded": True, "reason": "问题里没有明确的起终点"}, "degraded": True}
    else:
        origin, destination = places[0].name, places[1].name

    preferred = _guess_mode(state)

    if not amap_config.enabled or is_open("amap_direction"):
        result = {"origin": origin, "destination": destination, "options": [], "degraded": True,
                  "error": "amap_unavailable", "source": "高德地图"}
    else:
        result = amap_utils.plan_route_for_query(origin, destination, city=city or None, preferred=preferred)

    record_tool(
        trace_id,
        "amap_direction",
        not result.get("degraded"),
        0,
        origin=origin,
        destination=destination,
        options=len(result.get("options") or []),
    )

    # 给答案用的可读摘要
    summaries = []
    for option in result.get("options") or []:
        mode_label = {"transit": "公共交通", "driving": "驾车", "walking": "步行", "bicycling": "骑行"}.get(
            option.get("mode", ""), option.get("mode", "")
        )
        text = f"{mode_label}约 {human_duration((option.get('duration_s') or 0) / 60)}，{human_distance(option.get('distance_m') or 0)}"
        if option.get("estimated"):
            text += "（估算）"
        summaries.append(text)

    result.update({"origin": origin, "destination": destination, "summaries": summaries, "preferred": preferred})
    return {"route_facts": result, "degraded": bool(state.get("degraded")) or bool(result.get("degraded"))}


def _guess_mode(state: QueryGraphState) -> str | None:
    """从槽位或问句里认出出行方式；认不出返回 None 让工具给两种方案。"""
    query = state.get("original_query") or ""
    slot_mode = (state.get("slots") or {}).get("transport")
    if slot_mode:
        return _MODE_MAP.get(slot_mode)
    for label, mode in _MODE_MAP.items():
        if label in query:
            return mode
    return None
