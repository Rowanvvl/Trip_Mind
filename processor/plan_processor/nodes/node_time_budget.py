"""时间预算与雨天替换。

两件必须在这里完成、不能只在提示词里"提醒"的事：
1. **算清每天到底要多少分钟**：点位游玩时长 + 路段车程 + 用餐 2 小时。
   超了就要砍点，砍哪个由规则定（先砍非必去、再砍与片区不一致的）；
2. **雨天真的替换**：把当天的户外点位换成同城室内候选，
   而不是在提示词里写「如遇雨天建议室内」——那种建议模型会忽略。

这一节点的输出是 `plan_generate` 的输入约束，模型只能在被砍过的候选里选。
"""
from __future__ import annotations

from typing import List

from common.config import trip_config
from common.logging.logger import logger, node_log
from processor.plan_processor.state import PlanGraphState
from utils.format_utils import human_duration
from utils.geo.geo_utils import estimate_distance_km, travel_minutes
from utils.tools import weather_utils


@node_log("node_time_budget")
def node_time_budget(state: PlanGraphState) -> dict:
    """计算每日时长并在雨天替换点位。

    输出：`daily_time_budget`、`weather_days`、`daily_candidates`（可能被替换）、`route_legs`（重建后）
    """
    city = state.get("city") or ""
    clustered = state.get("clustered_days") or []
    legs = state.get("route_legs") or []
    daily_candidates = dict(state.get("daily_candidates") or {})
    constraints = state.get("constraints") or {}

    weather = weather_utils.get_weather(city) if city else {}
    forecast = weather.get("forecast") or []
    rainy = set(weather_utils.rain_days(forecast))

    # 日期 → 是否雨天：有开始日期就按真实日期对，没有就按第 1 天对第 1 条预报
    weather_days = []
    for entry in clustered:
        day = entry["day"]
        cast = forecast[day - 1] if day - 1 < len(forecast) else {}
        is_rain = bool(cast) and (
            weather_utils.is_rainy(cast.get("day_weather")) or weather_utils.is_rainy(cast.get("night_weather"))
        )
        weather_days.append({"day": day, "date": cast.get("date", ""), "rain": is_rain, "cast": cast})

    rain_day_numbers = {item["day"] for item in weather_days if item["rain"]}

    budgets: List[dict] = []
    all_legs: List[dict] = []
    for entry in clustered:
        day = entry["day"]
        members = list(entry["members"])
        day_legs = [leg for leg in legs if leg.get("day") == day]

        # 雨天替换：户外 → 室内
        replaced = []
        if day in rain_day_numbers:
            members, replaced = _replace_outdoor(city, members, daily_candidates.get(day, []), constraints)

        # 砍点之后必须**重建路段**：被砍掉的点不能再占车程。
        # 之前只减 play_min 不减 travel_min，导致「砍掉都江堰」后当天仍记着 181 分钟车程，
        # 校验环节还会拿这条不存在的路段判折返。
        live_legs = _rebuild_legs(city, members, day_legs, day)
        play_min = sum(_duration_of(name, daily_candidates.get(day, [])) for name in members)
        travel_min = sum((leg.get("duration_s") or 0) / 60 for leg in live_legs)
        meal_min = 120

        total = play_min + travel_min + meal_min
        dropped = []
        # 超时就砍非必去的最后一个点，直到符合上限
        while total > trip_config.day_minutes and len(members) > 1:
            candidate = next((m for m in reversed(members) if not _is_must(m, daily_candidates.get(day, []))), None)
            if candidate is None:
                break
            members.remove(candidate)
            dropped.append(candidate)
            play_min -= _duration_of(candidate, daily_candidates.get(day, []))
            live_legs = _rebuild_legs(city, members, day_legs, day)
            travel_min = sum((leg.get("duration_s") or 0) / 60 for leg in live_legs)
            total = play_min + travel_min + meal_min

        all_legs.extend(live_legs)

        daily_candidates[day] = [c for c in daily_candidates.get(day, []) if c["name"] in members]

        if total > trip_config.day_minutes:
            logger.warning(f"[time_budget] 第 {day} 天仍超时：{total:.0f} 分钟")

        budgets.append(
            {
                "day": day,
                "area": entry.get("area", ""),
                "members": members,
                "play_min": round(play_min),
                "travel_min": round(travel_min),
                "meal_min": meal_min,
                "total_min": round(total),
                "limit_min": trip_config.day_minutes,
                "rain": day in rain_day_numbers,
                "replaced_indoor": replaced,
                "dropped": dropped,
                "summary": f"游玩 {human_duration(play_min)} + 车程 {human_duration(travel_min)} + 用餐 2 小时 "
                           f"= {human_duration(total)}",
            }
        )

    return {
        "daily_time_budget": budgets,
        "weather_days": weather_days,
        "daily_candidates": daily_candidates,
        "route_legs": all_legs,
    }


def _rebuild_legs(city: str, members: List[str], existing: List[dict], day: int) -> List[dict]:
    """按最终点位顺序重建当日路段。

    已知路段直接复用（里面是真实高德时长），新出现的相邻对用直线估算补上并标 `estimated`。
    """
    known = {(leg.get("from"), leg.get("to")): leg for leg in existing}
    rebuilt: List[dict] = []
    for origin, destination in zip(members, members[1:]):
        leg = known.get((origin, destination))
        if leg:
            rebuilt.append(leg)
            continue
        distance = estimate_distance_km(origin, destination, city)
        rebuilt.append(
            {
                "day": day,
                "from": origin,
                "to": destination,
                "distance_m": (distance or 0) * 1000,
                "duration_s": (travel_minutes(distance, "打车") or 0) * 60,
                "mode": "打车",
                "estimated": True,
            }
        )
    return rebuilt


def _duration_of(name: str, candidates: List[dict]) -> float:
    for item in candidates:
        if item.get("name") == name:
            return float(item.get("duration_min") or 120)
    return 120.0


def _is_must(name: str, candidates: List[dict]) -> bool:
    for item in candidates:
        if item.get("name") == name:
            return bool(item.get("must_go"))
    return False


def _replace_outdoor(city: str, members: List[str], candidates: List[dict], constraints: dict) -> tuple[list, list]:
    """把户外点位换成室内。没有可换的就减少户外数量（宁少一个点，不安排淋雨）。"""
    indoor_pool = [c for c in candidates if c.get("indoor") and c["name"] not in members]
    if not indoor_pool:
        from utils.geo.geo_utils import select_indoor_alternatives

        indoor_pool = select_indoor_alternatives(city, exclude=members, limit=3)

    kept, replaced = [], []
    used = 0
    for name in members:
        candidate = next((c for c in candidates if c.get("name") == name), {})
        is_outdoor = not candidate.get("indoor", False)
        if is_outdoor and used < len(indoor_pool) and not candidate.get("must_go"):
            replacement = indoor_pool[used]
            used += 1
            replaced.append({"from": name, "to": replacement.get("name")})
            kept.append(replacement.get("name"))
        elif is_outdoor and used >= len(indoor_pool) and not candidate.get("must_go"):
            replaced.append({"from": name, "to": None})
        else:
            kept.append(name)
    return kept, replaced
