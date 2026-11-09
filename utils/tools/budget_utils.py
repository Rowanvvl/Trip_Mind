"""预算测算：**规则算，不让模型算**。

模型算钱几乎必错，而且是错得看不出破绽的那种。这里的做法是：
1. 按城市档位 × 消费档次取单价区间（`budget_config`）；
2. 门票优先用语料命中的真实票价，取不到才用城市均值；
3. 输出**区间**而不是单值，并且必须附带 `assumptions`。

`assumptions` 不是装饰：用户投诉「你说的 5000 结果花了 8000」时，
唯一能解释清楚的就是「按 2 人 3 晚、舒适档、不含大交通测算」这几行字。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from common.config import budget_config

DEFAULT_MEALS = budget_config.meals_per_day


def _rooms_for(travelers: int, room_type: str = "双人") -> int:
    factor = budget_config.room_factor.get(room_type, 1.0)
    per_room = 2 * factor
    return max(1, math.ceil(travelers / per_room))


def _scale(price_range: tuple[float, float], times: float) -> tuple[float, float]:
    low, high = price_range
    return low * times, high * times


def estimate(
    city: str,
    days: int,
    travelers: int = 2,
    level: Optional[str] = None,
    ticket_total: Optional[float] = None,
    include_intercity: bool = False,
    intercity_km: Optional[float] = None,
    local_transport_km: Optional[float] = None,
    room_type: str = "双人",
) -> Dict:
    """测算一次行程的花费区间。所有数字都是「口径 + 区间」，不给假精确的单值。"""
    days = max(int(days or 1), 1)
    travelers = max(int(travelers or 1), 1)
    level = level or budget_config.default_level
    tier = budget_config.tier_of(city)

    hotel_table = budget_config.hotel_price.get(tier, budget_config.hotel_price["tier2"])
    meal_table = budget_config.meal_price.get(tier, budget_config.meal_price["tier2"])
    hotel_unit = hotel_table.get(level, hotel_table[budget_config.default_level])
    meal_unit = meal_table.get(level, meal_table[budget_config.default_level])

    nights = max(days - 1, 0)
    rooms = _rooms_for(travelers, room_type)

    # 住宿
    lodging = _scale(hotel_unit, nights * rooms)

    # 餐饮
    meals = _scale(meal_unit, days * travelers * DEFAULT_MEALS)

    # 门票
    if ticket_total:
        tickets = _scale((float(ticket_total), float(ticket_total)), travelers)
        ticket_note = "门票按语料命中的实际票价合计"
    else:
        ticket_unit = budget_config.ticket_per_day.get(tier, (50, 140))
        tickets = _scale(ticket_unit, days * travelers)
        ticket_note = f"门票按每人每天 {ticket_unit[0]:.0f}–{ticket_unit[1]:.0f} 元估算（未命中具体票价）"

    # 市内交通
    if local_transport_km:
        taxi_unit = budget_config.taxi_per_km
        low = local_transport_km * taxi_unit[0]
        high = local_transport_km * taxi_unit[1] + 15 * days
        local_transport = (low, high)
        transport_note = f"市内交通按约 {local_transport_km:.0f} 公里、{taxi_unit[0]}–{taxi_unit[1]} 元/公里估算"
    else:
        local_unit = budget_config.local_transport_per_day.get(tier, (20, 45))
        local_transport = _scale(local_unit, days * travelers)
        transport_note = f"市内交通按每人每天 {local_unit[0]:.0f}–{local_unit[1]:.0f} 元估算"

    # 城际
    intercity = (0.0, 0.0)
    intercity_note = "不含往返大交通（机票/高铁）"
    if include_intercity and intercity_km:
        rail = intercity_km * budget_config.rail_per_km * 2 * travelers
        if intercity_km <= budget_config.flight_short_km:
            flight = _scale(budget_config.flight_price_short, 2 * travelers)
            intercity = (min(rail, flight[0]), max(rail, flight[1]))
            intercity_note = f"含往返大交通：约 {intercity_km:.0f} 公里，高铁约 {rail:.0f} 元，机票 {flight[0]:.0f}–{flight[1]:.0f} 元"
        else:
            flight = _scale(budget_config.flight_price_long, 2 * travelers)
            intercity = (min(rail, flight[0]), max(rail, flight[1]))
            intercity_note = f"含往返大交通：约 {intercity_km:.0f} 公里，机票 {flight[0]:.0f}–{flight[1]:.0f} 元"

    total_low = lodging[0] + meals[0] + tickets[0] + local_transport[0] + intercity[0]
    total_high = lodging[1] + meals[1] + tickets[1] + local_transport[1] + intercity[1]

    assumptions: List[str] = [
        f"按 {travelers} 人出行、{days} 天 {nights} 晚、{rooms} 间房测算",
        f"住宿档次：{level}（每晚 {hotel_unit[0]:.0f}–{hotel_unit[1]:.0f} 元/间）",
        f"餐饮按每天 {DEFAULT_MEALS} 餐、人均 {meal_unit[0]:.0f}–{meal_unit[1]:.0f} 元/餐",
        ticket_note,
        transport_note,
        intercity_note,
        "不含购物、演出加项与临时改签",
    ]

    return {
        "city": city,
        "days": days,
        "travelers": travelers,
        "nights": nights,
        "rooms": rooms,
        "level": level,
        "currency": "CNY",
        "breakdown": {
            "lodging": [round(lodging[0]), round(lodging[1])],
            "meals": [round(meals[0]), round(meals[1])],
            "tickets": [round(tickets[0]), round(tickets[1])],
            "local_transport": [round(local_transport[0]), round(local_transport[1])],
            "intercity": [round(intercity[0]), round(intercity[1])],
        },
        "total": [round(total_low), round(total_high)],
        "per_person": [round(total_low / travelers), round(total_high / travelers)],
        "per_person_per_day": [
            round(total_low / travelers / days),
            round(total_high / travelers / days),
        ],
        "assumptions": assumptions,
        "source": "规则测算",
    }


def from_slots(slots: Dict) -> Optional[Dict]:
    """从槽位直接测算。缺城市或天数时返回 None——不猜。"""
    city = (slots or {}).get("city")
    days = (slots or {}).get("days")
    if not city or not days:
        return None
    return estimate(
        city=city,
        days=int(days),
        travelers=int(slots.get("travelers") or 2),
        level=slots.get("budget_level"),
    )


def compare(estimated: Dict, budget_cap: Optional[float]) -> Dict:
    """与用户预算比对，给「够不够、差多少」的结论，而不是只列数字。"""
    if not budget_cap or not estimated:
        return {"fits": None, "gap": None, "comment": "用户未给出预算，不做判断"}
    low, high = estimated.get("total", [0, 0])
    cap = float(budget_cap)
    if cap >= high:
        return {
            "fits": True,
            "gap": [round(cap - high), round(cap - low)],
            "comment": f"预算 {cap:.0f} 元足够覆盖上限，剩余约 {cap - high:.0f} 元可做加项",
        }
    if cap >= low:
        return {
            "fits": True,
            "gap": [0, round(cap - low)],
            "comment": f"预算 {cap:.0f} 元落在区间内，需按下限安排（{estimated.get('level')}档偏紧）",
        }
    return {
        "fits": False,
        "gap": [round(cap - high), round(cap - low)],
        "comment": f"预算 {cap:.0f} 元低于区间下限约 {low - cap:.0f} 元，建议降档或缩短天数",
    }


def taxi_estimate(distance_km: float, minutes: Optional[int] = None) -> Dict:
    """单段打车估算，用于「打车贵吗」这类问法。"""
    low_unit, high_unit = budget_config.taxi_per_km
    base = 10.0  # 起步价，按常见城市下限取
    low = base + distance_km * low_unit
    high = base + distance_km * high_unit
    if minutes and minutes > 15:
        # 低速等候费，粗略叠加
        high += (minutes - 15) * 0.5
    return {
        "distance_km": round(distance_km, 1),
        "fare": [round(low), round(high)],
        "assumptions": [f"含起步价 {base:.0f} 元，{low_unit}–{high_unit} 元/公里，未计高速费"],
    }


def format_lines(estimated: Dict) -> List[str]:
    """把测算结果转成可直接进答案的几行字（前端也用它）。"""
    if not estimated:
        return []
    low, high = estimated.get("total", [0, 0])
    breakdown = estimated.get("breakdown", {})
    lines = [f"总花费约 {low:.0f}–{high:.0f} 元（人均 {estimated['per_person'][0]}–{estimated['per_person'][1]} 元）"]
    labels = {
        "lodging": "住宿",
        "meals": "餐饮",
        "tickets": "门票",
        "local_transport": "市内交通",
        "intercity": "往返大交通",
    }
    for key, label in labels.items():
        value = breakdown.get(key)
        if value and value[1] > 0:
            lines.append(f"- {label}：{value[0]}–{value[1]} 元")
    return lines
