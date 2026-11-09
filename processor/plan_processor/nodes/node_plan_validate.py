"""行程校验。

四条硬规则 + 两条软提醒。硬规则不通过就回退重排，软提醒只写进 `warnings`：

| rule | 判定 | 类型 |
|---|---|---|
| `cross_area` | 单日活动跨度超 `day_span_km_limit`，或单段车程折返超限 | 硬 |
| `closed_day` | 室内馆类排在周一 | 硬 |
| `overtime` | 单日总时长超上限 | 硬 |
| `budget_over` | 测算上限超用户预算 15% 以上 | 硬 |
| `elderly_intensity` | 带老人且连续两天高强度 | 软 |
| `rain_outdoor` | 雨天户外占比过高 | 软 |

「闭馆日」只用周一这一条通用规则，**不编造具体馆的闭馆日**——
语料没写的信息不能当成规则用，那等于用猜测约束生成。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List

from common.config import trip_config
from common.logging.logger import logger, node_log
from processor.plan_processor.state import PlanGraphState
from utils.geo.city_gazetteer import resolve_attraction
from utils.geo.geo_utils import haversine_km

# 周一闭馆的场馆类型关键词（通用惯例，仅对馆类生效）
_MUSEUM_KEYWORDS = ("博物馆", "博物院", "纪念馆", "美术馆", "美术馆", "展览馆")
_CLOSED_WEEKDAY = 0  # 周一


@node_log("node_plan_validate")
def node_plan_validate(state: PlanGraphState) -> dict:
    """校验并决定是否回退。

    输出：`validate_errors`、`validate_retry`
    """
    plan = state.get("plan_json") or {}
    city = state.get("city") or ""
    budgets = {item["day"]: item for item in (state.get("daily_time_budget") or [])}
    start = _start_date(state.get("start_date"))
    budget_estimate = state.get("budget_estimate") or {}
    constraints = state.get("constraints") or {}

    errors: List[Dict] = []
    warnings: List[Dict] = []

    for day in plan.get("days", []):
        day_no = int(day.get("day") or 0)
        names = [item.get("name") for item in day.get("items", []) if item.get("name")]

        # 1. 跨区折返：按**距离**判，不按行政区名判。
        #    故宫（东城）和什刹海（西城）只隔 3 公里，按区名判会把合理的安排判成错误；
        #    而「上午故宫、下午八达岭」差 60 公里，无论怎么划区都是折返。
        districts = {resolve_attraction(n, city).district for n in names if resolve_attraction(n, city)}
        districts.discard("")
        span = _day_span_km(names, city)
        if span > trip_config.day_span_km_limit:
            errors.append(
                {
                    "rule": "cross_area",
                    "day": day_no,
                    "detail": f"单日跨度 {span:.0f} 公里（上限 {trip_config.day_span_km_limit:.0f}），属于折返"
                              + (f"；涉及片区：{'、'.join(sorted(districts))}" if districts else ""),
                }
            )
        if len(names) > 1 and any(
            (leg.get("duration_s") or 0) / 60 > trip_config.daily_reverse_limit_min
            for leg in (state.get("route_legs") or [])
            if leg.get("day") == day_no
        ):
            errors.append(
                {
                    "rule": "cross_area",
                    "day": day_no,
                    "detail": f"单段车程超过 {trip_config.daily_reverse_limit_min} 分钟，属于折返",
                }
            )

        # 2. 闭馆日
        weekday = (start + timedelta(days=day_no - 1)).weekday() if start else None
        if weekday == _CLOSED_WEEKDAY:
            closed = [n for n in names if any(k in n for k in _MUSEUM_KEYWORDS)]
            if closed:
                errors.append(
                    {
                        "rule": "closed_day",
                        "day": day_no,
                        "detail": f"周一日程排了馆类点位（{'、'.join(closed)}），普遍闭馆",
                    }
                )

        # 3. 超时
        budget = budgets.get(day_no) or {}
        total = budget.get("total_min") or _items_minutes(day)
        if total > trip_config.day_minutes:
            errors.append(
                {
                    "rule": "overtime",
                    "day": day_no,
                    "detail": f"单日 {total} 分钟，超过上限 {trip_config.day_minutes}",
                }
            )

        # 4. 带老人强度（软）
        if constraints.get("elderly") and len(names) >= 4:
            warnings.append({"rule": "elderly_intensity", "day": day_no, "detail": f"带老人却排了 {len(names)} 个点"})

        # 5. 雨天户外（软）
        if budget.get("rain") and budget.get("replaced_indoor") is None and len(names) > 2:
            warnings.append({"rule": "rain_outdoor", "day": day_no, "detail": "预报有雨但未做室内替换"})

    # 6. 预算
    cap = float(state.get("budget") or 0)
    high = float((budget_estimate.get("total") or [0, 0])[1])
    if cap and high > cap * 1.15:
        errors.append(
            {
                "rule": "budget_over",
                "day": 0,
                "detail": f"测算上限 {high:.0f} 元超出预算 {cap:.0f} 元 15% 以上",
            }
        )

    retry = state.get("validate_retry") or 0
    if errors:
        logger.info(f"[plan_validate] 未通过 {len(errors)} 条：{[e['rule'] for e in errors]}")

    payload: dict = {
        "validate_errors": errors,
        "debug": {**(state.get("debug") or {}), "validate": {"errors": errors, "warnings": warnings}},
    }
    if errors and retry < trip_config.plan_max_retry:
        payload["validate_retry"] = retry + 1
    return payload


def _day_span_km(names: List[str], city: str) -> float:
    """一天内任意两点之间的最大直线距离。取不到坐标的点直接跳过。"""
    points = []
    for name in names:
        attraction = resolve_attraction(name, city)
        if attraction:
            points.append((attraction.lng, attraction.lat))

    span = 0.0
    for index, (lng1, lat1) in enumerate(points):
        for lng2, lat2 in points[index + 1:]:
            span = max(span, haversine_km(lng1, lat1, lng2, lat2))
    return span


def _start_date(value) -> date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m月%d日"):
        try:
            return datetime.strptime(str(value), fmt).date()
        except ValueError:
            continue
    return None


def _items_minutes(day: dict) -> int:
    return int(sum(int(item.get("duration_min") or 120) for item in day.get("items", [])) + 120)


def after_validate(state: PlanGraphState) -> str:
    """硬错误且还有额度 → 回退重排；否则渲染。"""
    errors = state.get("validate_errors") or []
    retry = state.get("validate_retry") or 0
    if errors and retry <= trip_config.plan_max_retry and retry > 0:
        return "regenerate"
    return "render"
