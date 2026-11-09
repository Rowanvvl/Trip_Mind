"""生成行程。

模型只做「在被砍过的候选里组织叙事与顺序」，不做三件它做不好的事：
- 不在候选外新增点位（prompt 明确禁止）；
- 不算时长（时长已由 `node_time_budget` 给出）；
- 不定片区（片区已由 `node_route_cluster` 给出）。

模型不可用时用规则模板出稿：结构完整、语言朴素，但**可用**。
"""
from __future__ import annotations

from common.config import lm_config
from common.logging.logger import logger, node_log
from processor.plan_processor.state import PlanGraphState
from utils.format_utils import human_duration
from utils.load_prompt import render_safe
from utils.lm.lm_utils import chat_json, record_usage


@node_log("node_plan_generate")
def node_plan_generate(state: PlanGraphState) -> dict:
    """生成结构化行程。

    输出：`plan_json`
    """
    city = state.get("city") or ""
    days = int(state.get("days") or 3)
    budgets = state.get("daily_time_budget") or []
    validate_errors = state.get("validate_errors") or []

    candidates_text = _candidates_text(budgets)
    constraints_text = _constraints_text(state)
    weather_text = _weather_text(state.get("weather_days") or [])

    if not lm_config.enabled:
        return {"plan_json": _template_plan(state, budgets), "degraded": True}

    prompt = render_safe(
        "itinerary_plan",
        "",
        city=city,
        days=days,
        constraints=constraints_text,
        candidates=candidates_text,
        weather=weather_text,
    )
    if validate_errors:
        prompt += "\n\n【上一版被打回的问题，必须逐条修正】\n" + "\n".join(
            f"- 第 {err.get('day', '?')} 天 [{err.get('rule')}] {err.get('detail')}" for err in validate_errors
        )

    data, result = chat_json(prompt, tier="main", temperature=0.3, fallback={})
    record_usage(state.get("trace_id", ""), result)

    if not result.ok or not data.get("days"):
        logger.warning("[plan_generate] 模型输出不可用，改用模板生成")
        return {"plan_json": _template_plan(state, budgets), "degraded": True}

    # 兜底校验：模型返回的点位必须在候选里，多余的直接剔除
    allowed = {name for day in budgets for name in day.get("members", [])}
    cleaned = []
    for day in data.get("days", []):
        items = [it for it in (day.get("items") or []) if it.get("name") in allowed]
        cleaned.append({**day, "items": items})
    data["days"] = cleaned
    data.setdefault("city", city)

    return {"plan_json": data}


def _candidates_text(budgets: list) -> str:
    lines = []
    for day in budgets:
        names = "、".join(day.get("members") or [])
        lines.append(f"第 {day['day']} 天（{day.get('area', '')}，预算 {day.get('total_min')} 分钟）：{names or '（无）'}")
    return "\n".join(lines) if lines else "（无候选）"


def _constraints_text(state: PlanGraphState) -> str:
    constraints = state.get("constraints") or {}
    parts = [
        f"出行人数：{state.get('travelers')} 人",
        f"节奏：{state.get('pace')}",
        f"同行：{'、'.join(state.get('companions') or []) or '未说明'}",
        f"预算上限：{state.get('budget') or '未给'}",
    ]
    if constraints.get("elderly"):
        parts.append("有老人：避免连续爬山、减少换乘")
    if constraints.get("must_go"):
        parts.append(f"必去：{'、'.join(constraints['must_go'])}")
    if constraints.get("avoid"):
        parts.append(f"不去：{'、'.join(constraints['avoid'])}")
    return "\n".join(parts)


def _weather_text(weather_days: list) -> str:
    if not weather_days:
        return "（无天气数据）"
    lines = []
    for item in weather_days:
        cast = item.get("cast") or {}
        flag = "有雨" if item.get("rain") else "无雨"
        lines.append(f"第 {item['day']} 天 {cast.get('date', '')}：{cast.get('day_weather', '未知')}，{flag}")
    return "\n".join(lines)


def _template_plan(state: PlanGraphState, budgets: list) -> dict:
    """规则模板出稿。没有模型时的兜底，保证结构可用。"""
    days_json = []
    for day in budgets:
        members = day.get("members") or []
        items = []
        for index, name in enumerate(members):
            period = "上午" if index == 0 else ("下午" if index == 1 else "傍晚")
            items.append(
                {
                    "period": period,
                    "name": name,
                    "duration_min": 120,
                    "transport": "步行/打车",
                    "note": "",
                }
            )
        days_json.append(
            {
                "day": day["day"],
                "area": day.get("area", ""),
                "theme": f"{day.get('area', '')}片区",
                "items": items,
                "meals": ["午餐：片区内", "晚餐：回住处附近"],
                "rain_plan": "；".join(f"{r['from']}→{r['to'] or '取消'}" for r in day.get("replaced_indoor") or []),
                "time_budget": human_duration(day.get("total_min") or 0),
            }
        )

    return {
        "city": state.get("city"),
        "days": days_json,
        "summary": f"按片区划分为 {len(days_json)} 天，每天集中在一个区域，避免跨区折返。",
        "open_questions": ["未接入生成模型，以上为规则排布，可在此基础上微调"],
    }
