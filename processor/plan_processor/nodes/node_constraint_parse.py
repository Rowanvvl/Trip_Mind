"""行程约束解析。

规则优先（显式条件）+ 模型补充（隐含偏好）。所有数值都过 `param_guard`，
非法值直接报错而不是取默认——「排一个 30 天的行程」这种输入应该在入口被挡住，
不是排到一半崩掉。
"""
from __future__ import annotations

from common.config import lm_config
from common.logging.logger import logger, node_log
from processor.plan_processor.state import PlanGraphState
from processor.query_processor.nodes.node_query_understand import _history_text
from utils.guard.param_guard import ParamError, validate_budget, validate_city, validate_days, validate_pace
from utils.load_prompt import render_safe
from utils.lm.lm_utils import chat_json, record_usage
from utils.memory.slot_utils import extract_slots_from_query, merge_slots, slots_to_text


@node_log("node_constraint_parse")
def node_constraint_parse(state: PlanGraphState) -> dict:
    """把自然语言约束变成结构化字段。

    输出：`city / days / budget / travelers / pace / companions / must_go / exclude / constraints`
    """
    query = state.get("raw_query") or ""
    slots = merge_slots(state.get("slots") or {}, extract_slots_from_query(query), source="plan_rule")

    fallback = {
        "city": slots.get("city") or state.get("city") or "",
        "days": slots.get("days") or state.get("days") or 3,
        "budget": slots.get("budget") or state.get("budget") or 0,
        "travelers": slots.get("travelers") or state.get("travelers") or 2,
        "pace": slots.get("pace") or "适中",
        "companions": slots.get("companions") or [],
        "must_go": slots.get("must_go") or [],
        "exclude": slots.get("exclude") or [],
    }

    data = fallback
    if lm_config.enabled and query:
        prompt = render_safe(
            "query_understand",
            "",
            today="",
            slots=slots_to_text(slots),
            profile="（略）",
            history=_history_text(state.get("history") or []),
            query=f"（这是行程规划请求，请重点抽取约束）{query}",
        )
        parsed, result = chat_json(prompt, tier="light", fallback=fallback)
        record_usage(state.get("trace_id", ""), result)
        if result.ok:
            parsed_slots = parsed.get("slots") or {}
            data = {
                **fallback,
                "city": parsed.get("city") or fallback["city"],
                "days": parsed_slots.get("days") or fallback["days"],
                "budget": parsed_slots.get("budget") or fallback["budget"],
                "travelers": parsed_slots.get("travelers") or fallback["travelers"],
                "pace": parsed_slots.get("pace") or fallback["pace"],
                "companions": parsed_slots.get("companions") or fallback["companions"],
                "must_go": parsed_slots.get("must_go") or fallback["must_go"],
                "exclude": parsed_slots.get("exclude") or fallback["exclude"],
            }

    try:
        city = validate_city(data.get("city"), allow_none=False)
        days = validate_days(data.get("days"))
        # 预算可以不提。0 在这里的语义是「用户没说」，不是「用户说了 0 元」，
        # 所以先转成 None 再校验，让 validate_budget 走可选分支；
        # 真正的预算上限由 node_time_budget / budget_utils 估算后给出参考区间。
        raw_budget = data.get("budget")
        budget_input = None if raw_budget in (0, "0", "", None) else raw_budget
        budget = validate_budget(budget_input) or 0.0
        pace = validate_pace(data.get("pace"))
    except ParamError as exc:
        logger.warning(f"[constraint_parse] 约束非法：{exc}")
        return {"error": str(exc), "degraded": True}

    constraints = {
        "companions": data.get("companions") or [],
        "elderly": "老人" in (data.get("companions") or []),
        "with_kids": "小孩" in (data.get("companions") or []),
        "budget_cap": budget,
        "pace": pace,
        "must_go": data.get("must_go") or [],
        "avoid": data.get("exclude") or [],
    }

    return {
        "city": city,
        "days": days,
        "budget": budget,
        "travelers": int(data.get("travelers") or 2),
        "pace": pace,
        "companions": data.get("companions") or [],
        "must_go": data.get("must_go") or [],
        "exclude": data.get("exclude") or [],
        "constraints": constraints,
        "slots": slots,
        "plan_id": state.get("plan_id") or f"plan_{state.get('session_id') or 'anon'}_{days}d",
    }
