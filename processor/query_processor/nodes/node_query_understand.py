"""问题理解节点。

**规则先行，模型兜底**：
- 城市、天数、人数、预算、忌口这些显式条件用正则抽，稳定且零成本；
- 改写、意图、tool_flags、goal_spec 交给模型（`query_understand.prompt`）；
- 模型不可用时也有完整兜底：规则抽取 + 关键词判意图，图照样能跑。

设计上刻意不让模型决定「要不要调工具」之外的事——工具开关错了只是慢一点，
城市/类型判错则是答非所问，所以城市与类型都再过一道规则校验。
"""
from __future__ import annotations

from common.config import INTENT_KEYWORDS, lm_config
from common.logging.logger import logger, node_log
from processor.query_processor.state import QueryGraphState
from utils.format_utils import now
from utils.geo.city_gazetteer import find_city_in_text, find_cities_in_text, normalize_city
from utils.load_prompt import render_safe
from utils.lm.lm_utils import chat_json, record_usage
from utils.memory.profile_utils import profile_to_text
from utils.memory.slot_utils import (
    extract_slots_from_query,
    inherit_from_history,
    merge_slots,
    slots_to_text,
)
from utils.trace_utils import record_route


@node_log("node_query_understand")
def node_query_understand(state: QueryGraphState) -> dict:
    """改写、抽槽位、判意图与工具开关。

    输出：`rewritten_query / city / cities / content_type / entities / intent /
          tool_flags / slots / profile / need_clarify / clarify_question / goal_spec`
    """
    query = state.get("original_query") or ""
    history = state.get("history") or []
    trace_id = state.get("trace_id", "")

    # ---------- 规则层：先拿到最有把握的部分 ----------
    rule_slots = extract_slots_from_query(query)
    city_from_rules = rule_slots.get("city") or find_city_in_text(query) or ""
    cities = find_cities_in_text(query)

    # 槽位：历史继承 + 本轮覆盖
    slots = merge_slots(state.get("slots") or {}, rule_slots, source="rule")
    slots = inherit_from_history(slots, history)

    fallback = {
        "rewritten_query": query,
        "city": city_from_rules,
        "cities": cities,
        "content_type": _guess_content_type(query),
        "entities": [],
        "intent": "",
        "tool_flags": _guess_tool_flags(query),
        "slots": {},
        "need_clarify": False,
        "clarify_question": "",
        "goal_spec": {"must_cover": [], "min_citations": 1, "needs_fresh_facts": False},
        "reason": "rule_fallback",
    }

    data = fallback
    if lm_config.enabled:
        prompt = render_safe(
            "query_understand",
            "",
            today=now().strftime("%Y-%m-%d"),
            slots=slots_to_text(slots),
            profile=profile_to_text(state.get("profile") or {}),
            history=_history_text(history),
            query=query,
        )
        parsed, llm_result = chat_json(prompt, tier="light", fallback=fallback)
        record_usage(trace_id, llm_result)
        if llm_result.ok and parsed:
            data = {**fallback, **parsed}
        else:
            logger.warning("[query_understand] 模型不可用，使用规则兜底")

    # ---------- 合并与校验 ----------
    city = _resolve_city(data.get("city"), slots.get("city"), city_from_rules, cities)
    content_type = _validate_content_type(data.get("content_type"), query)

    if city:
        slots = merge_slots(slots, {"city": city}, source="understand")
    if data.get("slots"):
        slots = merge_slots(slots, data["slots"], source="model")

    tool_flags = _merge_tool_flags(data.get("tool_flags") or {}, query)
    need_clarify = bool(data.get("need_clarify")) and not city

    goal_spec = data.get("goal_spec") or {}
    goal_spec.setdefault("min_citations", 3 if content_type in ("综合", "线路推荐") else 1)
    goal_spec.setdefault("must_cover", [])

    rewritten = (data.get("rewritten_query") or query).strip()
    if city and city not in rewritten:
        rewritten = f"{city}：{rewritten}"

    record_route(
        trace_id,
        city=city,
        content_type=content_type,
        intent=data.get("intent", ""),
        need_clarify=need_clarify,
        tool_flags=tool_flags,
        reason=data.get("reason", ""),
        rewritten=rewritten,
    )

    return {
        "rewritten_query": rewritten,
        "city": city,
        "cities": cities or ([city] if city else []),
        "content_type": content_type,
        "entities": list(data.get("entities") or []),
        "intent": data.get("intent", ""),
        "tool_flags": tool_flags,
        "slots": slots,
        "need_clarify": need_clarify,
        "clarify_question": data.get("clarify_question", "") or "",
        "goal_spec": goal_spec,
    }


def _history_text(history: list, limit: int = 3) -> str:
    if not history:
        return "（无）"
    lines = []
    for item in history[-limit * 2 :]:
        role = "用户" if item.get("role") == "user" else "助手"
        lines.append(f"{role}：{item.get('text', '')[:120]}")
    return "\n".join(lines)


def _resolve_city(*candidates) -> str:
    """按可信度顺序取第一个有效城市：模型 > 槽位 > 规则 > 多城市列表。"""
    for value in candidates:
        if not value:
            continue
        if isinstance(value, list):
            if value:
                return normalize_city(value[0])
            continue
        normalized = normalize_city(value)
        if normalized:
            return normalized
    return ""


def _guess_content_type(query: str) -> str:
    for content_type, keywords in INTENT_KEYWORDS.items():
        if any(kw in query for kw in keywords):
            return content_type
    return "综合"


def _validate_content_type(value, query: str) -> str:
    allowed = {"景点攻略", "线路推荐", "酒店信息", "美食推荐", "交通指南", "综合"}
    if value in allowed:
        return value if value else "综合"
    guessed = _guess_content_type(query)
    return guessed


def _guess_tool_flags(query: str) -> dict:
    """规则判工具开关。宁可少开，也不要为了保险全开。"""
    route_words = ["怎么去", "怎么走", "多久", "多远", "顺路", "打车", "地铁", "车程", "自驾"]
    weather_words = ["天气", "下雨", "会不会冷", "冷不冷", "穿什么", "台风", "能下海"]
    budget_words = ["预算", "花多少", "多少钱", "贵不贵", "费用", "花费"]
    return {
        "route": any(w in query for w in route_words),
        "weather": any(w in query for w in weather_words),
        "budget": any(w in query for w in budget_words),
    }


def _merge_tool_flags(model_flags: dict, query: str) -> dict:
    """模型与规则取并集，但只有模型能关闭规则误判（例如「门票多少钱」不是交通问题）。"""
    rule_flags = _guess_tool_flags(query)
    merged = {
        "route": bool(model_flags.get("route")) or rule_flags["route"],
        "weather": bool(model_flags.get("weather")) or rule_flags["weather"],
        "budget": bool(model_flags.get("budget")) or rule_flags["budget"],
    }
    # 纯门票类问题不该触发预算测算（那会给出一个莫名其妙的总额）
    if "门票" in query and "预算" not in query and "花多少" not in query:
        merged["budget"] = False
    return merged
