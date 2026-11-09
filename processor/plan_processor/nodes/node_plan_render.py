"""渲染行程。

产出 Markdown：结构固定（每天一节 + 预算 + 提醒），前端直接渲染，
也可以原样复制到聊天里。**不在这里做任何事实补充**——所有内容都来自
`plan_json`、`budget_estimate` 与 `debug.validate`，渲染只负责排版。
"""
from __future__ import annotations

import hashlib
from typing import List

from common.logging.logger import node_log
from processor.plan_processor.state import PlanGraphState
from utils.format_utils import money_range, now_iso
from utils.tools import budget_utils


@node_log("node_plan_render")
def node_plan_render(state: PlanGraphState) -> dict:
    """生成 Markdown 与 plan_id。

    输出：`markdown`、`plan_id`、`budget_estimate`
    """
    plan = state.get("plan_json") or {}
    city = state.get("city") or ""
    days = int(state.get("days") or len(plan.get("days") or []))
    budgets = state.get("daily_time_budget") or []

    estimate = state.get("budget_estimate") or budget_utils.estimate(
        city=city,
        days=days,
        travelers=int(state.get("travelers") or 2),
        level=(state.get("slots") or {}).get("budget_level"),
    )

    lines: List[str] = [f"# {city} {days} 天行程", ""]

    summary = plan.get("summary")
    if summary:
        lines += [summary, ""]

    for day in plan.get("days", []) or []:
        day_no = day.get("day")
        area = day.get("area") or ""
        theme = day.get("theme") or ""
        head = f"## 第 {day_no} 天" + (f" · {area}" if area else "")
        lines.append(head)
        if theme:
            lines.append(f"*{theme}*")
        lines.append("")

        for item in day.get("items", []) or []:
            period = item.get("period", "")
            name = item.get("name", "")
            transport = item.get("transport", "")
            note = item.get("note", "")
            duration = item.get("duration_min")
            bits = [f"- **{period}｜{name}**"]
            if duration:
                bits.append(f"约 {duration} 分钟")
            if transport:
                bits.append(transport)
            lines.append("　".join(bits))
            if note:
                lines.append(f"  - {note}")

        for meal in day.get("meals", []) or []:
            lines.append(f"- {meal}")

        if day.get("rain_plan"):
            lines.append(f"- 雨天备选：{day['rain_plan']}")

        budget = next((b for b in budgets if b.get("day") == day_no), None)
        if budget and budget.get("summary"):
            lines.append(f"- 时间：{budget['summary']}")
        lines.append("")

    # 预算
    low, high = estimate.get("total", [0, 0])
    lines += ["## 预算", f"总花费 {money_range(low, high)}（人均 {estimate.get('per_person', [0, 0])[0]}–"
                        f"{estimate.get('per_person', [0, 0])[1]} 元）", ""]
    for key, label in (("lodging", "住宿"), ("meals", "餐饮"), ("tickets", "门票"),
                       ("local_transport", "市内交通"), ("intercity", "往返大交通")):
        value = (estimate.get("breakdown") or {}).get(key)
        if value and value[1] > 0:
            lines.append(f"- {label}：{value[0]}–{value[1]} 元")
    lines.append("")
    lines.append("测算口径：" + "；".join(estimate.get("assumptions") or []))
    lines.append("")

    # 没排进去的点位
    # 这一点必须写出来：用户点名想去的地方被静默丢掉，比排得不好更让人不信任。
    planned = {item.get("name") for day in (plan.get("days") or []) for item in (day.get("items") or [])}
    dropped = []
    for entry in budgets:
        for name in entry.get("dropped") or []:
            dropped.append(f"- {name}（第 {entry.get('day')} 天候选，当天时间排不下）")
    for candidate in state.get("candidate_pool") or []:
        name = candidate.get("name")
        if name and name not in planned and not any(name in line for line in dropped):
            dropped.append(f"- {name}（在候选里，但天数不够分到某一天）")
    if dropped:
        lines.append("## 没排进去的点位")
        lines.append("以下点位这次没有安排。想去的话，可以加天数、把节奏调慢一档，或者单独说明保留哪一个：")
        lines += dropped
        lines.append("")

    # 校验提醒
    warnings = ((state.get("debug") or {}).get("validate") or {}).get("warnings") or []
    errors = state.get("validate_errors") or []
    if warnings or errors:
        lines.append("## 需要留意")
        for error in errors:
            lines.append(f"- 第 {error.get('day')} 天：{error.get('detail')}（已尽量调整，仍建议人工确认）")
        for warning in warnings:
            lines.append(f"- 第 {warning.get('day')} 天：{warning.get('detail')}")
        lines.append("")

    questions = plan.get("open_questions") or []
    if questions:
        lines.append("## 需要你确认")
        lines += [f"- {q}" for q in questions]
        lines.append("")

    lines.append("> 门票、开放时间与预约规则会调整，出行前请以景区官方公告为准。")

    markdown = "\n".join(lines)
    plan_id = state.get("plan_id") or (
        "plan_" + hashlib.md5(f"{city}{days}{now_iso()}".encode("utf-8")).hexdigest()[:12]
    )

    return {"markdown": markdown, "plan_id": plan_id, "budget_estimate": estimate}
