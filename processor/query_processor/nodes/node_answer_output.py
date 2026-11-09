"""生成答案。

三条路径：
1. 同步：`chat()` 一次拿全；
2. 流式：`chat_stream()` 逐段推 SSE `delta`，首字延迟是这条链路的唯一指标；
3. **模型不可用**：抽取式兜底——把命中的切片按引用编号列出来，
   并明确说明"以下为知识库原文摘录"。这不是"优雅降级"，但比返回空字符串诚实。

输出护栏（脱敏、软化绝对表述、补声明）在三条路径上都执行，
唯一例外是流式：流式时护栏只处理 `final` 事件的全文，delta 直接透传，
否则会把已经吐出去的字再改一遍，前端出现"重影"。
"""
from __future__ import annotations

from common.config import lm_config, trip_config
from common.logging.logger import logger, node_log
from processor.query_processor.state import QueryGraphState
from utils.guard.output_guard import apply as apply_output_guard
from utils.load_prompt import render_safe
from utils.lm.lm_utils import chat, chat_stream, record_usage
from utils.memory.profile_utils import profile_to_text
from utils.memory.slot_utils import slots_to_text
from utils.sse_utils import push_delta


@node_log("node_answer_output")
def node_answer_output(state: QueryGraphState) -> dict:
    """组装 prompt 并生成。

    输出：`answer`、`answer_raw`、`prompt`、`degraded`
    """
    query = state.get("original_query") or ""
    critique_text = _critique_text(state.get("critique") or {})

    prompt = render_safe(
        "answer_out",
        "",
        query=query,
        slots=slots_to_text(state.get("slots") or {}),
        profile=profile_to_text(state.get("profile") or {}),
        history=_history_text(state.get("history") or []),
        context=state.get("context") or "（无资料）",
        facts=(state.get("facts") or {}).get("text") or "（无实时数据）",
        critique=critique_text or "（无）",
    )

    if not lm_config.enabled:
        answer = _extractive_fallback(state)
        return {"answer": answer, "answer_raw": answer, "prompt": prompt, "degraded": True}

    session_id = state.get("session_id") or ""
    is_stream = bool(state.get("is_stream")) and bool(session_id)

    if is_stream:
        chunks = []
        for delta in chat_stream([{"role": "user", "content": prompt}], tier="main"):
            chunks.append(delta)
            push_delta(session_id, delta)
        raw = "".join(chunks).strip()
        if not raw:
            raw = _extractive_fallback(state)
            push_delta(session_id, raw)
        # 流式已发出增量，这里只做最终整文的护栏处理
        guarded = apply_output_guard(raw, _flags(state))
        return {"answer": guarded.text, "answer_raw": raw, "prompt": prompt, "degraded": bool(state.get("degraded"))}

    result = chat([{"role": "user", "content": prompt}], tier="main", temperature=lm_config.answer_temperature)
    record_usage(state.get("trace_id", ""), result)

    if not result.ok:
        logger.warning(f"[answer_output] 生成失败，走抽取式兜底：{result.error}")
        answer = _extractive_fallback(state)
        return {"answer": answer, "answer_raw": answer, "prompt": prompt, "degraded": True}

    guarded = apply_output_guard(result.text, _flags(state))
    return {"answer": guarded.text, "answer_raw": result.text, "prompt": prompt, "degraded": bool(state.get("degraded"))}


def _flags(state: QueryGraphState) -> list[str]:
    return list((state.get("guard_result") or {}).get("flags") or [])


def _history_text(history: list, limit: int | None = None) -> str:
    if not history:
        return "（无）"
    limit = limit or trip_config.history_keep_turns
    lines = []
    for item in history[-limit * 2 :]:
        role = "用户" if item.get("role") == "user" else "助手"
        lines.append(f"{role}：{item.get('text', '')[:150]}")
    return "\n".join(lines)


def _critique_text(critique: dict) -> str:
    issues = critique.get("issues") or []
    if not issues:
        return ""
    lines = []
    for issue in issues:
        lines.append(f"- [{issue.get('type')}] {issue.get('detail')} → {issue.get('fix')}")
    return "\n".join(lines)


def _extractive_fallback(state: QueryGraphState) -> str:
    """模型不可用时的抽取式答案：只列原文，不做推断。"""
    citations = state.get("citations") or []
    if not citations:
        return (
            "知识库里没有找到与这个问题直接相关的内容。"
            "可以换个问法，或者告诉我具体城市与天数，我再帮你查。"
        )

    lines = ["当前未接入生成模型，下面是知识库中与问题最相关的原文摘录："]
    for item in citations[:4]:
        lines.append(f"[{item['index']}] {item.get('doc_title')} · {item.get('section')}")
    facts_text = (state.get("facts") or {}).get("text")
    if facts_text and not (state.get("facts") or {}).get("degraded"):
        lines.append("")
        lines.append(facts_text)
    lines.append("")
    lines.append("（配置 OPENAI_API_KEY 后可获得完整的分析式回答。）")
    return "\n".join(lines)
