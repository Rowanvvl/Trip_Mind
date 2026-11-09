"""信息不足时反问。

只在**真的缺关键条件**时走这条路（缺城市、线路类问题缺天数）。
能给出通用建议的问题不要反问——用户宁可要一个"通用但有用"的回答，
也不愿意先填一张表单。

反问后直接结束本轮，不再检索：这是刻意省掉的一次完整检索开销。
"""
from __future__ import annotations

from common.config import lm_config
from common.logging.logger import logger, node_log
from processor.query_processor.state import QueryGraphState
from utils.load_prompt import render_safe
from utils.lm.lm_utils import chat, record_usage
from utils.memory.slot_utils import missing_slots, slots_to_text

_FALLBACK = "想去哪座城市、玩几天？给我这两个条件就能排具体安排。"


@node_log("node_clarify_answer")
def node_clarify_answer(state: QueryGraphState) -> dict:
    """生成一句追问。

    输出：`answer`、`citations=[]`、`degraded=True`
    """
    query = state.get("original_query") or ""
    slots = state.get("slots") or {}
    missing = missing_slots(slots, ("city", "days"))

    question = state.get("clarify_question") or ""
    if not question:
        question = _FALLBACK
        if lm_config.enabled:
            prompt = render_safe(
                "clarify",
                _FALLBACK,
                query=query,
                slots=slots_to_text(slots),
                missing="、".join(missing) or "城市或天数",
            )
            messages = [{"role": "user", "content": prompt}]
            result = chat(messages, tier="light", temperature=0.3)
            record_usage(state.get("trace_id", ""), result)
            if result.ok:
                question = result.text.strip().splitlines()[0][:60]

    logger.info(f"[clarify_answer] 反问：{question}")
    return {"answer": question, "citations": [], "degraded": True}
