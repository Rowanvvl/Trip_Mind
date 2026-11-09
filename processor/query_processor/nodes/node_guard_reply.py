"""拒答节点。

直接短路，不检索、不生成。话术来自护栏（已包含「指向官方渠道」），
再经输出护栏做一次软化与脱敏，保证与正常答案的措辞风格一致。
"""
from __future__ import annotations

from common.logging.logger import node_log
from processor.query_processor.state import QueryGraphState
from utils.guard.output_guard import apply as apply_output_guard
from utils.guard.output_guard import refusal_reply


@node_log("node_guard_reply")
def node_guard_reply(state: QueryGraphState) -> dict:
    """输出拒答话术。

    输出：`answer`、`citations=[]`、`degraded=True`
    """
    guard = state.get("guard_result") or {}
    reply = guard.get("reply") or refusal_reply("这个问题不在我的服务范围内。")
    guarded = apply_output_guard(reply, add_disclaimer=False)

    return {
        "answer": guarded.text,
        "citations": [],
        "facts": {},
        "images": [],
        "degraded": True,
    }
