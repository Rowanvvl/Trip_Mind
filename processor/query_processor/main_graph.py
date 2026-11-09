"""查询图装配。

```
START → node_input_guard ─┬─(拦截)→ node_guard_reply → END
                          └─(放行)→ node_query_understand ─┬─(缺条件)→ node_clarify_answer → END
                                                             └─(可检索)→ ┌ node_search_dense  ┐
                                                                          ├ node_search_sparse ├→ node_rrf
                                                                          ├ node_search_hyde   │
                                                                          └ node_web_search_mcp┘
                                                                                            ▼
                                                                                      node_rerank
                                                                                            ▼
                                                                                  node_retrieve_assess
                                                                       ┌────(覆盖不足)→ node_sub_query_retrieve
                                                                       │                        │
                                                                       │                  (回到 rerank 重排)
                                                                       └────(够)→ node_context_build
                                                                                            ▼
                                                              ┌── 按 tool_flags 并行，可跳过 ──┐
                                                              ├ node_route_plan  ─┐
                                                              ├ node_weather     ─┼→ node_fact_merge
                                                              └ node_budget      ─┘        │
                                                                                            ▼
                                                                                   node_answer_output
                                                                                            ▼
                                                                                   node_answer_critic
                                                                       ┌──(未通过且可重写)──┘
                                                                       ▼                    │(通过/超额)
                                                              node_answer_output       node_goal_check
                                                                       │                    │
                                                                       │        ┌──(可补救)→ node_rerank
                                                                       │        └──(收尾)→ node_image_attach → END
```

三条回边都有额度上限（`critic_max_retry` / `goal_max_retry` / `sub_query_used` 一次性），
LangGraph 的环在这里是**受控的**，不会出现无限循环。
"""
from __future__ import annotations

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from processor.query_processor.nodes.node_answer_critic import after_critic, node_answer_critic
from processor.query_processor.nodes.node_answer_output import node_answer_output
from processor.query_processor.nodes.node_budget import node_budget
from processor.query_processor.nodes.node_clarify_answer import node_clarify_answer
from processor.query_processor.nodes.node_context_build import node_context_build
from processor.query_processor.nodes.node_fact_merge import node_fact_merge
from processor.query_processor.nodes.node_goal_check import after_goal, node_goal_check
from processor.query_processor.nodes.node_guard_reply import node_guard_reply
from processor.query_processor.nodes.node_image_attach import node_image_attach
from processor.query_processor.nodes.node_input_guard import after_input_guard, node_input_guard
from processor.query_processor.nodes.node_query_understand import node_query_understand
from processor.query_processor.nodes.node_rerank import node_rerank
from processor.query_processor.nodes.node_retrieve_assess import after_assess, node_retrieve_assess
from processor.query_processor.nodes.node_route_plan import node_route_plan
from processor.query_processor.nodes.node_rrf import node_rrf
from processor.query_processor.nodes.node_search_dense import node_search_dense
from processor.query_processor.nodes.node_search_hyde import node_search_hyde
from processor.query_processor.nodes.node_search_sparse import node_search_sparse
from processor.query_processor.nodes.node_sub_query_retrieve import node_sub_query_retrieve
from processor.query_processor.nodes.node_weather import node_weather
from processor.query_processor.nodes.node_web_search_mcp import node_web_search_mcp
from processor.query_processor.state import QueryGraphState


def after_understand(state: QueryGraphState):
    """缺条件 → 反问；否则返回四条并行检索节点。

    这里必须用「返回节点名列表」的写法：条件边与普通边混用会导致
    反问分支也触发并行检索（普通边无条件执行），白跑三次召回。
    """
    if state.get("need_clarify"):
        return "node_clarify_answer"
    return ["node_search_dense", "node_search_sparse", "node_search_hyde", "node_web_search_mcp"]


def fact_fanout(state: QueryGraphState) -> list[str]:
    """按 tool_flags 决定并行几条支路；一个都不需要时直接合并。

    返回 `node_fact_merge` 单元素是刻意的：LangGraph 的条件边返回列表即并行，
    但如果三条都不需要，必须有一条确定的路径到 fact_merge，否则图会断。
    """
    flags = state.get("tool_flags") or {}
    branches = []
    if flags.get("route"):
        branches.append("node_route_plan")
    if flags.get("weather"):
        branches.append("node_weather")
    if flags.get("budget"):
        branches.append("node_budget")
    return branches or ["node_fact_merge"]


def build_query_graph():
    builder = StateGraph(QueryGraphState)

    # 入口与理解
    builder.add_node("node_input_guard", node_input_guard)
    builder.add_node("node_guard_reply", node_guard_reply)
    builder.add_node("node_query_understand", node_query_understand)
    builder.add_node("node_clarify_answer", node_clarify_answer)

    # 检索
    builder.add_node("node_search_dense", node_search_dense)
    builder.add_node("node_search_sparse", node_search_sparse)
    builder.add_node("node_search_hyde", node_search_hyde)
    builder.add_node("node_web_search_mcp", node_web_search_mcp)
    builder.add_node("node_rrf", node_rrf)
    builder.add_node("node_rerank", node_rerank)
    builder.add_node("node_retrieve_assess", node_retrieve_assess)
    builder.add_node("node_sub_query_retrieve", node_sub_query_retrieve)
    builder.add_node("node_context_build", node_context_build)

    # 事实层
    builder.add_node("node_route_plan", node_route_plan)
    builder.add_node("node_weather", node_weather)
    builder.add_node("node_budget", node_budget)
    builder.add_node("node_fact_merge", node_fact_merge)

    # 生成与质量
    builder.add_node("node_answer_output", node_answer_output)
    builder.add_node("node_answer_critic", node_answer_critic)
    builder.add_node("node_goal_check", node_goal_check)
    builder.add_node("node_image_attach", node_image_attach)

    # ---- 边 ----
    builder.add_edge(START, "node_input_guard")
    builder.add_conditional_edges(
        "node_input_guard",
        after_input_guard,
        {"reject": "node_guard_reply", "understand": "node_query_understand"},
    )
    builder.add_edge("node_guard_reply", END)

    builder.add_conditional_edges(
        "node_query_understand",
        after_understand,
        [
            "node_clarify_answer",
            "node_search_dense",
            "node_search_sparse",
            "node_search_hyde",
            "node_web_search_mcp",
        ],
    )
    builder.add_edge("node_clarify_answer", END)

    for name in ("node_search_dense", "node_search_sparse", "node_search_hyde", "node_web_search_mcp"):
        builder.add_edge(name, "node_rrf")

    builder.add_edge("node_rrf", "node_rerank")
    builder.add_edge("node_rerank", "node_retrieve_assess")
    builder.add_conditional_edges(
        "node_retrieve_assess",
        after_assess,
        {"retrieve_more": "node_sub_query_retrieve", "context": "node_context_build"},
    )
    builder.add_edge("node_sub_query_retrieve", "node_rerank")

    builder.add_conditional_edges(
        "node_context_build",
        fact_fanout,
        ["node_route_plan", "node_weather", "node_budget", "node_fact_merge"],
    )
    for name in ("node_route_plan", "node_weather", "node_budget"):
        builder.add_edge(name, "node_fact_merge")

    builder.add_edge("node_fact_merge", "node_answer_output")
    builder.add_edge("node_answer_output", "node_answer_critic")
    builder.add_conditional_edges(
        "node_answer_critic",
        after_critic,
        {"rewrite": "node_answer_output", "goal": "node_goal_check"},
    )
    builder.add_conditional_edges(
        "node_goal_check",
        after_goal,
        {"retry": "node_rerank", "images": "node_image_attach"},
    )
    builder.add_edge("node_image_attach", END)

    return builder.compile()


query_app = build_query_graph()
