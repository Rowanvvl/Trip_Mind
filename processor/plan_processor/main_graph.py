"""行程图装配。

```
START → node_constraint_parse → node_retrieve_by_day → node_route_cluster
→ node_time_budget → node_plan_generate → node_plan_validate
                                    ▲              │
                                    └──(硬错误回退)─┤
                                                   ▼
                                            node_plan_render → END
```

回退只在**硬规则**违反时发生（跨区、闭馆、超时、超预算），
软提醒不会触发重排——否则会陷入「改完这条坏那条」的震荡。
`plan_max_retry` 默认 2，两次之后即便仍有错误也出稿，并把问题写进 Markdown。
"""
from __future__ import annotations

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from processor.plan_processor.nodes.node_constraint_parse import node_constraint_parse
from processor.plan_processor.nodes.node_plan_generate import node_plan_generate
from processor.plan_processor.nodes.node_plan_render import node_plan_render
from processor.plan_processor.nodes.node_plan_validate import after_validate, node_plan_validate
from processor.plan_processor.nodes.node_retrieve_by_day import node_retrieve_by_day
from processor.plan_processor.nodes.node_route_cluster import node_route_cluster
from processor.plan_processor.nodes.node_time_budget import node_time_budget
from processor.plan_processor.state import PlanGraphState


def build_plan_graph():
    builder = StateGraph(PlanGraphState)

    builder.add_node("node_constraint_parse", node_constraint_parse)
    builder.add_node("node_retrieve_by_day", node_retrieve_by_day)
    builder.add_node("node_route_cluster", node_route_cluster)
    builder.add_node("node_time_budget", node_time_budget)
    builder.add_node("node_plan_generate", node_plan_generate)
    builder.add_node("node_plan_validate", node_plan_validate)
    builder.add_node("node_plan_render", node_plan_render)

    builder.add_edge(START, "node_constraint_parse")
    builder.add_edge("node_constraint_parse", "node_retrieve_by_day")
    builder.add_edge("node_retrieve_by_day", "node_route_cluster")
    builder.add_edge("node_route_cluster", "node_time_budget")
    builder.add_edge("node_time_budget", "node_plan_generate")
    builder.add_edge("node_plan_generate", "node_plan_validate")
    builder.add_conditional_edges(
        "node_plan_validate",
        after_validate,
        {"regenerate": "node_plan_generate", "render": "node_plan_render"},
    )
    builder.add_edge("node_plan_render", END)

    return builder.compile()


plan_app = build_plan_graph()
