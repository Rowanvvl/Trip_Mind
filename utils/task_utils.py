"""任务状态机（单进程内存态）。

与掌柜智库保持同一套接口形状：`add_running_task` / `add_done_task` / `get_task_progress`。
多 worker 部署时换成 Redis 实现即可，调用方不需要改。

任务状态：pending → processing → completed / failed
节点状态：running（进行中）→ done（已完成）

进度百分比的算法是**按该图预期的节点序列**算的，不是按已跑节点数除以总数：
条件分支会跳过节点，用总数算会让进度条永远到不了 100%。
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from utils.sse_utils import push_progress

TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# 三张图的节点序列（也是前端进度条的刻度）
IMPORT_SEQUENCE = [
    "node_entry",
    "node_pdf_to_md",
    "node_md_normalize",
    "node_md_img",
    "node_meta_extract",
    "node_document_split",
    "node_chunk_enrich",
    "node_geo_resolve",
    "node_bge_embedding",
    "node_import_milvus",
]

QUERY_SEQUENCE = [
    "node_input_guard",
    "node_query_understand",
    "node_clarify_answer",
    "node_search_dense",
    "node_search_sparse",
    "node_search_hyde",
    "node_web_search_mcp",
    "node_rrf",
    "node_rerank",
    "node_retrieve_assess",
    "node_sub_query_retrieve",
    "node_context_build",
    "node_route_plan",
    "node_weather",
    "node_budget",
    "node_fact_merge",
    "node_answer_output",
    "node_answer_critic",
    "node_goal_check",
    "node_image_attach",
]

PLAN_SEQUENCE = [
    "node_constraint_parse",
    "node_retrieve_by_day",
    "node_route_cluster",
    "node_time_budget",
    "node_plan_generate",
    "node_plan_validate",
    "node_plan_render",
]

# 节点 ID → 中文展示名。key 必须与 add_node 里的名字完全一致。
_NODE_NAME_TO_CN: Dict[str, str] = {
    # ---- 导入图 ----
    "node_entry": "检查文件",
    "node_pdf_to_md": "PDF 转 Markdown",
    "node_md_normalize": "Markdown 规范化",
    "node_md_img": "图片处理",
    "node_meta_extract": "元数据抽取",
    "node_document_split": "按小节切分",
    "node_chunk_enrich": "切片增强",
    "node_geo_resolve": "地理实体消歧",
    "node_bge_embedding": "向量生成",
    "node_import_milvus": "写入向量库",
    # ---- 查询图 ----
    "node_input_guard": "输入检查",
    "node_guard_reply": "安全回复",
    "node_query_understand": "理解问题",
    "node_clarify_answer": "补充提问",
    "node_search_dense": "语义检索",
    "node_search_sparse": "关键词检索",
    "node_search_hyde": "假设性检索",
    "node_web_search_mcp": "网络检索",
    "node_rrf": "融合排序",
    "node_rerank": "精排",
    "node_retrieve_assess": "证据评估",
    "node_sub_query_retrieve": "子问题补检索",
    "node_context_build": "组装上下文",
    "node_route_plan": "路线规划",
    "node_weather": "天气查询",
    "node_budget": "预算测算",
    "node_fact_merge": "事实汇总",
    "node_answer_output": "生成答案",
    "node_answer_critic": "答案校验",
    "node_goal_check": "目标验收",
    "node_image_attach": "配图",
    # ---- 行程图 ----
    "node_constraint_parse": "解析约束",
    "node_retrieve_by_day": "按天取候选",
    "node_route_cluster": "片区聚类",
    "node_time_budget": "时间校验",
    "node_plan_generate": "生成行程",
    "node_plan_validate": "行程校验",
    "node_plan_render": "渲染行程",
    # ---- 结束 ----
    "__end__": "处理完成",
    "END": "处理完成",
}

# 图名 → 节点序列
_SEQUENCE_BY_GRAPH = {
    "import": IMPORT_SEQUENCE,
    "query": QUERY_SEQUENCE,
    "plan": PLAN_SEQUENCE,
}

_tasks_running: Dict[str, List[str]] = {}
_tasks_done: Dict[str, List[str]] = {}
_tasks_status: Dict[str, str] = {}
_tasks_result: Dict[str, Dict] = {}
_tasks_meta: Dict[str, Dict] = {}
# 图类型，用于算进度
_task_graph: Dict[str, str] = {}


def _ensure(task_id: str, graph: Optional[str] = None) -> None:
    _tasks_running.setdefault(task_id, [])
    _tasks_done.setdefault(task_id, [])
    _tasks_result.setdefault(task_id, {})
    _tasks_meta.setdefault(task_id, {"created_at": time.time(), "error": None})
    _tasks_status.setdefault(task_id, TASK_STATUS_PENDING)
    if graph:
        _task_graph[task_id] = graph


def to_cn(node_name: str) -> str:
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def create_task(task_id: str, graph: str = "query") -> None:
    """新建任务并把状态置为 processing。"""
    _ensure(task_id, graph)
    _tasks_status[task_id] = TASK_STATUS_PROCESSING


def add_running_task(task_id: str, node_name: str) -> None:
    _ensure(task_id)
    running = _tasks_running[task_id]
    if node_name not in running:
        running.append(node_name)


def add_done_task(task_id: str, node_name: str) -> None:
    _ensure(task_id)
    running = _tasks_running[task_id]
    if node_name in running:
        running.remove(node_name)
    if node_name not in _tasks_done[task_id]:
        _tasks_done[task_id].append(node_name)


def push_node_running(task_id: str, node_name: str) -> None:
    """节点开始：登记 + 推送进度事件。由 `node_log` 装饰器自动调用。"""
    add_running_task(task_id, node_name)
    percent = compute_percent(task_id, node_name)
    push_progress(task_id, node_name, to_cn(node_name), percent, status="processing")


def push_node_done(task_id: str, node_name: str) -> None:
    """节点结束：登记 + 推送完成事件。节点自己调用，因为只有节点知道它跑完了。"""
    add_done_task(task_id, node_name)
    percent = compute_percent(task_id, node_name)
    push_progress(task_id, node_name, to_cn(node_name), percent, status="done")


def compute_percent(task_id: str, node_name: str) -> int:
    """按图序列计算进度。不在序列里的节点不改变进度。"""
    graph = _task_graph.get(task_id, "query")
    sequence = _SEQUENCE_BY_GRAPH.get(graph, QUERY_SEQUENCE)
    if node_name not in sequence:
        return 0
    index = sequence.index(node_name)
    return int(round((index + 1) / len(sequence) * 100))


def set_task_status(task_id: str, status: str) -> None:
    _ensure(task_id)
    _tasks_status[task_id] = status
    if status in (TASK_STATUS_COMPLETED, TASK_STATUS_FAILED):
        _tasks_running[task_id].clear()
        _tasks_meta[task_id]["finished_at"] = time.time()


def set_task_result(task_id: str, result: Dict) -> None:
    _ensure(task_id)
    _tasks_result[task_id].update(result or {})


def set_task_error(task_id: str, message: str) -> None:
    _ensure(task_id)
    _tasks_meta[task_id]["error"] = message
    set_task_status(task_id, TASK_STATUS_FAILED)


def get_task_status(task_id: str) -> str:
    return _tasks_status.get(task_id, TASK_STATUS_PENDING)


def get_task_result(task_id: str) -> Dict:
    return _tasks_result.get(task_id, {})


def get_task_progress(task_id: str) -> Dict:
    """给 `/status` 用的完整快照。"""
    _ensure(task_id)
    graph = _task_graph.get(task_id, "query")
    sequence = _SEQUENCE_BY_GRAPH.get(graph, QUERY_SEQUENCE)
    done = [n for n in _tasks_done[task_id] if n in sequence]
    percent = int(round(len(done) / len(sequence) * 100)) if sequence else 0
    status = _tasks_status.get(task_id, TASK_STATUS_PENDING)
    if status == TASK_STATUS_COMPLETED:
        percent = 100
    return {
        "task_id": task_id,
        "graph": graph,
        "status": status,
        "percent": percent,
        "running": [to_cn(n) for n in _tasks_running[task_id]],
        "done": [to_cn(n) for n in _tasks_done[task_id]],
        "error": _tasks_meta[task_id].get("error"),
        "created_at": _tasks_meta[task_id].get("created_at"),
        "finished_at": _tasks_meta[task_id].get("finished_at"),
        "result": _tasks_result.get(task_id, {}),
    }


def is_task_finished(task_id: str) -> bool:
    return get_task_status(task_id) in (TASK_STATUS_COMPLETED, TASK_STATUS_FAILED)


def cleanup(max_age_seconds: int = 86400) -> int:
    """清理超期任务，避免内存里堆历史。返回清理条数。"""
    now_ts = time.time()
    stale = [
        tid
        for tid, meta in _tasks_meta.items()
        if is_task_finished(tid) and now_ts - meta.get("created_at", now_ts) > max_age_seconds
    ]
    for tid in stale:
        for bucket in (_tasks_running, _tasks_done, _tasks_status, _tasks_result, _tasks_meta, _task_graph):
            bucket.pop(tid, None)
    return len(stale)


def reset() -> None:
    """测试用。"""
    for bucket in (_tasks_running, _tasks_done, _tasks_status, _tasks_result, _tasks_meta, _task_graph):
        bucket.clear()


__all__ = [
    "TASK_STATUS_PENDING",
    "TASK_STATUS_PROCESSING",
    "TASK_STATUS_COMPLETED",
    "TASK_STATUS_FAILED",
    "IMPORT_SEQUENCE",
    "QUERY_SEQUENCE",
    "PLAN_SEQUENCE",
    "create_task",
    "add_running_task",
    "add_done_task",
    "push_node_running",
    "push_node_done",
    "compute_percent",
    "set_task_status",
    "set_task_result",
    "set_task_error",
    "get_task_status",
    "get_task_result",
    "get_task_progress",
    "is_task_finished",
    "to_cn",
    "cleanup",
    "reset",
]
