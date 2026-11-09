"""导入图状态。

导入是**长任务**，所以 state 里既有数据也有进度信息：
`stage` 记录当前阶段名（写进 `/status`），`stats` 记录每个阶段的产出入库量。
任何节点失败都写到 `error` 并把 stage 停住，不要让后续节点在空数据上继续跑。
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, TypedDict


class ImportGraphState(TypedDict, total=False):
    # ---------- 标识 ----------
    task_id: str
    trace_id: str

    # ---------- 输入 ----------
    source_path: str
    file_name: str
    file_ext: str
    force: bool              # 忽略哈希，强制重导

    # ---------- 解析 ----------
    markdown: str
    markdown_path: str
    images: List[dict]       # {object_name, url, alt}
    meta: Dict[str, Any]     # 元数据块解析结果
    city: str
    content_type: str

    # ---------- 切分 ----------
    sections: List[dict]     # {heading, level, text}
    chunks: List[dict]       # 最终待入库切片
    entities: List[dict]

    # ---------- 索引 ----------
    embeddings_ok: bool
    embedding_backend: str
    milvus_result: dict

    # ---------- 控制 ----------
    stage: str
    stats: Dict[str, Any]
    final_status: str        # success / failed / skipped
    error: str


IMPORT_DEFAULT_STATE: ImportGraphState = {
    "task_id": "",
    "trace_id": "",
    "source_path": "",
    "file_name": "",
    "file_ext": "",
    "force": False,
    "markdown": "",
    "markdown_path": "",
    "images": [],
    "meta": {},
    "city": "",
    "content_type": "",
    "sections": [],
    "chunks": [],
    "entities": [],
    "embeddings_ok": False,
    "embedding_backend": "",
    "milvus_result": {},
    "stage": "init",
    "stats": {},
    "final_status": "",
    "error": "",
}


def create_import_state(**overrides) -> ImportGraphState:
    state = copy.deepcopy(IMPORT_DEFAULT_STATE)
    state.update(overrides)
    if not state.get("file_name") and state.get("source_path"):
        state["file_name"] = Path(state["source_path"]).name
    if not state.get("file_ext") and state.get("source_path"):
        state["file_ext"] = Path(state["source_path"]).suffix.lower()
    return state


def copy_import_state(state: ImportGraphState, **overrides) -> ImportGraphState:
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state
