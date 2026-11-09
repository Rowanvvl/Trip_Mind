"""导入图入口：检查文件是否可以导入。

只做校验，不做解析。校验失败时把 `final_status` 置为 failed 并短路，
避免后续节点在空内容上继续跑（那样日志里会全是"切出 0 个切片"这种噪音）。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from common.logging.logger import node_log
from processor.import_processor.state import ImportGraphState

ALLOWED_EXT = {".md", ".markdown", ".pdf"}


@node_log("node_entry")
def node_entry(state: ImportGraphState) -> dict:
    """校验路径与扩展名，并把 Markdown 原文读进 state。

    输出：`markdown`（.md 时）、`file_ext`、`stage`、`stats.entry`
    """
    raw_path = state.get("source_path") or ""
    path = Path(raw_path)

    if not raw_path:
        return {"error": "source_path 为空", "final_status": "failed", "stage": "entry"}

    if not path.exists():
        return {"error": f"文件不存在：{path}", "final_status": "failed", "stage": "entry"}

    if path.is_dir():
        return {"error": f"需要具体文件而不是目录：{path}", "final_status": "failed", "stage": "entry"}

    ext = path.suffix.lower()
    if ext not in ALLOWED_EXT:
        return {
            "error": f"不支持的扩展名 {ext}，仅支持 {sorted(ALLOWED_EXT)}",
            "final_status": "failed",
            "stage": "entry",
        }

    payload: dict = {
        "file_name": path.name,
        "file_ext": ext,
        "stage": "entry",
        "stats": {**(state.get("stats") or {}), "entry": {"size_kb": round(path.stat().st_size / 1024, 1)}},
    }

    if ext in {".md", ".markdown"}:
        try:
            payload["markdown"] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            payload["markdown"] = path.read_text(encoding="utf-8", errors="ignore")
        if not payload["markdown"].strip():
            return {**payload, "error": "文件内容为空", "final_status": "failed"}

    return payload


def after_entry(state: ImportGraphState) -> str | List[str]:
    """入口后的分支：失败直接结束，PDF 走转换，Markdown 走规范化。"""
    if state.get("final_status") == "failed" or state.get("error"):
        return "end"
    return "node_pdf_to_md" if state.get("file_ext") == ".pdf" else "node_md_normalize"
