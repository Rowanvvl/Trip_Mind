"""Markdown 规范化。

清洗动作只做「不改变语义」的几件事，任何会改动措辞的操作都不在这里做——
语料里的判断句是知识本身，改写等于伪造来源。
"""
from __future__ import annotations

import re

from common.logging.logger import node_log
from processor.import_processor.state import ImportGraphState

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_FRONT_MATTER = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


@node_log("node_md_normalize")
def node_md_normalize(state: ImportGraphState) -> dict:
    """规范化换行、去 BOM、去 HTML 注释、压缩多余空行。

    输出：`markdown`、`stage`、`stats.normalize`
    """
    raw = state.get("markdown") or ""
    if not raw.strip():
        return {"error": "规范化前内容为空", "final_status": "failed", "stage": "md_normalize"}

    text = raw.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    text = _FRONT_MATTER.sub("", text, count=1)
    text = _HTML_COMMENT.sub("", text)
    text = _TRAILING_SPACE.sub("\n", text)
    text = _MULTI_BLANK.sub("\n\n", text).strip() + "\n"

    headings = sum(1 for line in text.splitlines() if line.startswith("#"))
    return {
        "markdown": text,
        "stage": "md_normalize",
        "stats": {
            **(state.get("stats") or {}),
            "normalize": {"chars": len(text), "lines": text.count("\n"), "headings": headings},
        },
    }
