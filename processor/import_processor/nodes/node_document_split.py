"""按小节切分。

**这是本项目效果上限所在**。语料是判断式长文（「东线和西线不在一个方向，一天跑完只能两头都浅」），
如果按固定字数硬切，结论句会被腰斩，检索出来的切片读起来像断章取义，
生成阶段就只能靠模型脑补——这是所有 RAG 项目最常见的隐形质量问题。

所以切分策略是：
1. 一级标题是文档名，不切；
2. **按 `## ` 切小节**，小节内的 `###` 保留；
3. `## 元数据` 单独剔除（它是元信息，不是知识）；
4. 小节超过 1.2×chunk_size 时才做句级滑窗，且**只在句子边界切**。
"""
from __future__ import annotations

import re
from typing import List

from common.config import trip_config
from common.logging.logger import node_log
from processor.import_processor.state import ImportGraphState

_H2 = re.compile(r"^##\s+(?!#)(.+)$", re.MULTILINE)
_SENTENCE_END = re.compile(r"(?<=[。！？；!?;])")


@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> dict:
    """按 `##` 切小节。

    输出：`sections`（[{heading, level, text, index}]）、`stage`、`stats.split`
    """
    text = state.get("markdown") or ""
    if not text.strip():
        return {"error": "切分前内容为空", "final_status": "failed", "stage": "document_split"}

    # 去掉文档级标题行，文档名单独存进 meta，不重复进切片正文
    body = re.sub(r"^#\s+.+$", "", text, count=1, flags=re.MULTILINE)

    matches = list(_H2.finditer(body))
    raw_sections: List[dict] = []

    if not matches:
        # 没有 ## 的文档：整篇当一个 section，靠句级滑窗切
        raw_sections.append({"heading": state.get("meta", {}).get("doc_title", "全文"), "text": body.strip()})
    else:
        for index, match in enumerate(matches):
            heading = match.group(1).strip()
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
            raw_sections.append({"heading": heading, "text": body[start:end].strip()})

    sections: List[dict] = []
    for item in raw_sections:
        heading = item["heading"]
        content = item["text"]
        if heading in {"元数据", "metadata"}:
            continue
        if not content:
            continue
        if len(content) <= trip_config.chunk_size * 1.2:
            sections.append({"heading": heading, "level": 2, "text": content, "index": len(sections)})
            continue
        for part in _split_long(content):
            sections.append({"heading": heading, "level": 2, "text": part, "index": len(sections)})

    if not sections:
        return {"error": "切分后无有效小节", "final_status": "failed", "stage": "document_split"}

    return {
        "sections": sections,
        "stage": "document_split",
        "stats": {
            **(state.get("stats") or {}),
            "split": {"sections": len(sections), "avg_chars": round(sum(len(s["text"]) for s in sections) / len(sections))},
        },
    }


def _split_long(text: str) -> List[str]:
    """句级滑窗。只在句末标点处断开，overlap 以句子为单位。"""
    sentences = [s for s in _SENTENCE_END.split(text) if s.strip()]
    if not sentences:
        return [text]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sentence in sentences:
        length = len(sentence)
        if current and current_len + length > trip_config.chunk_size:
            chunks.append("".join(current).strip())
            # 保留最后 1~2 句作为 overlap
            overlap: List[str] = []
            overlap_len = 0
            for prev in reversed(current):
                if overlap_len + len(prev) > trip_config.chunk_overlap:
                    break
                overlap.insert(0, prev)
                overlap_len += len(prev)
            current = overlap
            current_len = overlap_len
        current.append(sentence)
        current_len += length
    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if len(c) >= trip_config.min_chunk_len] or [text]
