"""切片增强：给每段切片补元数据与可检索引擎需要的正文。

做三件事：
1. **正文拼上标题路径**（`三亚景点推荐 · 东线与西线`）：BGE-M3 对短文本的语义
   依赖上下文，光有「东线和西线不在一个方向」没有城市，召回时会串到别的城市；
2. 抽实体（景点名）写进切片，供 `array_contains_any` 过滤；
3. 生成稳定的 `chunk_id` 与 `doc_id`：同一切片反复导入必须覆盖而不是新增。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

from common.config import trip_config
from common.logging.logger import node_log
from processor.import_processor.state import ImportGraphState
from utils.format_utils import now_iso
from utils.geo.city_gazetteer import find_attractions_in_text, normalize_city


def _stable_int(text: str, mod: int = 2**31 - 1) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) % mod


@node_log("node_chunk_enrich")
def node_chunk_enrich(state: ImportGraphState) -> dict:
    """生成待入库切片。

    输出：`chunks`、`stage`、`stats.enrich`
    """
    sections = state.get("sections") or []
    meta = state.get("meta") or {}
    source_path = state.get("source_path") or ""
    doc_title = meta.get("doc_title") or Path(source_path).stem
    city = normalize_city(meta.get("city") or state.get("city") or "")
    content_type = meta.get("content_type") or state.get("content_type") or ""
    doc_id = _stable_int(source_path)
    updated_at = now_iso()[:10]
    task_id = state.get("task_id") or "manual"

    chunks: List[dict] = []
    for section in sections:
        heading = section.get("heading", "")
        body = section.get("text", "")
        text = f"{doc_title} · {heading}\n{body}" if heading else f"{doc_title}\n{body}"

        entities = [a.name for a in find_attractions_in_text(body, city)][:16]

        chunks.append(
            {
                "chunk_id": f"{doc_id}-{section.get('index', len(chunks))}",
                "doc_id": doc_id,
                "chunk_index": section.get("index", len(chunks)),
                "doc_title": doc_title,
                "section": heading,
                "city": city,
                "province": "",
                "content_type": content_type,
                "district": "",
                "entities": entities,
                "source_path": source_path,
                "updated_at": updated_at,
                "text": text,
                "task_id": task_id,
            }
        )

    return {
        "chunks": chunks,
        "stage": "chunk_enrich",
        "stats": {
            **(state.get("stats") or {}),
            "enrich": {
                "chunks": len(chunks),
                "with_entities": sum(1 for c in chunks if c["entities"]),
                "avg_chars": round(sum(len(c["text"]) for c in chunks) / max(len(chunks), 1)),
            },
        },
    }
