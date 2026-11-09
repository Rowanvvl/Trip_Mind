"""写入 Milvus。

幂等策略：**先按 source_path 删旧切片，再写新的**。
不用 upsert 直接覆盖是因为切片数量会变（改了切分参数后 8 片变 6 片，
剩下 2 片会变成永远删不掉的幽灵数据，检索时冒出来与新版冲突）。

实体表单独写，用 `实体名+城市` 聚合 doc_ids，用于消歧与候选扩展。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from common.config import milvus_config
from common.logging.logger import node_log
from processor.import_processor.state import ImportGraphState
from utils.clients import milvus_utils


@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> dict:
    """删除旧切片 → 写入新切片 → 写实体表。

    输出：`milvus_result`、`final_status`、`stage`、`stats.milvus`
    """
    chunks: List[dict] = list(state.get("chunks") or [])
    if not chunks:
        return {"error": "无切片可写库", "final_status": "failed", "stage": "import_milvus"}

    if not milvus_utils.available():
        return {
            "error": "Milvus 不可用，导入未完成（切片已生成，可稍后重试本步骤）",
            "final_status": "failed",
            "stage": "import_milvus",
            "stats": {**(state.get("stats") or {}), "milvus": {"rows": len(chunks), "written": 0}},
        }

    milvus_utils.ensure_collections()

    source_path = state.get("source_path") or ""
    delete_result = milvus_utils.delete_by_source(source_path) if source_path else {"ok": True}

    rows = [_to_row(chunk) for chunk in chunks]
    write_result = milvus_utils.upsert_chunks(rows)

    entity_rows = _aggregate_entities(state.get("entities") or [])
    entity_result = milvus_utils.upsert_entities(entity_rows) if entity_rows else {"ok": True, "written": 0}

    ok = write_result.get("ok") and delete_result.get("ok")

    return {
        "milvus_result": {
            "deleted": delete_result,
            "chunks": write_result,
            "entities": entity_result,
        },
        "final_status": "success" if ok else "failed",
        "error": "" if ok else write_result.get("error", ""),
        "stage": "import_milvus",
        "stats": {
            **(state.get("stats") or {}),
            "milvus": {
                "rows": len(rows),
                "written": write_result.get("written", 0),
                "entities": entity_result.get("written", 0),
                "degraded_embedding": (state.get("stats") or {}).get("embedding", {}).get("degraded", False),
            },
        },
    }


def _to_row(chunk: dict) -> dict:
    """只保留 schema 里声明过的字段，多余字段会被 Milvus 拒绝。"""
    return {
        "chunk_id": chunk.get("chunk_id"),
        "doc_id": int(chunk.get("doc_id") or 0),
        "chunk_index": int(chunk.get("chunk_index") or 0),
        "doc_title": chunk.get("doc_title", "")[:200],
        "section": chunk.get("section", "")[:200],
        "city": chunk.get("city", "")[:32],
        "province": chunk.get("province", "")[:32],
        "content_type": chunk.get("content_type", "")[:32],
        "district": chunk.get("district", "")[:32],
        "entities": list(chunk.get("entities") or [])[:16],
        "source_path": chunk.get("source_path", "")[:512],
        "updated_at": chunk.get("updated_at", "")[:32],
        "text": chunk.get("text", "")[:8192],
        milvus_config.dense_field: chunk.get(milvus_config.dense_field) or [],
        milvus_config.sparse_field: chunk.get(milvus_config.sparse_field) or {},
    }


def _aggregate_entities(rows: List[dict]) -> List[dict]:
    """按 (name, city) 聚合 doc_ids，写 entity_id 保证幂等。"""
    grouped: Dict[tuple, dict] = defaultdict(lambda: {"doc_ids": set(), "district": "", "types": set()})
    for row in rows:
        key = (row.get("name"), row.get("city"))
        if not key[0]:
            continue
        entry = grouped[key]
        if row.get("doc_id") is not None:
            entry["doc_ids"].add(int(row["doc_id"]))
        entry["district"] = entry["district"] or row.get("district", "")
        if row.get("content_type"):
            entry["types"].add(row["content_type"])

    out = []
    for (name, city), entry in grouped.items():
        out.append(
            {
                "entity_id": f"{city}:{name}",
                "name": name,
                "alias": name,
                "city": city,
                "district": entry["district"],
                "entity_type": next(iter(entry["types"]), "景点"),
                "doc_ids": sorted(entry["doc_ids"])[:64],
            }
        )
    return out
