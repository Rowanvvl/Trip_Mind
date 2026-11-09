"""Milvus 封装：两个集合，一套 schema。

`trip_chunks`（切片）是检索主体，元数据**冗余在切片上**而不是靠 join：
城市、类型、景点、片区、来源路径全部落在切片行里，过滤零关联，这是
「城市/类型过滤」能做到毫秒级的前提。

`trip_entities`（实体）存 city → 实体 → 文档的对应关系，用于
① 同名景点消歧；② 行程候选扩展（某个片区还有哪些点）。

dense 与 sparse 同表两字段，一次查询两个 AnnSearchRequest 用 RRFRanker 融合——
这样就不用自己写一遍 RRF 后再去重，也避免两路召回条数不一致。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from common.config import milvus_config
from common.logging.logger import logger

_client = None
_unavailable_reason = ""


def _reason() -> str:
    return _unavailable_reason or "milvus_unavailable"


def get_client():
    """懒连接。pymilvus 未安装或连不上时返回 None，并把原因记下来。"""
    global _client, _unavailable_reason
    if _client is not None:
        return _client
    try:
        from pymilvus import MilvusClient

        _client = MilvusClient(uri=milvus_config.url, token=milvus_config.token or None, db_name=milvus_config.db_name)
        logger.info(f"[milvus] 已连接 {milvus_config.url}")
    except Exception as exc:
        _unavailable_reason = f"{type(exc).__name__}: {exc}"
        logger.warning(f"[milvus] 连接失败：{_unavailable_reason}")
        _client = None
    return _client


def available() -> bool:
    return get_client() is not None


# ---------------------------------------------------------------- schema
def _chunks_schema():
    from pymilvus import DataType, MilvusClient

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("doc_id", DataType.INT64)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("doc_title", DataType.VARCHAR, max_length=200)
    schema.add_field("section", DataType.VARCHAR, max_length=200)
    schema.add_field("city", DataType.VARCHAR, max_length=32)
    schema.add_field("province", DataType.VARCHAR, max_length=32)
    schema.add_field("content_type", DataType.VARCHAR, max_length=32)
    schema.add_field("district", DataType.VARCHAR, max_length=32)
    schema.add_field("entities", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=16, max_length=64)
    schema.add_field("source_path", DataType.VARCHAR, max_length=512)
    schema.add_field("updated_at", DataType.VARCHAR, max_length=32)
    schema.add_field("text", DataType.VARCHAR, max_length=8192)
    schema.add_field(milvus_config.dense_field, DataType.FLOAT_VECTOR, dim=milvus_config.dense_dim)
    schema.add_field(milvus_config.sparse_field, DataType.SPARSE_FLOAT_VECTOR)
    return schema


def _entities_schema():
    from pymilvus import DataType, MilvusClient

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("entity_id", DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field("name", DataType.VARCHAR, max_length=100)
    schema.add_field("alias", DataType.VARCHAR, max_length=100)
    schema.add_field("city", DataType.VARCHAR, max_length=32)
    schema.add_field("district", DataType.VARCHAR, max_length=32)
    schema.add_field("entity_type", DataType.VARCHAR, max_length=32)
    schema.add_field("doc_ids", DataType.ARRAY, element_type=DataType.INT64, max_capacity=64)
    return schema


def _chunks_index_params():
    from pymilvus import MilvusClient

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name=milvus_config.dense_field, index_type="HNSW", metric_type="COSINE",
                           params={"M": 16, "efConstruction": 200})
    index_params.add_index(field_name=milvus_config.sparse_field, index_type="SPARSE_INVERTED_INDEX",
                           metric_type="IP")
    return index_params


def ensure_collections(recreate: bool = False) -> dict:
    """建集合与索引。已存在则跳过，`recreate=True` 会先删（仅在重建索引时用）。"""
    client = get_client()
    if client is None:
        return {"ok": False, "error": _reason()}

    result = {}
    for name, schema_fn, index_fn in (
        (milvus_config.chunks_collection, _chunks_schema, _chunks_index_params),
        (milvus_config.entities_collection, _entities_schema, lambda: None),
    ):
        try:
            if client.has_collection(name):
                if not recreate:
                    result[name] = "exists"
                    continue
                client.drop_collection(name)
            client.create_collection(collection_name=name, schema=schema_fn(), index_params=index_fn())
            result[name] = "created"
        except Exception as exc:
            logger.exception(f"[milvus] 建集合失败 {name}")
            result[name] = f"error: {exc}"
    return {"ok": True, "collections": result}


# ---------------------------------------------------------------- 写入
def upsert_chunks(rows: Sequence[Dict[str, Any]], batch_size: int = 200) -> dict:
    """写入切片。分批是为了避免单次请求过大，同时让失败可定位。"""
    client = get_client()
    if client is None:
        return {"ok": False, "written": 0, "error": _reason()}
    if not rows:
        return {"ok": True, "written": 0}

    written = 0
    for start in range(0, len(rows), batch_size):
        batch = list(rows[start : start + batch_size])
        try:
            client.upsert(collection_name=milvus_config.chunks_collection, data=batch)
            written += len(batch)
        except Exception as exc:
            logger.exception(f"[milvus] 写入失败，批次 {start}")
            return {"ok": False, "written": written, "error": str(exc)}
    return {"ok": True, "written": written}


def upsert_entities(rows: Sequence[Dict[str, Any]]) -> dict:
    client = get_client()
    if client is None:
        return {"ok": False, "written": 0, "error": _reason()}
    if not rows:
        return {"ok": True, "written": 0}
    try:
        client.upsert(collection_name=milvus_config.entities_collection, data=list(rows))
        return {"ok": True, "written": len(rows)}
    except Exception as exc:
        logger.exception("[milvus] 实体写入失败")
        return {"ok": False, "written": 0, "error": str(exc)}


def delete_by_source(source_path: str) -> dict:
    """按来源文件删除（重新导入前先清旧切片，避免重复）。"""
    client = get_client()
    if client is None:
        return {"ok": False, "error": _reason()}
    try:
        from utils.escape_milvus_string_utils import eq_expr

        client.delete(collection_name=milvus_config.chunks_collection, filter=eq_expr("source_path", source_path))
        return {"ok": True}
    except Exception as exc:
        logger.exception("[milvus] 删除失败")
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------- 检索
def search_dense(
    vector: List[float],
    expr: str = "",
    limit: int | None = None,
    output_fields: Optional[List[str]] = None,
) -> List[dict]:
    client = get_client()
    if client is None or not vector:
        return []
    try:
        hits = client.search(
            collection_name=milvus_config.chunks_collection,
            data=[vector],
            anns_field=milvus_config.dense_field,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=limit or milvus_config.dense_top_k,
            filter=expr or "",
            output_fields=output_fields or _DEFAULT_OUTPUT,
        )
        return _flatten(hits, "dense_score")
    except Exception:
        logger.exception("[milvus] dense 检索失败")
        return []


def search_sparse(
    vector: Dict[int, float],
    expr: str = "",
    limit: int | None = None,
    output_fields: Optional[List[str]] = None,
) -> List[dict]:
    client = get_client()
    if client is None or not vector:
        return []
    try:
        hits = client.search(
            collection_name=milvus_config.chunks_collection,
            data=[vector],
            anns_field=milvus_config.sparse_field,
            search_params={"metric_type": "IP", "params": {}},
            limit=limit or milvus_config.sparse_top_k,
            filter=expr or "",
            output_fields=output_fields or _DEFAULT_OUTPUT,
        )
        return _flatten(hits, "sparse_score")
    except Exception:
        logger.exception("[milvus] sparse 检索失败")
        return []


def hybrid_search(
    dense: List[float],
    sparse: Dict[int, float],
    expr: str = "",
    limit: int | None = None,
    weights: tuple[float, float] = (0.7, 0.3),
) -> List[dict]:
    """一次请求内融合两路召回。`weights` 仅在用 WeightedRanker 时生效。"""
    client = get_client()
    if client is None or not dense or not sparse:
        # 缺一路就退回单路，比直接返回空好
        return search_dense(dense, expr, limit) or search_sparse(sparse, expr, limit)
    try:
        from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

        dense_req = AnnSearchRequest(
            data=[dense],
            anns_field=milvus_config.dense_field,
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=limit or milvus_config.dense_top_k,
            expr=expr or None,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse],
            anns_field=milvus_config.sparse_field,
            param={"metric_type": "IP", "params": {}},
            limit=limit or milvus_config.sparse_top_k,
            expr=expr or None,
        )
        ranker = RRFRanker(k=milvus_config.dense_top_k) if weights is None else WeightedRanker(*weights)
        hits = client.hybrid_search(
            collection_name=milvus_config.chunks_collection,
            reqs=[dense_req, sparse_req],
            ranker=ranker,
            limit=limit or milvus_config.dense_top_k,
            output_fields=_DEFAULT_OUTPUT,
        )
        return _flatten(hits, "hybrid_score")
    except Exception:
        logger.exception("[milvus] 混合检索失败，退回单路")
        return search_dense(dense, expr, limit) or search_sparse(sparse, expr, limit)


_DEFAULT_OUTPUT = [
    "chunk_id",
    "doc_id",
    "chunk_index",
    "doc_title",
    "section",
    "city",
    "province",
    "content_type",
    "district",
    "entities",
    "source_path",
    "updated_at",
    "text",
]


def _flatten(hits, score_key: str) -> List[dict]:
    """把 pymilvus 的命中结果压成统一 dict 列表。"""
    out: List[dict] = []
    if not hits:
        return out
    first = hits[0] if isinstance(hits, list) else hits
    for hit in first:
        entity = dict(getattr(hit, "entity", {}) or {})
        if not entity and isinstance(hit, dict):
            entity = dict(hit.get("entity", {}))
        score = float(getattr(hit, "distance", getattr(hit, "score", 0.0)) or 0.0)
        entity[score_key] = round(score, 6)
        entity.setdefault("chunk_id", getattr(hit, "id", ""))
        out.append(entity)
    return out


def query_entities(city: str, name: str | None = None, limit: int = 20) -> List[dict]:
    client = get_client()
    if client is None:
        return []
    try:
        from utils.escape_milvus_string_utils import and_expr, eq_expr

        expr = and_expr(eq_expr("city", city), eq_expr("name", name) if name else "")
        return client.query(collection_name=milvus_config.entities_collection, filter=expr, limit=limit, output_fields=["*"])
    except Exception:
        logger.exception("[milvus] 实体查询失败")
        return []


def stats() -> dict:
    client = get_client()
    if client is None:
        return {"available": False, "error": _reason()}
    out = {"available": True}
    for name in (milvus_config.chunks_collection, milvus_config.entities_collection):
        try:
            info = client.get_collection_stats(name)
            out[name] = info.get("row_count", 0) if isinstance(info, dict) else str(info)
        except Exception as exc:
            out[name] = f"error: {exc}"
    return out


def reset_client() -> None:
    global _client
    _client = None
