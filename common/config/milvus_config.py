"""Milvus 配置：两类集合，切片与实体。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class MilvusConfig:
    url: str = field(default_factory=lambda: os.getenv("MILVUS_URL", "http://127.0.0.1:19530"))
    token: str = field(default_factory=lambda: os.getenv("MILVUS_TOKEN", ""))
    db_name: str = field(default_factory=lambda: os.getenv("MILVUS_DB", "default"))

    chunks_collection: str = field(default_factory=lambda: os.getenv("TRIP_CHUNKS_COLLECTION", "trip_chunks"))
    entities_collection: str = field(default_factory=lambda: os.getenv("TRIP_ENTITIES_COLLECTION", "trip_entities"))

    dense_field: str = "dense_vector"
    sparse_field: str = "sparse_vector"
    dense_dim: int = field(default_factory=lambda: int(os.getenv("BGE_DENSE_DIM", "1024")))
    # dense/sparse 各自召回的条数
    dense_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVE_TOP_K", "20")))
    sparse_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVE_TOP_K", "20")))
    # 单次 IN 表达式的最大元素数，过长会被 Milvus 拒绝
    max_in_size: int = 200


milvus_config = MilvusConfig()
