"""BGE-M3 向量配置。dense 与 sparse 同出，一次前向两用。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class EmbeddingConfig:
    model_path: str = field(default_factory=lambda: os.getenv("BGE_M3_PATH", "./models/bge-m3"))
    device: str = field(default_factory=lambda: os.getenv("BGE_DEVICE", "cpu"))
    use_fp16: bool = field(default_factory=lambda: os.getenv("BGE_FP16", "0") == "1")
    batch_size: int = field(default_factory=lambda: int(os.getenv("BGE_BATCH_SIZE", "16")))
    max_length: int = field(default_factory=lambda: int(os.getenv("BGE_MAX_LENGTH", "1024")))
    dense_dim: int = field(default_factory=lambda: int(os.getenv("BGE_DENSE_DIM", "1024")))
    # 未加载本地权重时的兜底：走 OpenAI 兼容的 embedding 接口
    api_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_API_MODEL", "text-embedding-v3"))


embedding_config = EmbeddingConfig()
