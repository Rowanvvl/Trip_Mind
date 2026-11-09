"""重排模型配置。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class RerankerConfig:
    model_path: str = field(default_factory=lambda: os.getenv("BGE_RERANKER_LARGE", "./models/bge-reranker-large"))
    device: str = field(default_factory=lambda: os.getenv("BGE_RERANKER_DEVICE", "cpu"))
    use_fp16: bool = field(default_factory=lambda: os.getenv("BGE_RERANKER_FP16", "0") == "1")
    batch_size: int = field(default_factory=lambda: int(os.getenv("RERANKER_BATCH_SIZE", "8")))
    max_length: int = field(default_factory=lambda: int(os.getenv("RERANKER_MAX_LENGTH", "512")))
    # 低于该分数的切片直接丢弃；node_retrieve_assess 也用它判断「召回质量够不够」
    score_threshold: float = field(default_factory=lambda: float(os.getenv("RERANK_SCORE_THRESHOLD", "0.2")))


reranker_config = RerankerConfig()
