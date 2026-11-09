"""MinerU 配置。语料以 Markdown 为主，PDF 分支才用得上，默认关闭。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class MineruConfig:
    enabled: bool = field(default_factory=lambda: os.getenv("MINERU_ENABLED", "0") == "1")
    base_url: str = field(default_factory=lambda: os.getenv("MINERU_BASE_URL", "http://127.0.0.1:8888"))
    backend: str = field(default_factory=lambda: os.getenv("MINERU_BACKEND", "pipeline"))
    timeout: float = field(default_factory=lambda: float(os.getenv("MINERU_TIMEOUT", "300")))
    # 解析产物的落地目录（相对项目根）
    output_dir: str = field(default_factory=lambda: os.getenv("MINERU_OUTPUT_DIR", "output"))


mineru_config = MineruConfig()
