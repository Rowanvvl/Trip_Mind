"""MinIO 配置：攻略图片、行程海报。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class MinioConfig:
    endpoint: str = field(default_factory=lambda: os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000"))
    access_key: str = field(default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "minioadmin"))
    secret_key: str = field(default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "minioadmin"))
    bucket: str = field(default_factory=lambda: os.getenv("MINIO_BUCKET", "tripmind"))
    secure: bool = field(default_factory=lambda: os.getenv("MINIO_SECURE", "0") == "1")
    # 对外可访问前缀（内网直连时与 endpoint 一致）
    public_base: str = field(default_factory=lambda: os.getenv("MINIO_PUBLIC_BASE", ""))

    def object_url(self, object_name: str) -> str:
        base = self.public_base or f"http://{self.endpoint}"
        return f"{base.rstrip('/')}/{self.bucket}/{object_name.lstrip('/')}"


minio_config = MinioConfig()
