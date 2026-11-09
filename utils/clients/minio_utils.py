"""MinIO 封装：攻略图片与行程海报。

图片不落 Milvus（向量库不适合存二进制），也不落本地磁盘（多实例会丢），
统一进对象桶，切片里只存 object_name，展示时拼 URL。
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional

from common.config import minio_config
from common.logging.logger import logger

_client = None
_unavailable = False


def get_client():
    global _client, _unavailable
    if _client is not None or _unavailable:
        return _client
    try:
        from minio import Minio

        _client = Minio(
            minio_config.endpoint,
            access_key=minio_config.access_key,
            secret_key=minio_config.secret_key,
            secure=minio_config.secure,
        )
        logger.info(f"[minio] 已连接 {minio_config.endpoint}")
    except Exception as exc:
        _unavailable = True
        logger.warning(f"[minio] 不可用（{exc}），图片功能降级")
        _client = None
    return _client


def available() -> bool:
    return get_client() is not None


def ensure_bucket() -> bool:
    client = get_client()
    if client is None:
        return False
    try:
        if not client.bucket_exists(minio_config.bucket):
            client.make_bucket(minio_config.bucket)
            logger.info(f"[minio] 创建桶 {minio_config.bucket}")
        return True
    except Exception:
        logger.exception("[minio] 建桶失败")
        return False


def upload_bytes(data: bytes, object_name: str, content_type: str = "image/jpeg") -> Optional[str]:
    client = get_client()
    if client is None:
        return None
    ensure_bucket()
    try:
        client.put_object(
            minio_config.bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return object_name
    except Exception:
        logger.exception(f"[minio] 上传失败 {object_name}")
        return None


def upload_file(local_path: str | Path, object_name: str, content_type: str | None = None) -> Optional[str]:
    path = Path(local_path)
    if not path.exists():
        logger.warning(f"[minio] 本地文件不存在：{path}")
        return None
    return upload_bytes(path.read_bytes(), object_name, content_type or _guess_type(path))


def _guess_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/jpeg")


def object_url(object_name: str) -> str:
    return minio_config.object_url(object_name)


def list_objects(prefix: str = "", limit: int = 100) -> List[str]:
    client = get_client()
    if client is None:
        return []
    try:
        return [obj.object_name for obj in client.list_objects(minio_config.bucket, prefix=prefix, recursive=True)][:limit]
    except Exception:
        logger.exception("[minio] 列举对象失败")
        return []


def stats() -> dict:
    return {
        "available": available(),
        "endpoint": minio_config.endpoint,
        "bucket": minio_config.bucket,
    }
