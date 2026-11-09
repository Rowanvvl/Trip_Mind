"""存储客户端：Milvus / Mongo / MinIO。

三个客户端都是**懒连接 + 失败降级**，导入时不会因为某个组件没起就崩。
"""
from utils.clients import milvus_utils, mongo_history_utils, minio_utils

__all__ = ["milvus_utils", "mongo_history_utils", "minio_utils"]
