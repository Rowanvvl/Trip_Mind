"""Markdown 图片处理。

把文中的图片引用转成可访问 URL：
- 网络图（http/https）原样保留；
- 本地图上传 MinIO，替换为对象 URL；
- 上传失败时**保留原路径并记录错误**，不删图片引用（宁可坏链，不可丢图）。

导入期要把图片落到 MinIO 的原因：语料目录后续可能被移动或清理，
切片里存相对路径迟早全部失效。
"""
from __future__ import annotations

import re
from pathlib import Path

from common.logging.logger import logger, node_log
from processor.import_processor.state import ImportGraphState
from utils.clients import minio_utils

_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)(?:\s+\"[^\"]*\")?\)")


@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> dict:
    """抽取图片、上传对象存储、替换 URL。

    输出：`markdown`、`images`、`stage`、`stats.img`
    """
    text = state.get("markdown") or ""
    source_dir = Path(state.get("source_path") or ".").parent
    images: list[dict] = []
    uploaded = 0
    failed = 0

    def _replace(match: re.Match) -> str:
        nonlocal uploaded, failed
        alt = match.group("alt") or ""
        src = match.group("src")

        if src.startswith(("http://", "https://")):
            images.append({"alt": alt, "url": src, "object_name": "", "remote": True})
            return match.group(0)

        local = (source_dir / src).resolve()
        if not local.exists():
            failed += 1
            logger.warning(f"[md_img] 本地图片不存在：{local}")
            return match.group(0)

        object_name = f"images/{state.get('task_id') or 'import'}/{local.name}"
        saved = minio_utils.upload_file(local, object_name)
        if not saved:
            failed += 1
            return match.group(0)

        uploaded += 1
        url = minio_utils.object_url(saved)
        images.append({"alt": alt, "url": url, "object_name": saved, "remote": False})
        return f"![{alt}]({url})"

    new_text = _IMG_RE.sub(_replace, text)

    return {
        "markdown": new_text,
        "images": images,
        "stage": "md_img",
        "stats": {
            **(state.get("stats") or {}),
            "img": {"total": len(images), "uploaded": uploaded, "failed": failed},
        },
    }
