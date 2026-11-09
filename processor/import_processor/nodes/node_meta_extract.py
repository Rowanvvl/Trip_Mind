"""元数据抽取。

语料开头的 `## 元数据` 块是结构化信息的唯一来源，格式固定为 `- 键：值`。
抽不到时按目录名与文件名兜底（`景点攻略/三亚景点推荐.md` → 类型「景点攻略」、城市「三亚」）。
**兜底也失败时不要瞎猜**：city 留空，让 `node_geo_resolve` 再从正文里找。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

from common.config import CONTENT_TYPE_BY_DIR, normalize_content_type
from common.logging.logger import node_log
from processor.import_processor.state import ImportGraphState
from utils.geo.city_gazetteer import find_city_in_text, normalize_city

_META_BLOCK = re.compile(r"^##\s*元数据\s*$(.*?)(?=^##\s|\Z)", re.MULTILINE | re.DOTALL)
_KV_LINE = re.compile(r"^[-*]\s*([^：:]+)[：:]\s*(.+)$", re.MULTILINE)
_TITLE = re.compile(r"^#\s+(.+)$", re.MULTILINE)

# 元数据键 → 内部字段
_KEY_ALIAS = {
    "内容类型": "content_type",
    "类型": "content_type",
    "城市": "city",
    "目的地": "city",
    "景点名称": "attraction",
    "景点": "attraction",
    "线路名称": "route",
    "线路": "route",
    "酒店名称": "hotel",
    "酒店": "hotel",
    "主题": "theme",
    "文档名称": "doc_title",
}


@node_log("node_meta_extract")
def node_meta_extract(state: ImportGraphState) -> dict:
    """解析元数据块，并与目录/文件名兜底合并。

    输出：`meta`、`city`、`content_type`、`stage`、`stats.meta`
    """
    text = state.get("markdown") or ""
    source_path = Path(state.get("source_path") or "")

    meta: Dict[str, str] = {}
    block = _META_BLOCK.search(text)
    if block:
        for key, value in _KV_LINE.findall(block.group(1)):
            field = _KEY_ALIAS.get(key.strip(), key.strip())
            meta[field] = value.strip()

    # 标题
    title_match = _TITLE.search(text)
    meta.setdefault("doc_title", title_match.group(1).strip() if title_match else source_path.stem)

    # 类型兜底：父目录名
    if not meta.get("content_type"):
        for parent in source_path.parents:
            if parent.name in CONTENT_TYPE_BY_DIR:
                meta["content_type"] = CONTENT_TYPE_BY_DIR[parent.name]
                break

    # 类型归一：元数据里写的是「景点介绍 / 美食介绍」，内部标准取值是「景点攻略 / 美食推荐」
    content_type = normalize_content_type(meta.get("content_type", ""))
    meta["content_type"] = content_type

    # 城市兜底：元数据 → 文件名 → 正文
    city = normalize_city(meta.get("city")) if meta.get("city") else ""
    if not city:
        city = find_city_in_text(source_path.stem) or ""
    if not city:
        city = find_city_in_text(text[:500]) or ""
    if city:
        meta["city"] = city

    return {
        "meta": meta,
        "city": city,
        "content_type": content_type,
        "stage": "meta_extract",
        "stats": {**(state.get("stats") or {}), "meta": {"keys": sorted(meta.keys()), "city": city}},
    }
