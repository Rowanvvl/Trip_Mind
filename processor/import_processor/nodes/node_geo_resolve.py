"""地理实体消歧。

解决两类现实问题：
1. **同名景点**：西湖（杭州）、南山（三亚 / 重庆）、鼓楼（西安 / 南京）；
2. **区县归属**：行程聚类要知道某个景点属于哪个片区，切片里必须带上。

处理顺序：城市上下文优先 → 景点别名索引 → 都查不到就留空。
**留空比猜错好**：district 为空只会让聚类退回按距离分组，猜错会让行程跨区。
"""
from __future__ import annotations

from collections import Counter
from typing import List

from common.logging.logger import logger, node_log
from processor.import_processor.state import ImportGraphState
from utils.geo.city_gazetteer import get_city_info, normalize_city, resolve_attraction


@node_log("node_geo_resolve")
def node_geo_resolve(state: ImportGraphState) -> dict:
    """给每个切片补 province / district，并汇总本文档涉及的实体。

    输出：`chunks`（已补字段）、`entities`、`stage`、`stats.geo`
    """
    chunks: List[dict] = [dict(c) for c in (state.get("chunks") or [])]
    city = normalize_city(state.get("city") or (state.get("meta") or {}).get("city") or "")
    province = get_city_info(city).get("province", "") if city else ""

    unresolved = 0
    for chunk in chunks:
        chunk["province"] = province
        districts: List[str] = []
        for name in chunk.get("entities") or []:
            attraction = resolve_attraction(name, city)
            if attraction and attraction.district:
                districts.append(attraction.district)
        if districts:
            # 一段切片可能跨片区，取出现最多的那个作为主片区
            chunk["district"] = Counter(districts).most_common(1)[0][0]
        else:
            unresolved += 1
            chunk["district"] = ""

    # 实体汇总：给 trip_entities 用
    entity_rows = []
    for chunk in chunks:
        for name in chunk.get("entities") or []:
            attraction = resolve_attraction(name, city)
            entity_rows.append(
                {
                    "name": name,
                    "city": city,
                    "district": attraction.district if attraction else "",
                    "doc_id": chunk.get("doc_id"),
                    "content_type": chunk.get("content_type", ""),
                }
            )

    if unresolved:
        logger.debug(f"[geo_resolve] {unresolved} 个切片未识别到片区 city={city}")

    return {
        "chunks": chunks,
        "entities": entity_rows,
        "stage": "geo_resolve",
        "stats": {
            **(state.get("stats") or {}),
            "geo": {
                "province": province,
                "unresolved_chunks": unresolved,
                "entities": len({row["name"] for row in entity_rows}),
            },
        },
    }
