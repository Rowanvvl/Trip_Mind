"""高德 Web 服务封装。

只暴露四个业务语义的函数，不暴露原始接口：
- `geocode` 地址 → 坐标 + adcode
- `place_search` 关键词 → POI（含坐标、评分、照片）
- `route` 两点 → 距离/时长/分段（按出行方式）
- `distance_matrix` 多点 → 两两距离（行程聚类用）

三个原则：
1. **所有返回都归一化**，节点不接触高德的原始字段名；
2. **失败返回 None**，由 `tool_registry` 决定降级，节点自己不算里程；
3. **景区内部无路网是常态**（步行道、栈道），此时返回 `estimated=True` 的直线估算，
   并显式标注，不能让用户以为「亚龙湾到天涯海角 8 分钟」是真实车程。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import httpx

from common.config import amap_config
from common.logging.logger import logger
from utils.geo.geo_utils import haversine_km, suggest_transport, travel_minutes
from utils.tools.tool_registry import call, estimated_result, ToolResult

_STRAIGHT_FACTOR = amap_config.straight_line_factor

_MODE_PATH = {
    "driving": "/v3/direction/driving",
    "walking": "/v3/direction/walking",
    "bicycling": "/v4/direction/bicycling",
    "transit": "/v3/direction/transit/integrated",
}


def _get(path: str, params: Dict[str, Any], key: Optional[str] = None) -> Optional[dict]:
    """最底层的 GET。key 缺失或接口返回非 1 状态时返回 None。"""
    api_key = key or amap_config.key
    if not api_key:
        logger.debug("[amap] 未配置 AMAP_KEY，跳过调用")
        return None
    query = {**params, "key": api_key, "output": "JSON"}
    with httpx.Client(timeout=amap_config.timeout) as client:
        resp = client.get(f"{amap_config.host}{path}", params=query)
        resp.raise_for_status()
        payload = resp.json()
    if str(payload.get("status")) != "1":
        logger.warning(f"[amap] {path} 返回异常：{payload.get('info')} {payload.get('infocode')}")
        return None
    return payload


def _lnglat(location: str) -> tuple[float, float]:
    lng, lat = str(location).split(",")[:2]
    return float(lng), float(lat)


# ---------------------------------------------------------------- 地理编码
def geocode(address: str, city: str | None = None) -> ToolResult:
    """地址 → 坐标 + adcode。天气接口需要 adcode，所以这个函数是很多链路的前置。"""
    if not address:
        return ToolResult(ok=False, error="empty_address", source="amap_geocode")

    def _do():
        payload = _get("/v3/geocode/geo", {"address": address, "city": city or ""}, key=amap_config.import_key)
        geocodes = (payload or {}).get("geocodes") or []
        if not geocodes:
            return None
        item = geocodes[0]
        lng, lat = _lnglat(item["location"])
        return {
            "address": item.get("formatted_address", address),
            "lng": lng,
            "lat": lat,
            "adcode": item.get("adcode", ""),
            "city": item.get("city") or city or "",
            "level": item.get("level", ""),
        }

    return call("amap_geocode", _do, cache_key=f"geo:{address}:{city or ''}")


def adcode_of(city: str) -> Optional[str]:
    """取城市 adcode（天气接口必需）。带缓存，失败返回 None。"""
    result = geocode(city, city)
    if result.ok and isinstance(result.data, dict):
        return result.data.get("adcode") or None
    return None


# ---------------------------------------------------------------- POI 检索
def place_search(keywords: str, city: str | None = None, limit: int = 5) -> ToolResult:
    """关键词 → POI。用于补坐标、取照片、验证景点是否存在。"""

    def _do():
        payload = _get(
            "/v3/place/text",
            {"keywords": keywords, "city": city or "", "citylimit": "true" if city else "false", "offset": limit, "page": 1},
            key=amap_config.import_key,
        )
        pois = (payload or {}).get("pois") or []
        if not pois:
            return None
        out = []
        for poi in pois[:limit]:
            location = poi.get("location") or ""
            if not location:
                continue
            lng, lat = _lnglat(location)
            out.append(
                {
                    "name": poi.get("name"),
                    "address": poi.get("address") if isinstance(poi.get("address"), str) else "",
                    "lng": lng,
                    "lat": lat,
                    "type": poi.get("type", ""),
                    "adcode": poi.get("adcode", ""),
                    "photos": [p.get("url") for p in (poi.get("photos") or []) if p.get("url")],
                }
            )
        return out or None

    return call("amap_place", _do, cache_key=f"place:{keywords}:{city or ''}:{limit}")


def poi_photos(name: str, city: str | None = None, limit: int = 3) -> ToolResult:
    """取景点照片 URL。语料图优先，这里只做补充（框架文档 11.2）。"""
    result = place_search(name, city, limit=1)
    if not result.ok:
        return result
    pois = result.data or []
    urls: List[str] = []
    for poi in pois:
        urls.extend(poi.get("photos") or [])
    if not urls:
        return ToolResult(ok=False, error="no_photo", source="amap_poi_photo")
    return ToolResult(ok=True, data=urls[:limit], source="amap_poi_photo")


# ---------------------------------------------------------------- 路径规划
def route(
    origin: str | tuple[float, float],
    destination: str | tuple[float, float],
    mode: str = "transit",
    city: str | None = None,
    strategy: int | None = None,
) -> ToolResult:
    """路径规划。origin/destination 支持 "lng,lat" 或 (lng, lat)。

    无路网数据（景区内部）时退化为直线估算，`estimated=True`。
    """
    origin_text = _to_lnglat(origin)
    destination_text = _to_lnglat(destination)
    if not origin_text or not destination_text:
        return ToolResult(ok=False, error="bad_location", source="amap_direction")

    mode = mode if mode in _MODE_PATH else "transit"

    def _do():
        params: Dict[str, Any] = {"origin": origin_text, "destination": destination_text}
        if mode == "transit":
            params.update({"city": city or "", "cityd": city or "", "strategy": 0})
        elif mode == "driving":
            params.update({"strategy": strategy if strategy is not None else amap_config.driving_strategy, "extensions": "base"})
        else:
            params.update({"extensions": "base"})

        payload = _get(_MODE_PATH[mode], params)
        path = (payload or {}).get("route", {}).get("paths") or []
        if not path:
            return None
        first = path[0]
        distance_m = float(first.get("distance") or 0)
        duration_s = float(first.get("duration") or 0)

        steps = []
        for step in (first.get("steps") or [])[:8]:
            instruction = step.get("instruction") or step.get("road") or ""
            steps.append(
                {
                    "instruction": instruction,
                    "distance_m": float(step.get("distance") or 0),
                    "duration_s": float(step.get("duration") or 0),
                    "line": step.get("line_name") or step.get("road") or "",
                }
            )
        return {
            "mode": mode,
            "distance_m": distance_m,
            "duration_s": duration_s,
            "tolls": float(first.get("tolls") or 0),
            "steps": steps,
            "estimated": False,
        }

    cache_key = f"route:{mode}:{origin_text}:{destination_text}"
    result = call("amap_direction", _do, cache_key=cache_key)
    if result.ok:
        return result

    # 降级：直线距离 × 系数 + 速度表
    a, b = _lnglat(origin_text), _lnglat(destination_text)
    straight = haversine_km(a[0], a[1], b[0], b[1]) * _STRAIGHT_FACTOR
    mode_suggestion = mode if mode != "transit" else suggest_transport(straight)
    minutes = travel_minutes(straight, mode_suggestion) or 0
    estimated = estimated_result(
        {
            "mode": mode_suggestion,
            "distance_m": round(straight * 1000, -2),
            "duration_s": minutes * 60,
            "tolls": 0,
            "steps": [],
            "estimated": True,
            "reason": "无路网数据，按直线距离估算",
        }
    )
    estimated.error = result.error
    return estimated


def distance_matrix(
    origin: str | tuple[float, float],
    destinations: Sequence[str | tuple[float, float]],
    mode: str = "driving",
) -> ToolResult:
    """一对多距离矩阵。行程聚类用它替代「模型猜距离」。"""
    origin_text = _to_lnglat(origin)
    dest_texts = [_to_lnglat(d) for d in destinations]
    if not origin_text or not all(dest_texts):
        return ToolResult(ok=False, error="bad_location", source="amap_distance")

    mode_code = {"driving": 1, "walking": 3, "bicycling": 3}.get(mode, 1)

    def _do():
        payload = _get(
            "/v3/distance",
            {
                "origins": origin_text,
                "destination": "|".join(dest_texts),
                "type": mode_code,
            },
        )
        results = (payload or {}).get("results") or []
        if not results:
            return None
        out = []
        for index, item in enumerate(results):
            out.append(
                {
                    "index": index,
                    "destination": dest_texts[index] if index < len(dest_texts) else "",
                    "distance_m": float(item.get("distance") or 0),
                    "duration_s": float(item.get("duration") or 0),
                }
            )
        return out

    return call(
        "amap_distance",
        _do,
        cache_key=f"dist:{mode}:{origin_text}:{'|'.join(dest_texts)}",
    )


def _to_lnglat(value: str | tuple[float, float] | None) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        return f"{float(value[0]):.6f},{float(value[1]):.6f}"
    text = str(value).strip()
    if "," in text:
        parts = [p for p in text.replace("，", ",").split(",") if p.strip()]
        if len(parts) == 2:
            try:
                return f"{float(parts[0]):.6f},{float(parts[1]):.6f}"
            except ValueError:
                return None
    return None


def plan_route_for_query(
    origin: str,
    destination: str,
    city: str | None = None,
    preferred: str | None = None,
) -> dict:
    """面向问答的封装：未指定出行方式时给出两种方案对比。

    这是「知识库判断 + 真实数据」对齐的关键：语料说「得分天」，地图给出 42km / 62min。
    """
    origin_point = origin
    destination_point = destination

    # 名称 → 坐标
    if "," not in str(origin):
        found = place_search(origin, city, limit=1)
        if found.ok and found.data:
            origin_point = (found.data[0]["lng"], found.data[0]["lat"])
    if "," not in str(destination):
        found = place_search(destination, city, limit=1)
        if found.ok and found.data:
            destination_point = (found.data[0]["lng"], found.data[0]["lat"])

    modes = [preferred] if preferred else ["transit", "driving"]
    options = []
    degraded = False
    for mode in modes:
        result = route(origin_point, destination_point, mode=mode or "transit", city=city)
        if not result.ok:
            degraded = True
            continue
        data = dict(result.data)
        data["cached"] = result.cached
        data["fetched_at"] = result.fetched_at
        options.append(data)

    return {
        "origin": origin,
        "destination": destination,
        "city": city,
        "options": options,
        "degraded": degraded or not options,
        "source": "高德地图",
        "fetched_at": time.time(),
    }
