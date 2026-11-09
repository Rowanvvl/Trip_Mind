"""地理计算：距离估算、片区聚类、出行方式建议。

定位很明确：**这里只做「地图接口不可用时的兜底估算」**。
真实车程一律走 `utils/tools/amap_utils`；本模块的输出必须带 `estimated=True`，
答案与 trace 里要能看出这个数字是算出来的还是查到的。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from utils.geo.city_gazetteer import (
    Attraction,
    attractions_of,
    district_of,
    find_attractions_in_text,
    get_city_info,
    normalize_city,
    resolve_attraction,
)

EARTH_RADIUS_KM = 6371.0
# 直线 → 实际路网的放大系数（框架文档 11.3）
STRAIGHT_LINE_FACTOR = 1.3

# 各出行方式的平均速度（km/h），含等候与进出站
_SPEED_KMH = {
    "步行": 4.5,
    "骑行": 12.0,
    "公共交通": 18.0,
    "打车": 26.0,
    "自驾": 30.0,
}


def haversine_km(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    """两点球面距离（km）。"""
    lng1, lat1, lng2, lat2 = map(math.radians, (lng1, lat1, lng2, lat2))
    dlng, dlat = lng2 - lng1, lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def estimate_distance_km(
    origin: str | Attraction | tuple[float, float],
    destination: str | Attraction | tuple[float, float],
    city: str | None = None,
    factor: float = STRAIGHT_LINE_FACTOR,
) -> Optional[float]:
    """估算两点距离。任一坐标缺失时返回 None，**不要用 0 糊弄**。"""
    a = _coords_of(origin, city)
    b = _coords_of(destination, city)
    if a is None or b is None:
        return None
    return round(haversine_km(a[0], a[1], b[0], b[1]) * factor, 2)


def _coords_of(item, city: str | None) -> Optional[tuple[float, float]]:
    if isinstance(item, Attraction):
        return (item.lng, item.lat)
    if isinstance(item, tuple) and len(item) == 2:
        return (float(item[0]), float(item[1]))
    if isinstance(item, str):
        attraction = resolve_attraction(item, city)
        if attraction:
            return (attraction.lng, attraction.lat)
    return None


def suggest_transport(distance_km: float | None) -> str:
    """按距离给默认出行方式。用户明确说了方式时不调用本函数。"""
    if distance_km is None:
        return "公共交通"
    if distance_km <= 1.5:
        return "步行"
    if distance_km <= 5:
        return "骑行"
    if distance_km <= 25:
        return "公共交通"
    return "打车"


def travel_minutes(distance_km: float | None, mode: str = "公共交通") -> Optional[int]:
    """按平均速度估时间，含 10 分钟固定损耗（进出站/找车）。"""
    if distance_km is None:
        return None
    speed = _SPEED_KMH.get(mode, 20.0)
    return int(round(distance_km / speed * 60 + 10))


def group_by_district(names: Sequence[str], city: str | None = None) -> Dict[str, List[str]]:
    """按行政区归组。行程分天的第一步，比按距离聚类稳。"""
    groups: Dict[str, List[str]] = {}
    for name in names:
        district = district_of(name, city) or "未识别"
        groups.setdefault(district, []).append(name)
    return groups


@dataclass
class Cluster:
    label: str
    members: List[str] = field(default_factory=list)
    center: Optional[tuple[float, float]] = None

    @property
    def size(self) -> int:
        return len(self.members)

    def to_dict(self) -> dict:
        return {"label": self.label, "members": self.members, "size": self.size}


def cluster_attractions(
    names: Sequence[str],
    city: str | None = None,
    radius_km: float = 8.0,
    max_clusters: int = 6,
) -> List[Cluster]:
    """贪心聚类：先按行政区粗分，区内再按半径合并。

    不追求最优解——行程排布只需要「同一片区的东西别拆到两天」，
    过度精细的聚类反而会让每天的点位过少。
    """
    by_district = group_by_district(names, city)
    clusters: List[Cluster] = []

    for district, members in by_district.items():
        remaining = list(members)
        while remaining and len(clusters) < max_clusters:
            seed = remaining.pop(0)
            seed_coords = _coords_of(seed, city)
            group = [seed]
            still: List[str] = []
            for other in remaining:
                distance = None
                other_coords = _coords_of(other, city)
                if seed_coords and other_coords:
                    distance = estimate_distance_km(seed_coords, other_coords)
                if distance is None or distance <= radius_km:
                    group.append(other)
                else:
                    still.append(other)
            remaining = still
            clusters.append(Cluster(label=district, members=group, center=seed_coords))
        # 簇数到上限后剩下的点各自单独成簇。
        # 原来这里是「并入最后一个簇，避免丢点」，但那会把 55 公里外的都江堰
        # 粘进锦江区的一天里，等于亲手制造跨区折返。宁可让它单独成簇，
        # 由上层按天数和日跨度决定要不要排。
        for name in remaining:
            clusters.append(Cluster(label=district, members=[name], center=_coords_of(name, city)))

    return clusters


def select_indoor_alternatives(city: str, exclude: Iterable[str] = (), limit: int = 3) -> List[dict]:
    """雨天替换：返回室内候选。**替换动作发生在 node_time_budget，不是提示词里。**"""
    excluded = set(exclude)
    items = [a for a in attractions_of(city) if a.indoor and a.name not in excluded]
    return [a.to_dict() for a in items[:limit]]


def resolve_entities(text: str, city: str | None = None) -> List[dict]:
    """从切片文本里抽地理实体，供 `node_geo_resolve` 写 Milvus 元数据。"""
    out = []
    for item in find_attractions_in_text(text, city):
        out.append(
            {
                "name": item.name,
                "city": item.city,
                "district": item.district,
                "lng": item.lng,
                "lat": item.lat,
                "indoor": item.indoor,
            }
        )
    return out


def is_cross_city(a: str | None, b: str | None) -> bool:
    """跨城判断（「北京上海 7 天」属于跨城，需要分段排）。"""
    if not a or not b:
        return False
    return normalize_city(a) != normalize_city(b)


def districts_of_city(city: str) -> List[str]:
    return list(get_city_info(city).get("districts", []))


def area_hint(city: str, names: Sequence[str]) -> str:
    """给一句话的片区判断，用于答案里的「同一片区」表述。"""
    groups = group_by_district(names, city)
    parts = []
    for district, members in groups.items():
        if district == "未识别":
            continue
        parts.append(f"{district}（{'、'.join(members[:3])}{'等' if len(members) > 3 else ''}）")
    return "；".join(parts)
