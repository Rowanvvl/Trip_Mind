"""片区聚类与分天。

**这是行程质量的分水岭**。纯 LLM 排行程最容易犯的错就是跨区折返：
上午在亚龙湾、下午跑去天涯海角、晚上回亚龙湾住。所以分天由规则决定：
1. 按行政区聚成簇（`cluster_attractions`）；
2. 簇按规模从大到小装进天数，每天装 1 个主簇；
3. 簇数少于天数时，把最大簇拆成两天（避免某天只有 1 个点）；
4. 需要整天的景区独占一天。

路段（`route_legs`）用高德距离矩阵，取不到就用直线估算并标 `estimated`。
"""
from __future__ import annotations

from typing import List

from common.config import amap_config, trip_config
from common.logging.logger import node_log
from processor.plan_processor.state import PlanGraphState
from utils.geo.geo_utils import Cluster, cluster_attractions, estimate_distance_km, haversine_km, travel_minutes
from utils.tools import amap_utils
from utils.tools.tool_registry import is_open


@node_log("node_route_cluster")
def node_route_cluster(state: PlanGraphState) -> dict:
    """聚类并分配到天。

    输出：`clustered_days`、`route_legs`、`daily_candidates`
    """
    city = state.get("city") or ""
    days = int(state.get("days") or 3)
    pool = state.get("candidate_pool") or []

    if not pool:
        return {"error": "候选池为空，无法排布", "degraded": True}

    by_name = {c["name"]: c for c in pool}
    clusters = cluster_attractions([c["name"] for c in pool], city, radius_km=10.0, max_clusters=days * 2)

    # 整天景区单独成簇
    full_day = [c["name"] for c in pool if c.get("duration_min", 0) >= 240]
    clusters = [cluster for cluster in clusters if not (len(cluster.members) == 1 and cluster.members[0] in full_day)]
    for name in full_day:
        clusters.append(Cluster(label=by_name[name].get("district", "整天"), members=[name]))

    clusters.sort(key=lambda c: c.size, reverse=True)

    # 分配到天
    assigned: List[dict] = []
    for day in range(1, days + 1):
        cluster = clusters[day - 1] if day - 1 < len(clusters) else None
        if cluster is None:
            # 簇不够：复用最大簇，按顺序切分
            base = clusters[(day - 1) % max(len(clusters), 1)]
            assigned.append({"day": day, "area": base.label, "members": list(base.members)})
        else:
            assigned.append({"day": day, "area": cluster.label, "members": list(cluster.members)})

    # 去重：同一个点不出现在两天
    seen: set[str] = set()
    for entry in assigned:
        unique = []
        for name in entry["members"]:
            if name in seen:
                continue
            seen.add(name)
            unique.append(name)
        entry["members"] = unique

    _fill_leftovers(assigned, clusters, by_name, full_day)
    _refresh_area(assigned, by_name)

    # 路段
    legs = []
    for entry in assigned:
        members = entry["members"]
        for index in range(len(members) - 1):
            legs.append(_leg(members[index], members[index + 1], by_name, city, entry["day"]))

    return {
        "clustered_days": assigned,
        "route_legs": legs,
        "daily_candidates": {entry["day"]: [by_name[m] for m in entry["members"] if m in by_name] for entry in assigned},
    }


def _fill_leftovers(assigned: List[dict], clusters: List[Cluster], by_name: dict, full_day: List[str]) -> None:
    """把没排进前 N 个簇的点补进已有行程。

    簇数多于天数时，多出来的簇原本会被整簇丢掉，
    结果就是「北京 3 天」里第三天只有 798 一个点，而什刹海根本没出现。

    选哪一天，按这个顺序：
    1. 同片区的那一天（片区一致最符合「一天待一片」的直觉）；
    2. 否则给**当前点数最少**的那一天——按「离得最近」补会把所有剩余点堆进第一天，
       第一天被时间预算砍掉一半，最后一天仍只有一个点，不如摊开；
       同点数时再比距离。

    三条硬约束：
    1. 需要整天的景区不补（补进去只会被时间预算砍掉）；
    2. 已经放了整天景区的那一天不再补（那天本来就已经占满）；
    3. 会让当天跨度超过 `day_span_km_limit` 的点不补——
       补进去就是把「都江堰」硬塞进锦江区的一天，后面校验还是要打回来。
    """
    used = {name for entry in assigned for name in entry["members"]}
    day_by_area = {entry["area"]: entry for entry in assigned}
    full_day_set = set(full_day)

    for cluster in clusters:
        for name in cluster.members:
            if name in used or name in full_day_set:
                continue

            options = [
                entry
                for entry in assigned
                if entry["members"]
                and not (set(entry["members"]) & full_day_set)
                and _within_span(name, entry, by_name)
            ]
            if not options:
                continue

            target = day_by_area.get((by_name.get(name) or {}).get("district", ""))
            if target not in options:
                target = min(options, key=lambda e: (len(e["members"]), _distance_to(name, e, by_name)))
            target["members"].append(name)
            used.add(name)


def _refresh_area(assigned: List[dict], by_name: dict) -> None:
    """按最终成员重算当天的片区标签。

    补点之后当天可能真的横跨两个区（798 在朝阳、什刹海在西城），
    标题还写「朝阳区」就是在骗人——如实写出来，用户自己判断要不要拆两天。
    """
    for entry in assigned:
        districts: List[str] = []
        for name in entry["members"]:
            district = (by_name.get(name) or {}).get("district") or ""
            if district and district not in districts:
                districts.append(district)
        if not districts:
            continue
        entry["area"] = "、".join(districts) if len(districts) <= 2 else f"{districts[0]}等 {len(districts)} 个片区"


def _distance_to(name: str, entry: dict, by_name: dict) -> float:
    """点到某一天已有成员的最近距离（km）；取不到坐标时返回 0，等价于「不排斥」。"""
    point = by_name.get(name) or {}
    if not point.get("lng"):
        return 0.0
    distances = [
        haversine_km(point["lng"], point["lat"], item["lng"], item["lat"])
        for item in (by_name.get(m) or {} for m in entry["members"])
        if item.get("lng") and item.get("lat")
    ]
    return min(distances) if distances else 0.0


def _within_span(name: str, entry: dict, by_name: dict) -> bool:
    """这个点加进这一天，会不会让当天跨度超限。"""
    point = by_name.get(name) or {}
    if not point.get("lng"):
        return True
    coords = [
        (item["lng"], item["lat"])
        for item in (by_name.get(m) or {} for m in entry["members"])
        if item.get("lng") and item.get("lat")
    ]
    if not coords:
        return True
    return all(
        haversine_km(point["lng"], point["lat"], lng, lat) <= trip_config.day_span_km_limit for lng, lat in coords
    )


def _leg(origin: str, destination: str, by_name: dict, city: str, day: int) -> dict:
    """两点之间的路程。优先高德，失败用直线估算。"""
    if amap_config.enabled and not is_open("amap_distance"):
        o = by_name.get(origin) or {}
        d = by_name.get(destination) or {}
        if o.get("lng") and d.get("lng"):
            result = amap_utils.distance_matrix(
                (o["lng"], o["lat"]),
                [(d["lng"], d["lat"])],
                mode="driving",
            )
            if result.ok and result.data:
                item = result.data[0]
                return {
                    "day": day,
                    "from": origin,
                    "to": destination,
                    "distance_m": item.get("distance_m", 0),
                    "duration_s": item.get("duration_s", 0),
                    "mode": "驾车",
                    "estimated": False,
                }

    distance = estimate_distance_km(origin, destination, city)
    return {
        "day": day,
        "from": origin,
        "to": destination,
        "distance_m": (distance or 0) * 1000,
        "duration_s": (travel_minutes(distance, "打车") or 0) * 60,
        "mode": "打车",
        "estimated": True,
    }
