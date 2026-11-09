"""按天取候选。

候选池来自三处，按可信度合并：
1. 词典里的该城市景点（保证点位不会凭空捏造）；
2. 检索命中的切片里提到的景点（保证与语料一致）；
3. 用户指定必去（优先占用，即使聚类不理想）。

**只从候选池里排，不允许模型新增点位**——这是行程可信的底线。
"""
from __future__ import annotations

from typing import Dict, List

from common.config import trip_config
from common.logging.logger import node_log
from processor.plan_processor.state import PlanGraphState
from utils.clients import milvus_utils
from utils.geo.city_gazetteer import attractions_of, find_attractions_in_text, resolve_attraction
from utils.lm import embedding_utils


@node_log("node_retrieve_by_day")
def node_retrieve_by_day(state: PlanGraphState) -> dict:
    """构造候选池。

    输出：`candidate_pool`、`daily_candidates`（先给全量，聚类后再分天）
    """
    city = state.get("city") or ""
    days = int(state.get("days") or 3)
    exclude = set(state.get("exclude") or [])
    must_go = list(state.get("must_go") or [])

    pool: Dict[str, dict] = {}

    # 1. 词典候选
    for attraction in attractions_of(city):
        if attraction.name in exclude:
            continue
        pool[attraction.name] = {
            "name": attraction.name,
            "district": attraction.district,
            "indoor": attraction.indoor,
            "ticket": [attraction.ticket_low, attraction.ticket_high],
            "duration_min": 240 if _is_full_day(attraction.name) else 120,
            "source": "gazetteer",
            "lng": attraction.lng,
            "lat": attraction.lat,
        }

    # 2. 检索增强：语料里提到的点位优先，并补上语料给的判断
    if city:
        dense, _, _ = embedding_utils.encode_query(f"{city} 必去景点 与 取舍")
        if dense:
            expr = _city_expr(city)
            for hit in milvus_utils.search_dense(dense, expr, trip_config.retrieve_top_k):
                text = hit.get("text", "")
                for item in find_attractions_in_text(text, city):
                    entry = pool.get(item.name)
                    if entry is None:
                        if item.name in exclude:
                            continue
                        entry = {
                            "name": item.name,
                            "district": item.district,
                            "indoor": item.indoor,
                            "ticket": [item.ticket_low, item.ticket_high],
                            "duration_min": 240 if _is_full_day(item.name) else 120,
                            "source": "corpus",
                            "lng": item.lng,
                            "lat": item.lat,
                        }
                        pool[item.name] = entry
                    entry.setdefault("evidence", hit.get("section", ""))

    # 3. 用户必去（即使词典里没有，也从词典解析一次坐标）
    for name in must_go:
        if name in pool:
            pool[name]["must_go"] = True
            continue
        attraction = resolve_attraction(name, city)
        if attraction:
            pool[name] = {
                "name": attraction.name,
                "district": attraction.district,
                "indoor": attraction.indoor,
                "ticket": [attraction.ticket_low, attraction.ticket_high],
                "duration_min": 240 if _is_full_day(attraction.name) else 120,
                "source": "user",
                "must_go": True,
                "lng": attraction.lng,
                "lat": attraction.lat,
            }

    candidates: List[dict] = sorted(
        pool.values(),
        key=lambda c: (not c.get("must_go"), c.get("district", ""), c["name"]),
    )

    # 候选数量按天数控制：每天 2~3 个，多出来的只是干扰。
    # 但不能简单地按 (区名, 名字) 排序后截断——那样会把排在后面的整个区一起丢掉，
    # 于是「北京 3 天」里海淀区（颐和园、圆明园）和西城区（什刹海）全被砍掉。
    # 按区轮转取，每个区至少留一个。
    limit = min(len(candidates), max(days * 3, 6))
    candidates = _balance_by_district(candidates, limit)

    return {
        "candidate_pool": candidates,
        "daily_candidates": {day: candidates for day in range(1, days + 1)},
    }


def _balance_by_district(candidates: List[dict], limit: int) -> List[dict]:
    """按区轮转取候选，避免截断把整个区砍掉。

    必去点位无条件保留在最前面；其余按区轮转，每轮每个区取一个。
    """
    must = [c for c in candidates if c.get("must_go")]
    buckets: Dict[str, List[dict]] = {}
    for item in candidates:
        if item.get("must_go"):
            continue
        buckets.setdefault(item.get("district") or "未知", []).append(item)
    for group in buckets.values():
        group.sort(key=lambda c: c["name"])

    ordered = list(must)[:limit]
    index = 0
    while len(ordered) < limit:
        added = False
        for district in sorted(buckets):
            group = buckets[district]
            if index < len(group):
                ordered.append(group[index])
                added = True
                if len(ordered) >= limit:
                    break
        if not added:
            break
        index += 1
    return ordered[:limit]


def _is_full_day(name: str) -> bool:
    """需要整天的大景区。这类点位独占一天，不能叠加市区点位。"""
    keywords = ("长城", "迪士尼", "武隆", "森林公园", "天门山", "华山", "千岛湖", "长隆", "玉龙雪山", "石林")
    return any(k in name for k in keywords)


def _city_expr(city: str) -> str:
    from utils.escape_milvus_string_utils import ExprBuildError, build_filter

    try:
        return build_filter(city=city)
    except ExprBuildError:
        return ""
