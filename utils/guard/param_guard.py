"""参数护栏（第 2 层）。

所有「会被拼进 Milvus 表达式 / 拼进 URL / 传给地图 API」的取值，必须先过这里。
校验失败**抛异常**而不是静默取默认值——静默兜底会让错误数据一路流到答案里。
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

from common.config import CONTENT_TYPES

_CITY_RE = re.compile(r"^[\u4e00-\u9fa5A-Za-z·\-]{2,12}$")


class ParamError(ValueError):
    """参数非法。API 层捕获后返回 400。"""


# ---------------- 城市与枚举 ----------------
def validate_city(city: str | None, allow_none: bool = True) -> str | None:
    from utils.geo.city_gazetteer import is_known_city, normalize_city

    if not city:
        if allow_none:
            return None
        raise ParamError("城市不能为空")
    normalized = normalize_city(city)
    if not _CITY_RE.match(normalized):
        raise ParamError(f"城市格式非法：{city}")
    if not is_known_city(normalized):
        raise ParamError(f"不支持的城市：{city}")
    return normalized


def validate_content_types(types: Sequence[str] | None) -> list[str]:
    if not types:
        return []
    bad = [t for t in types if t not in CONTENT_TYPES]
    if bad:
        raise ParamError(f"未知内容类型：{bad}")
    return list(dict.fromkeys(types))


def validate_top_k(top_k: int | None, default: int = 20, upper: int = 50) -> int:
    if top_k is None:
        return default
    if not isinstance(top_k, int) or top_k <= 0:
        raise ParamError("top_k 必须是正整数")
    return min(top_k, upper)


# ---------------- 地理坐标 ----------------
def validate_lnglat(lng: float, lat: float) -> tuple[float, float]:
    """高德的坐标系是 GCJ-02，范围仍是常规经纬度；越界说明前端传错了顺序。"""
    try:
        lng_f, lat_f = float(lng), float(lat)
    except (TypeError, ValueError) as exc:
        raise ParamError("经纬度必须是数字") from exc
    if not (-180 <= lng_f <= 180) or not (-90 <= lat_f <= 90):
        raise ParamError(f"经纬度越界：{lng},{lat}")
    # 中国境内大致范围，超出的直接拒绝（避免把国外坐标送给国内地图接口）
    if not (73 <= lng_f <= 136 and 3 <= lat_f <= 54):
        raise ParamError(f"坐标不在服务范围内：{lng},{lat}")
    return lng_f, lat_f


def parse_lnglat(text: str) -> tuple[float, float]:
    """解析 `lng,lat` 字符串（高德接口的通用格式）。"""
    parts = re.split(r"[,，\s]+", (text or "").strip())
    if len(parts) != 2:
        raise ParamError(f"坐标格式应为 lng,lat：{text}")
    return validate_lnglat(parts[0], parts[1])


# ---------------- 行程参数 ----------------
def validate_days(days: int | None, default: int = 3, upper: int = 10) -> int:
    if days is None:
        return default
    try:
        d = int(days)
    except (TypeError, ValueError) as exc:
        raise ParamError("天数必须是整数") from exc
    if d <= 0:
        raise ParamError("天数必须大于 0")
    if d > upper:
        raise ParamError(f"暂不支持超过 {upper} 天的行程")
    return d


def validate_budget(budget: float | None) -> float | None:
    if budget is None:
        return None
    try:
        b = float(budget)
    except (TypeError, ValueError) as exc:
        raise ParamError("预算必须是数字") from exc
    if b <= 0:
        raise ParamError("预算必须大于 0")
    if b > 1_000_000:
        raise ParamError("预算数值过大，请确认单位（元）")
    return b


def validate_pace(pace: str | None) -> str:
    allowed = {"轻松", "适中", "紧凑"}
    if not pace:
        return "适中"
    if pace not in allowed:
        raise ParamError(f"节奏取值应为 {allowed} 之一")
    return pace


# ---------------- 文本 ----------------
def sanitize_text(value: str, max_len: int = 200) -> str:
    """去掉控制字符并限长，用于写入日志与展示。"""
    cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", str(value or "")).strip()
    return cleaned[:max_len]


def validate_user_id(user_id: str | None) -> str | None:
    if not user_id:
        return None
    if not re.match(r"^[\w\-:.]{1,64}$", user_id):
        raise ParamError("user_id 格式非法")
    return user_id


def validate_session_id(session_id: str | None) -> str:
    if not session_id:
        raise ParamError("session_id 不能为空")
    if not re.match(r"^[\w\-:.]{1,64}$", session_id):
        raise ParamError("session_id 格式非法")
    return session_id


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def dedup_keep_order(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys([i for i in items if i]))
