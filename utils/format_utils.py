"""展示与格式化工具：时间、耗时、金额区间、引用编号。

这类函数被 API 层、trace、前端回显三处共用，必须只有一个实现，
否则「6 小时前」和「2026-11-09 04:00」会在不同页面同时出现。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

CN_TZ = timezone(timedelta(hours=8))


def now() -> datetime:
    """统一时区（东八区）的当前时间，避免容器 UTC 造成「今天」判断错位。"""
    return datetime.now(CN_TZ)


def now_iso() -> str:
    return now().isoformat(timespec="seconds")


def parse_ts(value) -> datetime | None:
    """宽松解析时间戳：datetime / ISO 字符串 / 秒级与毫秒级时间戳。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=CN_TZ)
    if isinstance(value, (int, float)):
        seconds = value / 1000 if value > 1e11 else value
        return datetime.fromtimestamp(seconds, CN_TZ)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=CN_TZ)


def age_hours(value) -> float | None:
    """距今多少小时。用于「事实是否过期」判定（12.2 第 2 条）。"""
    dt = parse_ts(value)
    if dt is None:
        return None
    return round((now() - dt).total_seconds() / 3600, 2)


def is_fresh(value, max_age_hours: float) -> bool:
    age = age_hours(value)
    return age is not None and age <= max_age_hours


def human_duration(minutes: float) -> str:
    """分钟 → 「1 小时 20 分」。"""
    minutes = int(round(minutes or 0))
    if minutes < 60:
        return f"{minutes} 分钟"
    hours, rest = divmod(minutes, 60)
    return f"{hours} 小时" if rest == 0 else f"{hours} 小时 {rest} 分"


def human_distance(meters: float) -> str:
    meters = float(meters or 0)
    return f"{meters / 1000:.1f} 公里" if meters >= 1000 else f"{int(meters)} 米"


def money_range(low: float, high: float, unit: str = "元") -> str:
    """金额区间。不做四舍五入到「整百」，那会掩盖测算精度。"""
    low, high = float(low or 0), float(high or 0)
    if high < low:
        low, high = high, low
    if abs(high - low) < 1:
        return f"约 {int(round(low))} {unit}"
    return f"{int(round(low))}–{int(round(high))} {unit}"


def format_citations(items: Sequence[dict], style: str = "inline") -> str:
    """把引用列表渲染成 `[1] 三亚景点推荐 · 东线西线怎么分` 形式。"""
    lines = []
    for idx, item in enumerate(items, start=1):
        title = item.get("doc_title") or item.get("file_name") or "未知文档"
        section = item.get("section") or item.get("heading") or ""
        city = item.get("city") or ""
        head = f"[{idx}] {city}·{title}" if city else f"[{idx}] {title}"
        lines.append(f"{head} · {section}" if section else head)
    return "\n".join(lines) if style == "block" else "；".join(lines)


def join_nonempty(parts: Iterable[str], sep: str = " · ") -> str:
    return sep.join([str(p).strip() for p in parts if p and str(p).strip()])


def truncate(text: str, limit: int = 120, suffix: str = "…") -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: max(limit - 1, 0)] + suffix
