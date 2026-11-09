"""跨会话长期偏好（Memory 层）。

写入门槛是这个模块的全部重点：
- `explicit`（默认）：只写用户**明确确认**的信息（「记住我不吃辣」或点击了「记住」）；
- `inferred`：允许写模型推断的偏好，但必须带 `confidence >= 0.8` 且 `confirmed=False`，
  前端展示时要标「推测」，用户可一键删除。

任何情况下都不自动写入敏感信息（证件号、手机号、病史）。
没有写入策略的"记忆"等于给后续请求埋雷：一次误记会污染这个用户之后所有回答。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

from common.config import trip_config
from utils.format_utils import now_iso

# 允许写入长期记忆的偏好字段
PROFILE_KEYS: Dict[str, str] = {
    "city_home": "常住城市",
    "preferred_cities": "常去目的地",
    "budget_level": "习惯的消费档次",
    "pace": "偏好的节奏",
    "diet_avoid": "饮食禁忌",
    "companions": "常同行的人群",
    "transport": "交通偏好",
    "lodging_level": "住宿偏好",
    "notes": "其他稳定偏好",
}

# 永不写入的字段（哪怕用户主动提供）
FORBIDDEN_KEYS = {"id_card", "phone", "passport", "medical_history", "password"}

LIST_PROFILE_KEYS = {"preferred_cities", "diet_avoid", "companions"}


def empty_profile(user_id: str = "") -> Dict[str, Any]:
    return {
        "user_id": user_id,
        "city_home": None,
        "preferred_cities": [],
        "budget_level": None,
        "pace": None,
        "diet_avoid": [],
        "companions": [],
        "transport": None,
        "lodging_level": None,
        "notes": [],
        "confirmed": False,
        "confidence": 1.0,
        "updated_at": None,
        "source": "explicit",
    }


def can_write(confirmed: bool = False, confidence: float = 1.0, mode: str | None = None) -> bool:
    """写入门槛。默认模式只接受用户确认过的信息。"""
    mode = mode or trip_config.profile_write_mode
    if mode == "explicit":
        return bool(confirmed)
    if mode == "inferred":
        return bool(confirmed) or confidence >= 0.8
    return False


def sanitize_candidate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """过滤非法字段与空值，禁止写入敏感键。"""
    clean: Dict[str, Any] = {}
    for key, value in (payload or {}).items():
        if key in FORBIDDEN_KEYS or key not in PROFILE_KEYS:
            continue
        if value in (None, "", [], {}):
            continue
        clean[key] = value
    return clean


def merge_profile(
    profile: Dict[str, Any] | None,
    updates: Dict[str, Any],
    confirmed: bool = False,
    confidence: float = 1.0,
) -> Dict[str, Any]:
    """合并偏好。返回新 dict，不改入参。

    未达写入门槛时**原样返回**，并把 `rejected` 写到 `_meta` 里，便于排查
    「为什么说了不吃辣下次还是没记住」。
    """
    base = dict(profile or empty_profile())
    base.setdefault("_meta", {})

    clean = sanitize_candidate(updates)
    if not clean:
        base["_meta"] = {"rejected": "无可写字段"}
        return base

    if not can_write(confirmed=confirmed, confidence=confidence):
        base["_meta"] = {"rejected": "未达写入门槛（需用户确认）", "candidate": clean}
        return base

    for key, value in clean.items():
        if key in LIST_PROFILE_KEYS:
            merged = list(base.get(key) or [])
            for item in value if isinstance(value, list) else [value]:
                if item and item not in merged:
                    merged.append(item)
            base[key] = merged
        else:
            base[key] = value

    base["confirmed"] = bool(confirmed) or bool(base.get("confirmed"))
    base["confidence"] = max(float(base.get("confidence") or 0), min(float(confidence), 1.0))
    base["source"] = "explicit" if confirmed else "inferred"
    base["updated_at"] = now_iso()
    base["_meta"] = {"written": list(clean.keys())}
    return base


def profile_to_text(profile: Dict[str, Any] | None) -> str:
    """拼进 prompt 的偏好文本。推断类内容显式标注，避免模型当成事实。"""
    if not profile:
        return "（暂无长期偏好）"
    lines: List[str] = []
    for key, label in PROFILE_KEYS.items():
        value = profile.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            value = "、".join(str(v) for v in value)
        lines.append(f"- {label}：{value}")
    if not lines:
        return "（暂无长期偏好）"
    if profile.get("source") == "inferred":
        lines.insert(0, "（以下偏好为推测，未经用户确认，仅作参考）")
    return "\n".join(lines)


def extract_profile_candidates(slots: Dict[str, Any]) -> Dict[str, Any]:
    """从槽位里挑出可能变成长期偏好的字段（不直接写，先给节点判断）。"""
    mapping = {
        "budget_level": slots.get("budget_level"),
        "pace": slots.get("pace"),
        "diet_avoid": slots.get("diet_avoid"),
        "companions": slots.get("companions"),
        "transport": slots.get("transport"),
    }
    return {k: v for k, v in mapping.items() if v not in (None, "", [], {})}


def delete_keys(profile: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    """用户主动删除某些记忆（前端「忘掉这个」）。"""
    base = dict(profile or {})
    for key in keys:
        if key in base:
            base[key] = [] if key in LIST_PROFILE_KEYS else None
    base["updated_at"] = now_iso()
    return base
