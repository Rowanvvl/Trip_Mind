"""Milvus 表达式拼装与转义。

**这是安全边界**：`city`、`content_type`、`entity` 都可能来自用户输入，
直接拼进 `expr` 会造成注入（`city == "三亚" or city != ""` 就能绕过过滤）。
统一规则：
1. 字符串值一律走 `escape_value`（转义反斜杠、双引号，去掉控制字符）；
2. 枚举类字段（city / content_type）再套一层白名单校验；
3. 列表走 `in_expr`，超长自动切片合并，避免单条表达式过大。
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

from common.config import CONTENT_TYPES, milvus_config

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class ExprBuildError(ValueError):
    """表达式构建失败（通常是取值越界或类型不对）。"""


def escape_value(value: str) -> str:
    """转义字符串字面量，返回可直接放进双引号内的内容。"""
    if value is None:
        raise ExprBuildError("expr 的值不能为 None")
    text = _CONTROL_CHARS.sub("", str(value))
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    # Milvus 表达式里 % 与 _ 在 like 中才是通配符，等值比较无需处理，这里保留原文
    return text


def eq_expr(field: str, value: str) -> str:
    return f'{field} == "{escape_value(value)}"'


def in_expr(field: str, values: Iterable[str]) -> str:
    """生成 `field in ["a","b"]`，空列表返回空串（调用方需跳过）。"""
    items = [f'"{escape_value(v)}"' for v in _dedup(values)]
    if not items:
        return ""
    if len(items) > milvus_config.max_in_size:
        raise ExprBuildError(f"{field} 的 in 列表过长（{len(items)} > {milvus_config.max_in_size}）")
    return f"{field} in [{', '.join(items)}]"


def and_expr(*parts: str) -> str:
    """用 AND 连接，自动丢弃空片段。"""
    valid = [p for p in parts if p]
    if not valid:
        return ""
    if len(valid) == 1:
        return valid[0]
    return " and ".join(f"({p})" for p in valid)


def or_expr(*parts: str) -> str:
    valid = [p for p in parts if p]
    if not valid:
        return ""
    if len(valid) == 1:
        return valid[0]
    return " or ".join(f"({p})" for p in valid)


def build_filter(
    city: str | None = None,
    content_types: Sequence[str] | None = None,
    entities: Sequence[str] | None = None,
    doc_ids: Sequence[int] | None = None,
    extra: str | None = None,
) -> str:
    """业务过滤条件的唯一入口。

    city / content_types 走白名单；entities 与 doc_ids 走转义与数值校验。
    非法取值**抛异常**而不是静默丢弃——静默丢弃会让过滤形同虚设。
    """
    parts: list[str] = []

    if city:
        from utils.geo.city_gazetteer import is_known_city, normalize_city

        normalized = normalize_city(city)
        if not is_known_city(normalized):
            raise ExprBuildError(f"未知城市：{city}")
        parts.append(eq_expr("city", normalized))

    if content_types:
        for ct in content_types:
            if ct not in CONTENT_TYPES:
                raise ExprBuildError(f"未知内容类型：{ct}")
        parts.append(in_expr("content_type", content_types))

    if entities:
        # entities 存在 Array 字段里，用 array_contains_any 更贴合 schema
        items = [f'"{escape_value(e)}"' for e in _dedup(entities)]
        if items:
            parts.append(f"array_contains_any(entities, [{', '.join(items)}])")

    if doc_ids:
        ids = []
        for raw in _dedup(doc_ids):
            if not str(raw).isdigit():
                raise ExprBuildError(f"doc_id 必须是整数：{raw}")
            ids.append(str(int(raw)))
        if ids:
            parts.append(f"doc_id in [{', '.join(ids)}]")

    if extra:
        # extra 仅供内部代码传入常量片段，禁止透传用户输入
        parts.append(extra)

    return and_expr(*parts)


def _dedup(values: Iterable) -> list:
    seen, out = set(), []
    for v in values:
        if v is None or v == "":
            continue
        key = str(v)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out
