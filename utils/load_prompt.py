"""提示词加载与渲染。

约定：`common/prompt/*.prompt` 是纯文本模板，占位符用 `{name}`。
因为内部走 `str.format`，模板里如果要写 JSON 示例，花括号必须写成 `{{ }}`。

渲染失败（模板里漏了变量 / 多了变量）时抛 `PromptRenderError`，
并带上缺失变量名——这类错误必须在开发期暴露，不能静默降级成半截提示词。
"""
from __future__ import annotations

import string
from pathlib import Path
from typing import Any

from common.logging.logger import logger
from utils.path_util import PROMPT_DIR


class PromptRenderError(RuntimeError):
    """提示词渲染失败。"""


# 进程内缓存原始模板，避免每次请求都读盘
_TEMPLATE_CACHE: dict[str, str] = {}


def _prompt_path(name: str) -> Path:
    file_name = name if name.endswith(".prompt") else f"{name}.prompt"
    return PROMPT_DIR / file_name


def load_raw_prompt(name: str, use_cache: bool = True) -> str:
    """读取未渲染的模板原文。"""
    if use_cache and name in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[name]

    path = _prompt_path(name)
    if not path.exists():
        raise FileNotFoundError(f"提示词文件不存在：{path}")

    raw = path.read_text(encoding="utf-8")
    if use_cache:
        _TEMPLATE_CACHE[name] = raw
    return raw


def list_placeholders(name: str) -> set[str]:
    """列出模板里声明的占位符，可用于启动期自检。"""
    raw = load_raw_prompt(name)
    return {field for _, field, _, _ in string.Formatter().parse(raw) if field}


def load_prompt(name: str, **kwargs: Any) -> str:
    """渲染模板。未传的占位符会报错，而不是留下 `{xxx}` 交给模型。"""
    raw = load_raw_prompt(name)
    try:
        return raw.format(**kwargs)
    except KeyError as exc:
        missing = exc.args[0] if exc.args else "unknown"
        raise PromptRenderError(f"提示词 {name} 缺少变量：{missing}") from exc
    except (IndexError, ValueError) as exc:
        raise PromptRenderError(f"提示词 {name} 渲染失败：{exc}") from exc


def render_safe(name: str, fallback: str = "", **kwargs: Any) -> str:
    """不抛异常的渲染，用于「提示词坏了也不能让请求挂掉」的兜底场景。"""
    try:
        return load_prompt(name, **kwargs)
    except Exception:
        logger.exception(f"提示词 {name} 渲染失败，使用兜底文本")
        return fallback


def clear_cache() -> None:
    """清空模板缓存（改完 prompt 热更新用）。"""
    _TEMPLATE_CACHE.clear()
