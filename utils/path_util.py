"""项目路径工具。

统一收口「项目根目录」的推导，避免各模块用 `Path(__file__).parents[n]` 各自数层级。
优先级：环境变量 PROJECT_ROOT > 向上查找 .env > 向上查找 pyproject.toml > 本文件上两级。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 便于识别项目根的候选标识文件
_ROOT_MARKERS = (".env", "pyproject.toml", ".git")


def get_path_dir(ps: int = 0) -> Path:
    """获取当前文件向上第 ps 级的目录（parents[0] 即上一级）。"""
    return Path(__file__).resolve().parents[ps]


def _walk_up(markers: tuple[str, ...]) -> Path | None:
    """从本文件所在目录向上逐级查找标识文件。"""
    current = Path(__file__).resolve().parent
    for parent in (current, *current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    return None


def get_project_root(identifier: str = ".env") -> Path:
    """推导项目根目录。找不到标识文件时退回本文件上两级，不抛异常。"""
    env_root = os.getenv("PROJECT_ROOT")
    if env_root and Path(env_root).expanduser().exists():
        return Path(env_root).expanduser().resolve()

    found = _walk_up((identifier, *_ROOT_MARKERS))
    if found is not None:
        return found

    # utils/path_util.py 的上一级是 utils/，再上一级是项目根
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT: Path = get_project_root()

# 加载根目录 .env（存在才加载，缺失不报错；容器环境通常直接注入环境变量）
_ENV_FILE = PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    load_dotenv(dotenv_path=_ENV_FILE)


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在并返回 Path。"""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(path: str | Path) -> Path:
    """相对路径一律相对项目根解析。"""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def data_dir(*parts: str) -> Path:
    """语料目录下的子路径，不存在时按需创建。

    `data_dir()` 返回 `data/`，`data_dir("uploads")` 返回并创建 `data/uploads/`。
    """
    base = DATA_DIR
    if not parts:
        base.mkdir(parents=True, exist_ok=True)
        return base
    target = base.joinpath(*parts)
    target.mkdir(parents=True, exist_ok=True)
    return target


def output_dir(*parts: str) -> Path:
    target = OUTPUT_DIR.joinpath(*parts) if parts else OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


# 常用目录
DATA_DIR = PROJECT_ROOT / "data"
PAGE_DIR = PROJECT_ROOT / "page"
PROMPT_DIR = PROJECT_ROOT / "common" / "prompt"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "PAGE_DIR",
    "PROMPT_DIR",
    "OUTPUT_DIR",
    "LOG_DIR",
    "get_project_root",
    "get_path_dir",
    "ensure_dir",
    "resolve_path",
    "data_dir",
    "output_dir",
]
