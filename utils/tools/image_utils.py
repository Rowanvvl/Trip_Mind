"""图片：语料图优先，POI 图其次，搜索引擎图只在显式开启时兜底。

优先级不是随便定的：
1. **语料图**（导入时已进 MinIO）——与内容强相关、无版权风险、不会失效；
2. **高德 POI 图**——与地点绑定、有出处、可追溯；
3. **搜索引擎图**——版权与失效风险都高，默认关闭（`ENABLE_WEB_IMAGE=false`）。

两件绝对不做的事：
- 不做图像内容理解（判断「这张图是不是这个景点」需要视觉模型，成本高且不可靠）；
- 图片失败**绝不影响答案**——它挂在答案流式结束之后，独立 SSE 事件下发。
"""
from __future__ import annotations

import time
from typing import Iterable, List, Optional

import httpx

from common.config import minio_config, trip_config
from common.logging.logger import logger
from utils.tools.amap_utils import poi_photos
from utils.tools.tool_registry import call

_VALID_EXT = (".jpg", ".jpeg", ".png", ".webp")


def _head_ok(url: str) -> bool:
    """HEAD 校验：状态 200 且 content-type 是图片。失效链接必须在这里被拦掉。"""
    try:
        with httpx.Client(timeout=0.8, follow_redirects=True) as client:
            resp = client.head(url)
            if resp.status_code != 200:
                return False
            ctype = resp.headers.get("content-type", "")
            return ctype.startswith("image/") or url.lower().endswith(_VALID_EXT)
    except Exception:
        return False


def validate_urls(urls: Iterable[str], limit: int = 3) -> List[str]:
    """去重 + 校验 + 限量。校验结果进缓存，避免同一张图反复 HEAD。"""
    seen, out = set(), []
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        if _head_ok(url):
            out.append(url)
        if len(out) >= limit:
            break
    return out


def from_corpus(object_names: Iterable[str]) -> List[str]:
    """把 MinIO 里的对象名转成可访问 URL。"""
    return [minio_config.object_url(name) for name in object_names if name]


def from_poi(name: str, city: str | None = None, limit: int = 3) -> List[str]:
    result = poi_photos(name, city, limit=limit)
    if not result.ok:
        return []
    return list(result.data or [])[:limit]


def from_web(keyword: str, limit: int = 2) -> List[str]:
    """搜索引擎图兜底。默认关闭——打开前请确认版权与稳定性。"""
    if not trip_config.enable_web_image:
        return []
    logger.debug(f"[image] web 图兜底未实现 provider，跳过 keyword={keyword}")
    return []


def attach_for_entities(
    entities: List[dict],
    corpus_images: Optional[List[str]] = None,
    city: str | None = None,
    limit: int | None = None,
) -> dict:
    """给一组实体挂图。

    返回 `{items: [...], degraded: bool}`；items 每项带 `source` 与 `caption`，
    前端据此标注图片来源，不能把 POI 图说成官方图。
    """
    limit = limit or trip_config.image_max_per_answer
    items: List[dict] = []

    for url in validate_urls(corpus_images or [], limit=limit):
        items.append({"url": url, "source": "语料", "caption": "攻略配图"})

    remaining = limit - len(items)
    if remaining > 0:
        for entity in entities[:remaining]:
            name = entity.get("name") or entity.get("title")
            if not name:
                continue
            for url in validate_urls(from_poi(name, city, remaining), limit=remaining):
                items.append({"url": url, "source": "高德", "caption": name})
                remaining -= 1
                if remaining <= 0:
                    break

    return {
        "items": items[:limit],
        "degraded": len(items) == 0,
        "fetched_at": time.time(),
    }


def attach_safe(entities: List[dict], corpus_images: Optional[List[str]] = None, city: str | None = None) -> dict:
    """给节点用的安全版本：任何异常都吞掉，返回空结果。"""
    try:
        return attach_for_entities(entities, corpus_images, city)
    except Exception as exc:
        logger.warning(f"[image] 挂图失败，静默跳过：{exc}")
        return {"items": [], "degraded": True, "error": str(exc)}
