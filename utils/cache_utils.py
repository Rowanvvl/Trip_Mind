"""进程内 TTL 缓存。

为什么自己写而不上 Redis：地图/天气/图片三路都是**短 TTL + 强幂等**的读，
单机命中率目标 >70%（框架文档 11.6），进程内字典足够，且少一个运维组件。
多实例部署时把 `get_cache()` 换成 Redis 实现即可，接口不用动。

统计口径：
- hit / miss / expired 三个计数器对外暴露，`/stats` 直接读；
- 命中率是工具治理的唯一硬指标——没命中率的缓存等于没做。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, Optional, Tuple


@dataclass
class CacheStats:
    hit: int = 0
    miss: int = 0
    expired: int = 0
    evicted: int = 0
    set: int = 0

    @property
    def total(self) -> int:
        return self.hit + self.miss

    @property
    def hit_rate(self) -> float:
        return round(self.hit / self.total, 4) if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "hit": self.hit,
            "miss": self.miss,
            "expired": self.expired,
            "evicted": self.evicted,
            "set": self.set,
            "hit_rate": self.hit_rate,
        }


@dataclass
class _Entry:
    value: Any
    expire_at: float
    created_at: float = field(default_factory=time.time)


class TTLCache:
    """线程安全的 TTL 缓存，带容量上限与 LRU 式淘汰。"""

    def __init__(self, max_size: int = 4096, namespace: str = "default"):
        self.max_size = max_size
        self.namespace = namespace
        self._data: Dict[Hashable, _Entry] = {}
        self._lock = threading.RLock()
        self.stats = CacheStats()

    # ---------------- 基本读写 ----------------
    def get(self, key: Hashable, default: Any = None) -> Any:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.stats.miss += 1
                return default
            if entry.expire_at < time.time():
                self._data.pop(key, None)
                self.stats.expired += 1
                self.stats.miss += 1
                return default
            self.stats.hit += 1
            return entry.value

    def set(self, key: Hashable, value: Any, ttl: int) -> None:
        if ttl <= 0:
            return
        with self._lock:
            if len(self._data) >= self.max_size and key not in self._data:
                self._evict_locked()
            self._data[key] = _Entry(value=value, expire_at=time.time() + ttl)
            self.stats.set += 1

    def get_or_set(self, key: Hashable, ttl: int, producer: Callable[[], Any]) -> Any:
        """先查缓存，未命中再执行 producer。

        注意：producer 抛异常时**不写缓存**，避免把 null 当成结果缓存一整天。
        """
        sentinel = object()
        cached = self.get(key, sentinel)
        if cached is not sentinel:
            return cached
        value = producer()
        if value is not None:
            self.set(key, value, ttl)
        return value

    def delete(self, key: Hashable) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def _evict_locked(self) -> None:
        """先清过期，再按最早过期时间淘汰 10%。"""
        now_ts = time.time()
        expired = [k for k, e in self._data.items() if e.expire_at < now_ts]
        for k in expired:
            self._data.pop(k, None)
        self.stats.expired += len(expired)
        if len(self._data) < self.max_size:
            return
        victims = sorted(self._data.items(), key=lambda kv: kv[1].expire_at)[: max(1, self.max_size // 10)]
        for k, _ in victims:
            self._data.pop(k, None)
        self.stats.evicted += len(victims)

    # ---------------- 观测 ----------------
    def size(self) -> int:
        with self._lock:
            return len(self._data)

    def snapshot(self) -> dict:
        return {"namespace": self.namespace, "size": self.size(), **self.stats.to_dict()}


# 全局默认缓存 + 按用途分命名空间，便于分别看命中率
_default_cache = TTLCache(namespace="default")
_namespaced: Dict[str, TTLCache] = {"default": _default_cache}
_namespaced_lock = threading.Lock()


def get_cache(namespace: str = "default") -> TTLCache:
    """按用途取缓存实例：map / weather / image / geo / llm。"""
    with _namespaced_lock:
        cache = _namespaced.get(namespace)
        if cache is None:
            cache = TTLCache(namespace=namespace)
            _namespaced[namespace] = cache
        return cache


def cache_snapshot() -> Dict[str, dict]:
    with _namespaced_lock:
        return {name: cache.snapshot() for name, cache in _namespaced.items()}


def make_key(*parts: Any) -> str:
    """稳定的复合 key：把坐标按 3 位小数取整，避免「同一地点不同精度」重复请求。"""
    tokens = []
    for p in parts:
        if isinstance(p, float):
            tokens.append(f"{p:.3f}")
        else:
            tokens.append(str(p).strip().lower())
    return "|".join(tokens)


def memoize(namespace: str, ttl: int, key_func: Optional[Callable[..., str]] = None):
    """函数级缓存装饰器，给工具函数用。"""

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            key = key_func(*args, **kwargs) if key_func else make_key(func.__name__, *args, *(f"{k}={v}" for k, v in sorted(kwargs.items())))
            return get_cache(namespace).get_or_set(key, ttl, lambda: func(*args, **kwargs))

        wrapper.__name__ = getattr(func, "__name__", "wrapped")
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


def reset_all() -> None:
    """测试用：清空所有命名空间与计数。"""
    with _namespaced_lock:
        for cache in _namespaced.values():
            cache.clear()
            cache.stats = CacheStats()


__all__ = [
    "TTLCache",
    "CacheStats",
    "get_cache",
    "cache_snapshot",
    "make_key",
    "memoize",
    "reset_all",
]
