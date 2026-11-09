"""外部工具治理：超时、重试、熔断、配额、缓存、指标。

所有出网调用（地图 / 天气 / 图片 / 网络搜索）都必须从 `call()` 走，不允许在节点里
直接 `httpx.get`。原因很现实：地图一个接口超时 5 秒，整个问答就废了；
而地图接口偶发超时是常态，没有熔断就会出现「每个请求都等 5 秒」的雪崩。

四道闸：
1. **超时**：地图/天气 1.2s、图片 0.8s（配置里可调）；
2. **重试**：幂等 GET 重试 1 次，且只在超时/连接错误时重试，业务错误不重试；
3. **熔断**：连续 5 次失败 → 打开 60s，期间直接返回失败，不打网络；
4. **配额**：按日计数，超限直接降级（免费额度撑不住时，宁可不给也不能被封）。

统一返回 `ToolResult`，调用方只需判断 `ok`，不用 try/except。
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, Optional

from common.logging.logger import logger
from utils.cache_utils import get_cache, make_key

_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")


@dataclass(frozen=True)
class ToolPolicy:
    name: str
    timeout: float = 1.2
    retries: int = 1
    breaker_threshold: int = 5
    breaker_cooldown: float = 60.0
    cache_namespace: str = ""
    cache_ttl: int = 0
    daily_quota: int = 10000


# 各工具的治理策略，集中一处配置。改这里就等于改全局行为。
POLICIES: Dict[str, ToolPolicy] = {
    "amap_geocode": ToolPolicy("amap_geocode", 1.2, 1, 5, 60, "geo", 86400 * 7, 5000),
    "amap_place": ToolPolicy("amap_place", 1.2, 1, 5, 60, "map", 86400 * 7, 5000),
    "amap_direction": ToolPolicy("amap_direction", 1.5, 1, 5, 60, "map", 21600, 3000),
    "amap_distance": ToolPolicy("amap_distance", 1.2, 1, 5, 60, "map", 21600, 3000),
    "amap_weather": ToolPolicy("amap_weather", 1.2, 1, 5, 60, "weather", 1800, 2000),
    "amap_poi_photo": ToolPolicy("amap_poi_photo", 0.8, 0, 5, 60, "image", 86400, 1000),
    "web_search": ToolPolicy("web_search", 3.0, 0, 3, 120, "web", 3600, 500),
    "web_image": ToolPolicy("web_image", 0.8, 0, 3, 120, "", 0, 200),
}


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str = ""
    cost_ms: int = 0
    cached: bool = False
    estimated: bool = False      # 是否为本地估算值（而非真实查询）
    source: str = ""
    fetched_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "error": self.error,
            "cost_ms": self.cost_ms,
            "cached": self.cached,
            "estimated": self.estimated,
            "source": self.source,
            "fetched_at": self.fetched_at,
        }


@dataclass
class _ToolState:
    failures: int = 0            # 连续失败次数
    open_until: float = 0.0      # 熔断截止时间
    day: date = field(default_factory=date.today)
    calls_today: int = 0
    ok_total: int = 0
    fail_total: int = 0
    timeout_total: int = 0
    latency_total_ms: int = 0

    def roll_day(self) -> None:
        today = date.today()
        if today != self.day:
            self.day = today
            self.calls_today = 0


_states: Dict[str, _ToolState] = {}


def _state(name: str) -> _ToolState:
    state = _states.get(name)
    if state is None:
        state = _ToolState()
        _states[name] = state
    state.roll_day()
    return state


def _policy(name: str) -> ToolPolicy:
    return POLICIES.get(name, ToolPolicy(name))


def is_open(name: str) -> bool:
    """熔断是否处于打开状态。"""
    return time.time() < _state(name).open_until


def _record_success(state: _ToolState, cost_ms: int) -> None:
    state.failures = 0
    state.open_until = 0.0
    state.ok_total += 1
    state.latency_total_ms += cost_ms


def _record_failure(name: str, state: _ToolState, cost_ms: int, timeout: bool) -> None:
    state.failures += 1
    state.fail_total += 1
    state.latency_total_ms += cost_ms
    if timeout:
        state.timeout_total += 1
    policy = _policy(name)
    if state.failures >= policy.breaker_threshold:
        state.open_until = time.time() + policy.breaker_cooldown
        logger.warning(f"[tool] {name} 连续失败 {state.failures} 次，熔断 {policy.breaker_cooldown:.0f}s")


def call(
    name: str,
    func: Callable[..., Any],
    *args: Any,
    cache_key: Optional[str] = None,
    use_cache: bool = True,
    **kwargs: Any,
) -> ToolResult:
    """执行一次外部调用，任何异常都转成 `ToolResult(ok=False)`。"""
    policy = _policy(name)
    state = _state(name)

    # 1. 配额
    if state.calls_today >= policy.daily_quota:
        logger.warning(f"[tool] {name} 触发日配额 {policy.daily_quota}")
        return ToolResult(ok=False, error="quota_exceeded", source=name)

    # 2. 熔断
    if time.time() < state.open_until:
        return ToolResult(ok=False, error="circuit_open", source=name)

    # 3. 缓存
    key = cache_key
    if use_cache and policy.cache_ttl > 0:
        key = key or make_key(name, *args, *sorted(kwargs.items()))
        cached = get_cache(policy.cache_namespace or "default").get(key)
        if cached is not None:
            return ToolResult(ok=True, data=cached, cached=True, source=name)

    # 4. 带超时与重试的执行
    started = time.perf_counter()
    last_error = ""
    for attempt in range(policy.retries + 1):
        state.calls_today += 1
        future = _EXECUTOR.submit(func, *args, **kwargs)
        try:
            data = future.result(timeout=policy.timeout)
        except FutureTimeout:
            last_error = "timeout"
            cost_ms = int((time.perf_counter() - started) * 1000)
            logger.warning(f"[tool] {name} 超时（{policy.timeout}s，第 {attempt + 1} 次）")
            if attempt >= policy.retries:
                _record_failure(name, state, cost_ms, timeout=True)
                return ToolResult(ok=False, error="timeout", cost_ms=cost_ms, source=name)
            continue
        except Exception as exc:  # 网络/解析错误
            last_error = f"{type(exc).__name__}: {exc}"
            cost_ms = int((time.perf_counter() - started) * 1000)
            logger.warning(f"[tool] {name} 失败：{last_error}")
            if attempt >= policy.retries:
                _record_failure(name, state, cost_ms, timeout=False)
                return ToolResult(ok=False, error=last_error, cost_ms=cost_ms, source=name)
            continue

        cost_ms = int((time.perf_counter() - started) * 1000)
        _record_success(state, cost_ms)

        if data is None:
            # 工具正常返回但没数据（例如 POI 无匹配），不算失败，但也不是有效结果
            return ToolResult(ok=False, data=None, error="empty", cost_ms=cost_ms, source=name)

        if use_cache and policy.cache_ttl > 0 and key:
            get_cache(policy.cache_namespace or "default").set(key, data, policy.cache_ttl)
        return ToolResult(ok=True, data=data, cost_ms=cost_ms, source=name)

    return ToolResult(ok=False, error=last_error or "unknown", source=name)


def estimated_result(data: Any, reason: str = "") -> ToolResult:
    """本地估算结果的包装：`estimated=True`，答案里必须标注「估算」。"""
    return ToolResult(ok=True, data=data, estimated=True, source=reason or "local_estimate")


def metrics() -> Dict[str, dict]:
    out = {}
    for name, state in _states.items():
        avg = state.latency_total_ms / max(state.ok_total + state.fail_total, 1)
        out[name] = {
            "ok": state.ok_total,
            "fail": state.fail_total,
            "timeout": state.timeout_total,
            "calls_today": state.calls_today,
            "avg_ms": int(avg),
            "breaker_open": time.time() < state.open_until,
            "failures": state.failures,
        }
    return out


def reset_breaker(name: Optional[str] = None) -> None:
    targets = [name] if name else list(_states.keys())
    for target in targets:
        state = _state(target)
        state.failures = 0
        state.open_until = 0.0


def reset() -> None:
    _states.clear()


def policies() -> Dict[str, dict]:
    return {
        name: {
            "timeout": p.timeout,
            "retries": p.retries,
            "cache_ttl": p.cache_ttl,
            "daily_quota": p.daily_quota,
        }
        for name, p in POLICIES.items()
    }
