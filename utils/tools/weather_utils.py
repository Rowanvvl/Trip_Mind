"""天气：实时 + 3 日预报。

必须说清的能力边界（这一点会直接影响用户预期）：
- 高德 `weatherInfo` 只有**实时**与**今天起 3 天**预报，**没有 7 日、没有历史**；
- 所以能回答「这周末能下海吗」，不能回答「下个月初会不会下雨」——
  遇到超范围的问法要明确拒绝，而不是编一个温度出来；
- 每条结果都带 `fetched_at`，超过 `FACT_MAX_AGE_H` 的事实不用于「明天」类判断。

对行程的价值：雨天替换必须是**真的替换**（在 `node_time_budget` 里换室内项目），
只在提示词里写「如遇雨天建议室内」等于没做。
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import httpx

from common.config import amap_config, weather_config
from common.logging.logger import logger
from utils.tools.amap_utils import adcode_of
from utils.tools.tool_registry import call


def _get(path: str, params: Dict) -> Optional[dict]:
    if not amap_config.key:
        return None
    query = {**params, "key": amap_config.key, "output": "JSON"}
    with httpx.Client(timeout=weather_config.timeout) as client:
        resp = client.get(f"{amap_config.host}{path}", params=query)
        resp.raise_for_status()
        payload = resp.json()
    if str(payload.get("status")) != "1":
        logger.warning(f"[weather] 返回异常：{payload.get('info')}")
        return None
    return payload


def _normalize_live(raw: dict) -> dict:
    return {
        "province": raw.get("province"),
        "city": raw.get("city"),
        "weather": raw.get("weather"),
        "temperature": _to_int(raw.get("temperature")),
        "wind_direction": raw.get("winddirection"),
        "wind_power": raw.get("windpower"),
        "humidity": _to_int(raw.get("humidity")),
        "report_time": raw.get("reporttime"),
    }


def _normalize_cast(raw: dict) -> dict:
    return {
        "date": raw.get("date"),
        "week": raw.get("week"),
        "day_weather": raw.get("dayweather"),
        "night_weather": raw.get("nightweather"),
        "day_temp": _to_int(raw.get("daytemp")),
        "night_temp": _to_int(raw.get("nighttemp")),
        "day_wind": raw.get("daywind"),
        "day_power": raw.get("daypower"),
    }


def _to_int(value) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def get_weather(city: str) -> dict:
    """返回归一化天气。查不到时 `degraded=True, forecast=[]`，不抛异常。"""
    adcode = adcode_of(city)
    if not adcode:
        return {
            "city": city,
            "live": None,
            "forecast": [],
            "coverage_days": 0,
            "degraded": True,
            "error": "adcode_unavailable",
            "source": "高德天气",
            "fetched_at": time.time(),
        }

    def _live():
        payload = _get("/v3/weather/weatherInfo", {"city": adcode, "extensions": "base"})
        lives = (payload or {}).get("lives") or []
        return _normalize_live(lives[0]) if lives else None

    def _forecast():
        payload = _get("/v3/weather/weatherInfo", {"city": adcode, "extensions": "all"})
        forecasts = (payload or {}).get("forecasts") or []
        if not forecasts:
            return None
        casts = forecasts[0].get("casts") or []
        return [_normalize_cast(c) for c in casts]

    live_result = call("amap_weather", _live, cache_key=f"weather:live:{adcode}")
    cast_result = call("amap_weather", _forecast, cache_key=f"weather:cast:{adcode}")

    forecast = cast_result.data if cast_result.ok and isinstance(cast_result.data, list) else []
    live = live_result.data if live_result.ok else None

    return {
        "city": city,
        "adcode": adcode,
        "live": live,
        "forecast": forecast[: weather_config.forecast_days],
        "coverage_days": len(forecast[: weather_config.forecast_days]),
        "degraded": not (live or forecast),
        "error": "" if (live or forecast) else (live_result.error or cast_result.error),
        "source": "高德天气",
        "fetched_at": time.time(),
    }


def is_rainy(weather_text: str | None) -> bool:
    return any(k in (weather_text or "") for k in weather_config.rain_keywords)


def is_bad_outdoor(weather_text: str | None) -> bool:
    return any(k in (weather_text or "") for k in weather_config.bad_outdoor_keywords)


def rain_days(forecast: List[dict]) -> List[str]:
    """预报里的雨天日期，行程替换用。"""
    return [d["date"] for d in forecast if is_rainy(d.get("day_weather")) or is_rainy(d.get("night_weather"))]


def outdoor_risk_ratio(forecast: List[dict]) -> float:
    """预报期内户外不利天数占比，用于 `node_plan_validate` 的第 2 条校验。"""
    if not forecast:
        return 0.0
    bad = sum(1 for d in forecast if is_bad_outdoor(d.get("day_weather")))
    return round(bad / len(forecast), 2)


def describe(weather: dict) -> str:
    """把结构化成一句中文，给 prompt 与前端共用。"""
    if not weather or weather.get("degraded"):
        return "天气数据暂不可用"
    parts = []
    live = weather.get("live") or {}
    if live:
        parts.append(f"当前{live.get('weather')}，{live.get('temperature')}℃，湿度{live.get('humidity')}%")
    forecast = weather.get("forecast") or []
    if forecast:
        first = forecast[0]
        parts.append(f"今日{first.get('day_weather')}，{first.get('night_temp')}~{first.get('day_temp')}℃")
    if len(forecast) > 1:
        parts.append(f"预报覆盖 {len(forecast)} 天（仅含今天起 {weather_config.forecast_days} 天）")
    return "；".join(parts)


def capability_note() -> str:
    """在答案里说明能力边界时用它，避免各处措辞不一。"""
    return f"天气预报只覆盖今天起 {weather_config.forecast_days} 天，更远日期无法给出可靠结论。"
