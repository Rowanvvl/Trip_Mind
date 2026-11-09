"""天气配置。

高德 weatherInfo 只有实时 + 今天起 3 天预报，**没有 7 日与历史**。
需要更长周期时把 provider 换成和风/彩云，接口形状保持不变。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class WeatherConfig:
    provider: str = field(default_factory=lambda: os.getenv("WEATHER_PROVIDER", "amap"))
    qweather_key: str = field(default_factory=lambda: os.getenv("QWEATHER_KEY", ""))
    timeout: float = field(default_factory=lambda: float(os.getenv("WEATHER_TIMEOUT", "1.2")))

    # 缓存：实时 30min、预报 3h（框架文档 11.4）
    ttl_live: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_WEATHER", "1800")))
    ttl_forecast: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_WEATHER_FORECAST", "10800")))
    # 预报可覆盖的天数上限，超出要明确拒绝而不是硬编
    forecast_days: int = 3
    # 超过该时长的事实不用于「明天/这周末」类判断
    max_age_hours: int = field(default_factory=lambda: int(os.getenv("FACT_MAX_AGE_H", "6")))

    # 雨天替换为室内项目时的判定关键词
    rain_keywords: tuple[str, ...] = ("雨", "雷", "阵雨", "暴雨", "雪")
    # 户外活动不利条件
    bad_outdoor_keywords: tuple[str, ...] = ("雨", "雷", "暴雨", "台风", "大风", "高温")


weather_config = WeatherConfig()
