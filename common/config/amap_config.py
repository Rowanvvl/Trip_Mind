"""高德 Web 服务配置。

注意：这里用的是「Web 服务 API」的 key，不是 JS API 的 key；key 只在服务端使用，
不下发前端（框架文档 12.3 最小权限）。导入侧与查询侧建议用不同 key，走环境变量区分。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AmapConfig:
    key: str = field(default_factory=lambda: os.getenv("AMAP_KEY", ""))
    # 批量导入时（给切片补坐标）用另一个 key，避免与在线查询抢配额
    import_key: str = field(default_factory=lambda: os.getenv("AMAP_IMPORT_KEY", "") or os.getenv("AMAP_KEY", ""))
    host: str = field(default_factory=lambda: os.getenv("AMAP_HOST", "https://restapi.amap.com"))

    qps: int = field(default_factory=lambda: int(os.getenv("AMAP_QPS", "3")))
    timeout: float = field(default_factory=lambda: float(os.getenv("AMAP_TIMEOUT", "1.2")))
    daily_quota: int = field(default_factory=lambda: int(os.getenv("AMAP_DAILY_QUOTA", "5000")))

    # 路径规划默认策略
    driving_strategy: int = field(default_factory=lambda: int(os.getenv("AMAP_DRIVING_STRATEGY", "10")))
    # 出行方式判定的距离阈值（km）
    walk_max_km: float = float(os.getenv("AMAP_WALK_MAX_KM", "1.5"))
    transit_min_km: float = float(os.getenv("AMAP_TRANSIT_MIN_KM", "30"))
    # 景区内部无路网时的直线距离放大系数
    straight_line_factor: float = float(os.getenv("AMAP_STRAIGHT_FACTOR", "1.3"))

    @property
    def enabled(self) -> bool:
        return bool(self.key)

    def active_key(self, for_import: bool = False) -> str:
        return self.import_key if for_import else self.key


amap_config = AmapConfig()
