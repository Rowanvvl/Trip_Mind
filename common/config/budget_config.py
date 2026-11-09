"""预算口径与单价表。

**预算一律由规则测算，不让模型算数**（模型算钱几乎必错，见框架文档 11.5）。
这里集中维护口径：所有数字都是「区间」而非单值，输出时必须带 assumptions。

维护提示：单价随季节波动，建议每季度核对一次；改动只动本文件，节点代码不用碰。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

from dotenv import load_dotenv

load_dotenv()

PriceRange = Tuple[float, float]


def _city_tier(city: str) -> str:
    """城市消费档位。第一版用固定表，后续可换成配置中心。"""
    if city in {"北京", "上海"}:
        return "tier1"
    if city in {"三亚"}:
        return "resort"
    if city in {"杭州", "广州", "南京", "成都", "重庆", "西安", "武汉", "长沙", "厦门"}:
        return "tier2"
    # 张家界、云南（昆明/大理/丽江）这类以景区消费为主
    return "scenic"


@dataclass(frozen=True)
class BudgetConfig:
    """住宿 / 餐饮 / 门票 / 市内交通的单价口径。"""

    # 住宿：城市档位 × 档次 → 每晚每间
    hotel_price: Dict[str, Dict[str, PriceRange]] = field(
        default_factory=lambda: {
            "tier1": {"经济": (280, 420), "舒适": (550, 850), "品质": (1200, 2200)},
            "tier2": {"经济": (200, 320), "舒适": (400, 650), "品质": (900, 1600)},
            "resort": {"经济": (350, 550), "舒适": (700, 1200), "品质": (1800, 3500)},
            "scenic": {"经济": (150, 260), "舒适": (320, 550), "品质": (800, 1400)},
        }
    )
    # 房型系数：一间房住几人
    room_factor: Dict[str, float] = field(
        default_factory=lambda: {"单人": 1.0, "双人": 1.0, "家庭三人": 1.5, "家庭四人": 2.0}
    )
    # 餐饮：城市档位 × 档次 → 每人每餐
    meal_price: Dict[str, Dict[str, PriceRange]] = field(
        default_factory=lambda: {
            "tier1": {"经济": (25, 45), "舒适": (60, 110), "品质": (150, 300)},
            "tier2": {"经济": (20, 35), "舒适": (45, 90), "品质": (120, 240)},
            "resort": {"经济": (30, 55), "舒适": (70, 130), "品质": (180, 380)},
            "scenic": {"经济": (18, 30), "舒适": (40, 75), "品质": (100, 200)},
        }
    )
    meals_per_day: int = 3
    # 门票：城市档位 → 每人每天景点门票均值（切片能取到实际票价时优先用实际值）
    ticket_per_day: Dict[str, PriceRange] = field(
        default_factory=lambda: {
            "tier1": (60, 160),
            "tier2": (50, 140),
            "resort": (80, 220),
            "scenic": (120, 260),
        }
    )
    # 市内交通：城市档位 → 每人每天
    local_transport_per_day: Dict[str, PriceRange] = field(
        default_factory=lambda: {
            "tier1": (30, 60),
            "tier2": (20, 45),
            "resort": (40, 100),
            "scenic": (30, 80),
        }
    )
    # 打车单价（元/公里），用于按里程估算
    taxi_per_km: PriceRange = (2.3, 3.2)
    # 城际：高铁二等座约 0.45 元/公里，机票按距离分档
    rail_per_km: float = 0.45
    flight_short_km: float = 800.0
    flight_price_short: PriceRange = (500, 900)
    flight_price_long: PriceRange = (900, 1800)

    # 区间外扩：把点估计撑成区间，避免给出假精确的数字
    spread_ratio: float = float(os.getenv("BUDGET_SPREAD", "0.15"))
    # 默认档次
    default_level: str = field(default_factory=lambda: os.getenv("BUDGET_DEFAULT_LEVEL", "舒适"))
    # 是否默认包含城际大交通（默认不含，答案里必须写明）
    include_intercity_default: bool = False

    def tier_of(self, city: str) -> str:
        return _city_tier(city)


budget_config = BudgetConfig()
