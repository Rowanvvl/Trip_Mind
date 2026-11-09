"""业务域配置：语料、切分、检索、质量门槛、行程约束。

这里放的是「一改就影响全局行为」的领域常量。节点代码只读不写。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

# 语料的五个类别，与 data/ 下的目录名一致（也是 Milvus 的 content_type 取值）
CONTENT_TYPES: List[str] = ["景点攻略", "线路推荐", "酒店信息", "美食推荐", "交通指南"]

# 目录名 → content_type（目录名与类别不完全同名，这里显式对齐）
CONTENT_TYPE_BY_DIR: Dict[str, str] = {
    "景点攻略": "景点攻略",
    "景点介绍": "景点攻略",
    "线路推荐": "线路推荐",
    "酒店信息": "酒店信息",
    "住宿推荐": "酒店信息",
    "住宿介绍": "酒店信息",
    "美食推荐": "美食推荐",
    "美食介绍": "美食推荐",
    "交通指南": "交通指南",
}

# 语料元数据里的写法 → 内部标准取值。
# 真实语料写的是「景点介绍 / 美食介绍」，而内部类别叫「景点攻略 / 美食推荐」，
# 中间做一层显式映射，别指望子串匹配能兜住（「美食介绍」和「美食推荐」没有包含关系）。
CONTENT_TYPE_ALIAS: Dict[str, str] = {
    "景点介绍": "景点攻略",
    "景点推荐": "景点攻略",
    "景点攻略": "景点攻略",
    "攻略": "景点攻略",
    "门票信息": "景点攻略",
    "线路推荐": "线路推荐",
    "线路介绍": "线路推荐",
    "路线推荐": "线路推荐",
    "行程推荐": "线路推荐",
    "酒店信息": "酒店信息",
    "住宿介绍": "酒店信息",
    "住宿推荐": "酒店信息",
    "住宿信息": "酒店信息",
    "美食介绍": "美食推荐",
    "美食推荐": "美食推荐",
    "餐饮推荐": "美食推荐",
    "交通指南": "交通指南",
    "交通介绍": "交通指南",
    "交通信息": "交通指南",
}

# 别名都匹配不上时的关键词兜底，按顺序命中即返回
_CONTENT_TYPE_KEYWORDS: List[tuple] = [
    (("交通", "机场", "高铁", "地铁", "公交", "打车"), "交通指南"),
    (("线路", "路线", "行程", "天数"), "线路推荐"),
    (("美食", "餐饮", "吃", "小吃", "餐厅"), "美食推荐"),
    (("住宿", "酒店", "民宿", "住"), "酒店信息"),
    (("景点", "景区", "门票", "攻略"), "景点攻略"),
]


def normalize_content_type(value: str) -> str:
    """把各种写法的类别名归一到 `CONTENT_TYPES` 里的标准取值。

    认不出来时返回空串——**不要随便猜一个默认类型**，
    猜错会让切片落到错误的过滤桶里，检索时静默漏召。
    """
    text = (value or "").strip()
    if not text:
        return ""
    if text in CONTENT_TYPES:
        return text
    if text in CONTENT_TYPE_ALIAS:
        return CONTENT_TYPE_ALIAS[text]
    # 去掉括号注释之类的尾巴再试一次：「美食介绍（湘菜）」→「美食介绍」
    head = text.split("（")[0].split("(")[0].strip()
    if head in CONTENT_TYPE_ALIAS:
        return CONTENT_TYPE_ALIAS[head]
    for standard in CONTENT_TYPES:
        if standard in text or text in standard:
            return standard
    for keywords, standard in _CONTENT_TYPE_KEYWORDS:
        if any(k in text for k in keywords):
            return standard
    return ""

# intent → 必须覆盖的 content_type（框架文档 12.6）
INTENT_REQUIRED_TYPES: Dict[str, List[str]] = {
    "景点": ["景点攻略"],
    "线路": ["线路推荐", "交通指南"],
    "住宿": ["酒店信息"],
    "美食": ["美食推荐"],
    "交通": ["交通指南"],
    "综合": list(CONTENT_TYPES),
}

# 冲突调和时的权威序（越靠前越可信），见 12.6
CONTENT_TYPE_AUTHORITY: List[str] = ["交通指南", "线路推荐", "景点攻略", "酒店信息", "美食推荐"]

# intent 判定的规则关键词（规则先行，模型只在规则无法判定时兜底）
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "线路": ["几天", "怎么安排", "行程", "路线", "天数", "几日", "按天", "第几天", "先去哪"],
    "住宿": ["住哪", "住宿", "酒店", "民宿", "住在", "哪个区住"],
    "美食": ["吃", "美食", "餐厅", "小吃", "特色菜", "早点", "夜市"],
    "交通": ["怎么去", "怎么到", "交通", "机场", "高铁", "地铁", "打车", "自驾", "多久到"],
    "景点": ["景点", "好玩", "值得去", "门票", "开放时间", "闭馆", "打卡"],
}


@dataclass(frozen=True)
class TripConfig:
    # ---------- 语料 ----------
    data_dir: str = field(default_factory=lambda: os.getenv("TRIP_DATA_DIR", "./data"))
    default_city: str = field(default_factory=lambda: os.getenv("TRIP_DEFAULT_CITY", "三亚"))

    # ---------- 切分 ----------
    # 语料是判断式长文，按 ##/### 小节切；只有小节超长时才句级滑窗
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "80")))
    min_chunk_len: int = field(default_factory=lambda: int(os.getenv("MIN_CHUNK_LEN", "60")))
    # 小节标题最多保留几级
    max_heading_level: int = 3

    # ---------- 检索 ----------
    retrieve_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVE_TOP_K", "20")))
    rerank_top_k: int = field(default_factory=lambda: int(os.getenv("RERANK_TOP_K", "6")))
    sub_query_top_k: int = field(default_factory=lambda: int(os.getenv("SUB_QUERY_TOP_K", "5")))
    rrf_k: int = field(default_factory=lambda: int(os.getenv("RRF_K", "60")))
    # 过滤为空时自动放宽（城市过滤过严会漏召）
    relax_filter_on_empty: bool = True

    # ---------- 上下文打包 ----------
    # 压缩式打包的目标字符数（不是硬截断上限）
    context_char_budget: int = field(default_factory=lambda: int(os.getenv("CONTEXT_CHAR_BUDGET", "2000")))
    per_chunk_sentences: int = field(default_factory=lambda: int(os.getenv("CONTEXT_SENTENCES", "2")))

    # ---------- 质量门槛 ----------
    guard_enabled: bool = field(default_factory=lambda: os.getenv("GUARD_ENABLED", "true").lower() == "true")
    trace_enabled: bool = field(default_factory=lambda: os.getenv("TRACE_ENABLED", "true").lower() == "true")
    memory_enabled: bool = field(default_factory=lambda: os.getenv("MEMORY_ENABLED", "true").lower() == "true")
    critic_max_retry: int = field(default_factory=lambda: int(os.getenv("CRITIC_MAX_RETRY", "1")))
    goal_max_retry: int = field(default_factory=lambda: int(os.getenv("GOAL_MAX_RETRY", "1")))
    evidence_min_coverage: int = field(default_factory=lambda: int(os.getenv("EVIDENCE_MIN_COVERAGE", "3")))
    fact_max_age_h: int = field(default_factory=lambda: int(os.getenv("FACT_MAX_AGE_H", "6")))
    history_keep_turns: int = field(default_factory=lambda: int(os.getenv("HISTORY_KEEP_TURNS", "3")))
    profile_write_mode: str = field(default_factory=lambda: os.getenv("PROFILE_WRITE_MODE", "explicit"))
    # 输入超长截断
    max_input_chars: int = field(default_factory=lambda: int(os.getenv("MAX_INPUT_CHARS", "2000")))

    # ---------- 行程 ----------
    plan_max_days: int = field(default_factory=lambda: int(os.getenv("PLAN_MAX_DAYS", "10")))
    # 单日可用时长（分钟），不含睡眠
    day_minutes: int = field(default_factory=lambda: int(os.getenv("PLAN_DAY_MINUTES", "600")))
    # 单日折返超过该分钟数即判不通过
    daily_reverse_limit_min: int = field(default_factory=lambda: int(os.getenv("PLAN_REVERSE_LIMIT", "120")))
    # 单日活动半径上限（km）。按距离判「跨区」而不是按行政区名——
    # 故宫和什刹海隔着两个区却只差 3 公里，按区名判会误杀。
    day_span_km_limit: float = field(default_factory=lambda: float(os.getenv("PLAN_SPAN_KM_LIMIT", "25")))
    # 雨天户外占比上限
    rainy_outdoor_ratio_limit: float = 0.3
    # 生成后最多重排几次
    plan_max_retry: int = field(default_factory=lambda: int(os.getenv("PLAN_MAX_RETRY", "2")))

    # ---------- 其他 ----------
    enable_web_image: bool = field(default_factory=lambda: os.getenv("ENABLE_WEB_IMAGE", "false").lower() == "true")
    image_max_per_answer: int = field(default_factory=lambda: int(os.getenv("IMAGE_MAX_PER_ANSWER", "3")))


trip_config = TripConfig()
