"""会话级槽位（State 层记忆）。

三分记忆的中间层（框架文档 12.5）：
- Session：Mongo 里的原始消息；
- **State：本文件管理的 slots，每轮增量更新，只增改不覆盖**；
- Memory：跨会话的 `user_profiles`，见 `profile_utils`。

为什么必须有 slots：用户第一轮说的「带俩老人、预算一万、不吃辣」如果只躺在历史里，
第二轮追问「那住哪」时多半已经被稀释掉了。槽位是唯一的可信来源，
进 prompt 的是 slots，原始历史只保留最近 3 轮。

合并规则（重要）：
- 新值只有**非空**才覆盖旧值，空值一律忽略；
- 列表类槽位（must_go / exclude）取并集，不做替换；
- 每次合并记录 `slot_sources`，说明这个值来自哪一轮，便于用户质疑时回溯。
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from utils.format_utils import now_iso

# 槽位定义：键 → 中文说明（也用于拼 prompt）
SLOT_SPEC: Dict[str, str] = {
    "city": "目的地城市",
    "days": "游玩天数",
    "travelers": "出行人数",
    "companions": "同行人群（老人/小孩/情侣/朋友）",
    "budget": "总预算（元）",
    "budget_level": "消费档次（经济/舒适/品质）",
    "pace": "节奏（轻松/适中/紧凑）",
    "transport": "市内交通偏好（地铁/打车/自驾/公交/步行）",
    "diet_avoid": "忌口与饮食禁忌",
    "must_go": "必去清单",
    "exclude": "明确不去的地方",
    "arrive_time": "抵达时间",
    "depart_time": "返程时间",
    "lodging_area": "住宿区域偏好",
}

# 列表型槽位，合并时取并集
LIST_SLOTS = {"must_go", "exclude", "companions"}

_DAYS_RE = re.compile(r"(\d+)\s*(天|日)")
_TRAVELERS_RE = re.compile(r"(\d+)\s*(个人|人|位|名)")
_FAMILY_RE = re.compile(r"(\d)?\s*家\s*(\d)\s*口")
_BUDGET_RE = re.compile(r"预算\s*([0-9.]+)\s*(万|w|k|千|元)?", re.IGNORECASE)

_CN_HINT_RE = re.compile(r"(去|到|在|想去)\s*([\u4e00-\u9fa5]{2,6})")

# ---------------- 中文数字 ----------------
# 用户几乎不会说「玩 5 天」，说的都是「玩五天」。不归一化就等于槽位全空。
_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_CN_RUN = re.compile(r"[零一二两三四五六七八九十百千万]+")


def _cn_run_to_int(run: str) -> int | None:
    """把一段中文数字转成整数，支持「一万五」「三千八」这类口语。"""
    if not run:
        return None

    total = 0
    section = 0
    number = 0
    last_unit = 0
    for char in run:
        if char in _CN_DIGITS:
            number = _CN_DIGITS[char]
        elif char in _CN_UNITS:
            unit = _CN_UNITS[char]
            section += (number or 1) * unit
            number = 0
            last_unit = unit
        elif char == "万":
            total += (section + number) * 10000
            section = 0
            number = 0
            last_unit = 10000
        else:
            return None

    # 「一万五」= 15000，「三千八」= 3800：末尾单字表示下一级单位
    if number and last_unit >= 100 and len(run) >= 2:
        number = number * (last_unit // 10)

    return total + section + number


def normalize_cn_numbers(text: str) -> str:
    """把中文数字替换成阿拉伯数字，方便后续正则统一处理。

    **关键约束：单个数字字后面必须是量词才替换。**
    否则「三亚」会被替换成「3亚」，城市直接丢掉——这是这类归一化最典型的坑。
    含十/百/千/万的组合（一万、十二）语义明确，一律替换。
    """
    source = text or ""
    unit_after = set("天日人位名个晚口家元块张间次点号月周年的岁天")

    def _replace(match: re.Match) -> str:
        run = match.group(0)
        if len(run) == 1 and run in _CN_DIGITS:
            following = source[match.end() : match.end() + 1]
            if following not in unit_after:
                return run  # 三亚、三峡、二环 —— 不是数量
        value = _cn_run_to_int(run)
        return str(value) if value is not None else run

    return _CN_RUN.sub(_replace, source)


def empty_slots() -> Dict[str, Any]:
    return {
        "city": None,
        "days": None,
        "travelers": None,
        "companions": [],
        "budget": None,
        "budget_level": None,
        "pace": None,
        "transport": None,
        "diet_avoid": [],
        "must_go": [],
        "exclude": [],
        "arrive_time": None,
        "depart_time": None,
        "lodging_area": None,
        "updated_at": None,
        "slot_sources": {},
    }


def _to_amount(number: str, unit: str | None) -> float | None:
    try:
        value = float(number)
    except (TypeError, ValueError):
        return None
    if unit in ("万", "w", "W"):
        return value * 10000
    if unit in ("k", "K", "千"):
        return value * 1000
    return value


def extract_slots_from_query(query: str) -> Dict[str, Any]:
    """规则抽取。抽不到就留空，**不要猜**——猜错比不填更糟。

    只处理高置信度的显式表达（「5 天」「两个人」「预算一万」「不吃辣」）。
    隐含表达（「玩得轻松点」→ pace）交给 `node_query_understand` 的模型分支。
    """
    from utils.geo.city_gazetteer import find_city_in_text

    text = normalize_cn_numbers(query or "")
    slots: Dict[str, Any] = {}

    city = find_city_in_text(text)
    if city:
        slots["city"] = city

    m = _DAYS_RE.search(text)
    if m:
        slots["days"] = int(m.group(1))

    m = _TRAVELERS_RE.search(text)
    if m:
        slots["travelers"] = int(m.group(1))
    else:
        family = _FAMILY_RE.search(text)
        if family:
            slots["travelers"] = int(family.group(2))

    m = _BUDGET_RE.search(text)
    if m:
        amount = _to_amount(m.group(1), m.group(2))
        # 少于 200 的多半是「预算 5k」被截断或用户随手写的，不做处理
        if amount and amount >= 200:
            slots["budget"] = amount

    companions: List[str] = []
    if re.search(r"老人|父母|爸妈|长辈|爷爷|奶奶|外公|外婆", text):
        companions.append("老人")
    if re.search(r"孩子|小孩|宝宝|儿童|亲子|娃", text):
        companions.append("小孩")
    if re.search(r"情侣|女朋友|男朋友|蜜月|二人世界", text):
        companions.append("情侣")
    if companions:
        slots["companions"] = companions

    if re.search(r"轻松|慢一点|不赶|休闲|养老式", text):
        slots["pace"] = "轻松"
    elif re.search(r"紧凑|赶一点|尽量多|特种兵|一天跑", text):
        slots["pace"] = "紧凑"

    if re.search(r"自驾|开车|租车", text):
        slots["transport"] = "自驾"
    elif re.search(r"地铁|公交|公共交通", text):
        slots["transport"] = "公共交通"
    elif re.search(r"打车|滴滴|出租车", text):
        slots["transport"] = "打车"
    elif re.search(r"步行|走路|citywalk", text, re.IGNORECASE):
        slots["transport"] = "步行"

    diet: List[str] = []
    if re.search(r"不吃辣|不能吃辣|怕辣|微辣", text):
        diet.append("不吃辣")
    if re.search(r"素|素食|吃斋", text):
        diet.append("素食")
    if re.search(r"清真|回族", text):
        diet.append("清真")
    if re.search(r"过敏", text):
        diet.append("有过敏源")
    if diet:
        slots["diet_avoid"] = diet

    must_go = _extract_place_list(text, _MUST_GO_RE)
    if must_go:
        slots["must_go"] = must_go
    exclude = _extract_place_list(text, _EXCLUDE_RE)
    if exclude:
        slots["exclude"] = exclude

    return slots


# 「必去故宫和颐和园」这类表达是行程需求的常见句式，必须能被规则抽出来：
# 抽不出来，`_is_must` 就全是 False，超时砍点时会把用户点名要去的地方顺手砍掉。
_MUST_GO_RE = re.compile(r"(?:必去|一定要去|必须去|必打卡|想去|重点去|主要去|一定要看|别漏了)\s*([^，。；;！!？?\n]{1,40})")
_EXCLUDE_RE = re.compile(r"(?:不去|不想去|排除|跳过|别安排)\s*([^，。；;！!？?\n]{1,30})")

# 切分「故宫、天坛、长城」这类并列清单。
# 注意**不能把「和」当分隔符**：「颐和园」「太和殿」「中和殿」都会被切成两半，
# 所以「和」不参与切分，靠词典做最长匹配来消歧（见 _extract_place_list）。
_PLACE_SPLIT = re.compile(r"[、,，/]|还有|以及|及|与")
# 不像地名的词就别往里塞，否则 must_go 会被「不要太赶」这类短语污染。
# 只放明显是功能词/动作词的片段——「天」「太」这种在景点名里常见的字不能进，否则会误杀天坛、太庙。
_NOT_PLACE = re.compile(r"怎么|安排|住宿|住哪|吃|玩|看|的|了|吗|呢|吧|预算|小时|轻松|赶|时间|行程|门票多少钱")


def _extract_place_list(text: str, pattern: re.Pattern) -> List[str]:
    """从触发词后面截取地名清单，能对上词典的用词典标准名，对不上的原样保留。

    对不上词典并不丢弃：用户的叫法可能和词典别名不一致，
    原样交给下游，`node_retrieve_by_day` 还会再 `resolve_attraction` 一次。
    """
    from utils.geo.city_gazetteer import find_attractions_in_text

    found: List[str] = []
    for match in pattern.finditer(text):
        chunk = match.group(1)
        # 先整段丢给词典做最长匹配：这一步能正确认出「颐和园」，
        # 而先按「和」硬切就会变成「颐」+「园」。
        hits = find_attractions_in_text(chunk)
        if hits:
            for attraction in hits:
                if attraction.name not in found:
                    found.append(attraction.name)
            continue
        for piece in _PLACE_SPLIT.split(chunk):
            name = piece.strip(" 　的地")
            if not (2 <= len(name) <= 10):
                continue
            if _NOT_PLACE.search(name):
                continue
            if name not in found:
                found.append(name)
    return found


def _union(old: Iterable, new: Iterable) -> list:
    merged = list(old or [])
    for item in new or []:
        if item and item not in merged:
            merged.append(item)
    return merged


def merge_slots(old: Dict[str, Any] | None, new: Dict[str, Any] | None, source: str = "current_turn") -> Dict[str, Any]:
    """增量合并。空值不覆盖，列表取并集。"""
    base = dict(old or empty_slots())
    base.setdefault("slot_sources", {})
    incoming = new or {}

    for key, value in incoming.items():
        if key in ("updated_at", "slot_sources"):
            continue
        if value in (None, "", [], {}):
            continue

        if key in LIST_SLOTS:
            base[key] = _union(base.get(key), value)
        else:
            # 只有真正改变了才记录来源，避免同一句被反复标注
            if base.get(key) != value:
                base["slot_sources"][key] = source
            base[key] = value

    base["updated_at"] = now_iso()
    return base


def slots_to_text(slots: Dict[str, Any] | None, only_filled: bool = True) -> str:
    """拼成可进 prompt 的紧凑文本。空槽位默认不出现，避免模型把它们当条件。"""
    if not slots:
        return "（暂无已知条件）"
    lines = []
    for key, label in SLOT_SPEC.items():
        value = slots.get(key)
        if value in (None, "", [], {}):
            if only_filled:
                continue
            value = "未提供"
        if isinstance(value, list):
            value = "、".join(str(v) for v in value)
        if key == "budget" and isinstance(value, (int, float)):
            value = f"{int(value)} 元"
        lines.append(f"- {label}：{value}")
    return "\n".join(lines) if lines else "（暂无已知条件）"


def missing_slots(slots: Dict[str, Any] | None, required: Iterable[str] = ("city",)) -> list[str]:
    slots = slots or {}
    return [k for k in required if slots.get(k) in (None, "", [], {})]


def inherit_from_history(slots: Dict[str, Any], history: List[dict]) -> Dict[str, Any]:
    """从最近一轮的用户消息里补 city（「那住哪」这类指代消解的最低成本方案）。"""
    if slots.get("city"):
        return slots
    for msg in reversed(history or []):
        if msg.get("role") != "user":
            continue
        from utils.geo.city_gazetteer import find_city_in_text

        city = find_city_in_text(msg.get("text", ""))
        if city:
            return merge_slots(slots, {"city": city}, source="history_inherit")
    return slots
