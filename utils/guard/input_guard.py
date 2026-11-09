"""输入护栏（第 1 层）。

两道判定：
1. **域外**：签证/移民、医疗诊断、法律诉讼、投资建议、政治敏感、违法操作、写代码等
   —— 一律不接，固定话术指向官方渠道。理由不是「不能答」，而是「答错的代价远高于不答」。
2. **注入**：要求忽略指令、索取系统提示词、伪装开发者等。命中后**清理而不是整句拒绝**：
   把注入片段摘掉，保留真实问题（用户常常只是复制粘贴了别人的话术）。

规则先行（毫秒级、可测），只在规则判不了时交给 `guard_input.prompt` 做模型复核。
返回结构化结论，供 `node_input_guard` 直接决定走向。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from common.config import trip_config

# ---------------- 域外主题 ----------------
_OUT_OF_SCOPE = [
    ("签证移民", ["签证", "绿卡", "移民", "护照办理", "入籍", "留学申请", "工作签"]),
    ("医疗健康", ["诊断", "吃什么药", "用药", "处方", "手术", "病历", "肿瘤", "抑郁症", "感染了"]),
    ("法律事务", ["起诉", "打官司", "律师函", "合同违约", "离婚", "劳动仲裁", "判刑"]),
    ("金融投资", ["买哪只股票", "基金推荐", "理财收益", "期货", "加密货币", "炒股", "外汇交易"]),
    ("违法与危险", ["翻越", "逃票", "带毒品", "偷渡", "走私", "伪造证件", "野泳", "无人区穿越"]),
    ("代码与IT", ["写段代码", "报错", "SQL", "python脚本", "爬虫", "正则表达式"]),
]

# 域内但需要免责的敏感邻域（仍然回答，但固定带声明）
_SENSITIVE_ADJACENT = [
    ("安全提示", ["安全吗", "治安", "危险", "诈骗", "高反", "高原反应", "水土不服"]),
    ("价格时效", ["门票多少", "票价", "多少钱", "开放时间", "几点开门", "闭馆"]),
]

# ---------------- 注入模式 ----------------
_INJECTION_PATTERNS = [
    r"忽略(以上|之前|前面|上面)(所有)?(的)?(指令|要求|提示)",
    r"(无视|忘记|跳过)(你的|所有)?(系统)?(指令|设定|规则)",
    r"(打印|输出|重复|告诉我)(你的)?(系统)?(提示词|system\s*prompt|初始指令)",
    r"你现在(是|扮演)(?!.*旅游)",
    r"(developer|debug|admin)\s*mode",
    r"ignore\s+(all\s+)?(previous|above)\s+instructions?",
    r"reveal\s+your\s+(system\s*)?prompt",
]

_REPLY_OUT_OF_SCOPE = (
    "这个问题超出我的服务范围，我只做旅游相关的内容（景点、线路、住宿、美食、交通、预算与行程安排）。"
    "{topic}类问题建议以官方渠道为准，我不做判断，避免给你错误信息。"
)

_REPLY_INJECTION = "我只回答旅游相关问题，继续按你的真实需求来问就好。"


@dataclass
class GuardResult:
    allowed: bool = True
    action: str = "pass"            # pass | sanitize | reject
    category: str = ""              # 域外类别 / injection / too_long / empty
    reason: str = ""
    sanitized_query: str = ""       # 清理后的提问
    reply: str = ""                 # 拒绝时的固定话术
    need_llm_check: bool = False    # 规则判不了，交给模型复核
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "action": self.action,
            "category": self.category,
            "reason": self.reason,
            "sanitized_query": self.sanitized_query,
            "reply": self.reply,
            "need_llm_check": self.need_llm_check,
            "flags": self.flags,
        }


def _match_any(text: str, table) -> tuple[str, str]:
    for category, keywords in table:
        for kw in keywords:
            if kw.lower() in text:
                return category, kw
    return "", ""


def detect_injection(text: str) -> list[str]:
    hits = []
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(pattern)
    return hits


def strip_injection(text: str) -> str:
    """把注入句子整句摘掉，保留其余内容。"""
    cleaned = text
    for pattern in _INJECTION_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    # 顺手清掉残留的「以上」「如下」式引导词与多余标点
    cleaned = re.sub(r"[，,、；;]{2,}", "，", cleaned).strip(" ，,、；;")
    return cleaned.strip()


def check_input(query: str, max_chars: int | None = None) -> GuardResult:
    """输入侧总入口。纯规则，无网络依赖，可单测。"""
    limit = max_chars or trip_config.max_input_chars
    text = (query or "").strip()

    if not text:
        return GuardResult(
            allowed=False,
            action="reject",
            category="empty",
            reason="空提问",
            reply="想了解哪座城市？告诉我目的地，我给你具体的安排建议。",
        )

    flags: list[str] = []

    if len(text) > limit:
        text = text[:limit]
        flags.append("truncated")

    # 1. 注入
    injection_hits = detect_injection(text)
    if injection_hits:
        sanitized = strip_injection(text)
        flags.append("injection")
        if len(sanitized) < 4:
            return GuardResult(
                allowed=False,
                action="reject",
                category="injection",
                reason="注入且无有效问题",
                sanitized_query=sanitized,
                reply=_REPLY_INJECTION,
                flags=flags,
            )
        text = sanitized

    # 2. 域外
    category, keyword = _match_any(text, _OUT_OF_SCOPE)
    if category:
        return GuardResult(
            allowed=False,
            action="reject",
            category=category,
            reason=f"命中域外关键词：{keyword}",
            sanitized_query=text,
            reply=_REPLY_OUT_OF_SCOPE.format(topic=category),
            flags=flags + ["out_of_scope"],
        )

    # 3. 敏感邻域：放行但打标记，输出层要加声明
    adjacent_category, _ = _match_any(text, _SENSITIVE_ADJACENT)
    if adjacent_category:
        flags.append(f"disclaimer:{adjacent_category}")

    # 4. 太短且无地理实体 → 交给模型判断是不是敷衍提问（例如「在吗」）
    if len(text) <= 4 and not flags:
        return GuardResult(
            allowed=True,
            action="pass",
            sanitized_query=text,
            need_llm_check=True,
            reason="过短，需复核意图",
            flags=flags + ["short"],
        )

    return GuardResult(
        allowed=True,
        action="sanitize" if "injection" in flags else "pass",
        sanitized_query=text,
        reason="；".join(f"命中 {f}" for f in flags),
        flags=flags,
    )


def build_guard_prompt_payload(result: GuardResult) -> dict:
    """给模型复核用的最小载荷（不传原始系统提示）。"""
    return {"query": result.sanitized_query, "flags": result.flags}
