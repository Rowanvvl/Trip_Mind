"""输出护栏（第 3 层）。

生成之后、返回之前的最后一道处理，做三件确定性的事（不依赖模型判断）：

1. **必带声明**：涉及票价、开放时间、预约规则 → 附「以官方公告为准」；
   涉及安全/高原/涉水 → 附安全提示。
2. **软化绝对化表述**：「一定」「保证」「肯定能」→ 改成判断式语气。
   旅游场景里最危险的不是说错，而是说得像官方规定。
3. **PII 脱敏**：手机号、身份证号一律打码（用户可能把自己的证件号贴进来）。

不做的事：不改写事实、不删结论。事实校验属于 `node_answer_critic`，
这里只做「措辞与声明」，因此可以放在流式结束后同步执行，零额外延迟。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

PRICE_DISCLAIMER = "门票、开放时间与预约规则会调整，出行前请以景区官方公告为准。"
SAFETY_DISCLAIMER = "安全与健康状况请以官方提示和自身情况为准，必要时咨询专业人士。"
OFFICIAL_DISCLAIMER = "签证、医疗、法律等事务请以主管部门或专业机构口径为准。"

_ABSOLUTE_WORDS = {
    "一定能": "通常可以",
    "肯定会": "大概率会",
    "保证能": "一般来说可以",
    "绝对": "基本",
    "百分百": "大多数情况下",
    "必然": "往往会",
    "肯定不会": "一般不会",
    "绝对不会": "基本不会",
}

_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
_ID_RE = re.compile(r"(?<!\d)(\d{17}[\dXx]|\d{15})(?!\d)")

_PRICE_HINTS = ("门票", "票价", "价格", "开放时间", "闭馆", "预约", "营业时间", "元/人")
_SAFETY_HINTS = ("高原", "海拔", "涉水", "游泳", "徒步", "自驾", "夜爬", "溯溪", "安全")


@dataclass
class OutputGuardResult:
    text: str
    disclaimers: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    masked: int = 0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "disclaimers": self.disclaimers,
            "issues": self.issues,
            "masked": self.masked,
        }


def mask_pii(text: str) -> tuple[str, int]:
    count = 0

    def _mask_phone(m: re.Match) -> str:
        nonlocal count
        count += 1
        s = m.group(1)
        return f"{s[:3]}****{s[-4:]}"

    def _mask_id(m: re.Match) -> str:
        nonlocal count
        count += 1
        s = m.group(1)
        return f"{s[:4]}**********{s[-4:]}"

    text = _PHONE_RE.sub(_mask_phone, text)
    text = _ID_RE.sub(_mask_id, text)
    return text, count


def soften_absolute(text: str) -> tuple[str, list[str]]:
    issues = []
    for word, replacement in _ABSOLUTE_WORDS.items():
        if word in text:
            text = text.replace(word, replacement)
            issues.append(f"软化绝对表述：{word} → {replacement}")
    return text, issues


def pick_disclaimers(text: str, flags: list[str] | None = None) -> list[str]:
    """按内容与上游 flags 选声明，最多两条，避免答案尾部堆一堆免责。"""
    picked: list[str] = []
    flags = flags or []

    if any(h in text for h in _PRICE_HINTS) or any(f.startswith("disclaimer:价格") for f in flags):
        picked.append(PRICE_DISCLAIMER)
    if any(h in text for h in _SAFETY_HINTS) or any(f.startswith("disclaimer:安全") for f in flags):
        picked.append(SAFETY_DISCLAIMER)
    if any(f.startswith("disclaimer:签证") or f.startswith("disclaimer:医疗") for f in flags):
        picked.append(OFFICIAL_DISCLAIMER)

    # 去重并限制条数
    return list(dict.fromkeys(picked))[:2]


def apply(text: str, flags: list[str] | None = None, add_disclaimer: bool = True) -> OutputGuardResult:
    """输出侧总入口：脱敏 → 软化 → 补声明。"""
    result = OutputGuardResult(text=text or "")

    result.text, result.masked = mask_pii(result.text)
    if result.masked:
        result.issues.append(f"脱敏 {result.masked} 处个人信息")

    result.text, soften_issues = soften_absolute(result.text)
    result.issues.extend(soften_issues)

    if add_disclaimer:
        result.disclaimers = pick_disclaimers(result.text, flags)
        if result.disclaimers:
            suffix = "\n\n" + " ".join(result.disclaimers)
            if result.disclaimers[0] not in result.text:
                result.text = result.text.rstrip() + suffix

    return result


def refusal_reply(reason: str = "") -> str:
    """统一拒答话术，避免各处措辞不一致。"""
    base = "这个问题我不做判断，避免给你错误信息。"
    return f"{base}{reason}" if reason else base
