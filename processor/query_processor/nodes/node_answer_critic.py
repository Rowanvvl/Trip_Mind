"""答案校验（producer-critic 的最小可用形态）。

**只做能被验证的硬检查**，不评价文风。三类检查：
1. 规则（零成本，每次必跑）：
   - `citation_missing`：有引用资料但答案里一个 `[n]` 都没有；
   - `citation_out_of_range`：标注的编号超出 citations 范围；
   - `wrong_city`：答案里出现了目标城市以外的城市名，且该城市不在引用资料里（串城）；
2. 模型（可选）：把答案、引用、事实一起给它，按 `answer_critic.prompt` 报问题；
3. 不通过时**最多重写一次**，仍不通过就保留答案但在末尾加一句"部分信息未核实"。

只在失败时付费：规则通过时才调模型，模型也不便宜。
"""
from __future__ import annotations

import re
from typing import List

from common.config import lm_config, trip_config
from common.logging.logger import logger, node_log
from processor.query_processor.state import QueryGraphState
from utils.geo.city_gazetteer import ALL_CITIES, find_cities_in_text
from utils.load_prompt import render_safe
from utils.lm.lm_utils import chat_json, record_usage
from utils.trace_utils import record_critique

_CITATION_RE = re.compile(r"\[(\d+)\]")
_BLOCKING = {"unsupported_number", "wrong_city", "out_of_scope_claim"}


@node_log("node_answer_critic")
def node_answer_critic(state: QueryGraphState) -> dict:
    """校验并决定是否重写。

    输出：`critique`、`critic_retry`
    """
    answer = state.get("answer") or ""
    citations = state.get("citations") or []
    trace_id = state.get("trace_id", "")

    issues = _rule_checks(answer, citations, state)

    if not issues and lm_config.enabled and citations:
        issues = _model_checks(answer, citations, state)

    passed = not any(issue.get("type") in _BLOCKING for issue in issues)

    retried = bool(state.get("critic_retry"))
    if not passed and (state.get("critic_retry") or 0) < trip_config.critic_max_retry:
        logger.info(f"[answer_critic] 未通过，准备重写：{issues}")

    record_critique(trace_id, passed, issues, retried)

    payload: dict = {"critique": {"passed": passed, "issues": issues}}
    if not passed and (state.get("critic_retry") or 0) < trip_config.critic_max_retry:
        payload["critic_retry"] = (state.get("critic_retry") or 0) + 1
    elif not passed:
        # 重试用尽：不硬发，也不假装没问题，在答案末尾标注
        payload["answer"] = answer + "\n\n（以上部分信息未能在知识库或官方渠道核实，请以官方公告为准。）"
    return payload


def _rule_checks(answer: str, citations: List[dict], state: QueryGraphState) -> List[dict]:
    issues: List[dict] = []

    markers = [int(m) for m in _CITATION_RE.findall(answer)]
    if citations and not markers:
        issues.append(
            {
                "type": "citation_missing",
                "detail": "答案没有任何 [n] 标注",
                "fix": "为每处结论补上资料编号",
            }
        )
    if markers and citations:
        out_of_range = [m for m in markers if m < 1 or m > len(citations)]
        if out_of_range:
            issues.append(
                {
                    "type": "citation_out_of_range",
                    "detail": f"引用了不存在的编号 {sorted(set(out_of_range))}，共 {len(citations)} 条资料",
                    "fix": "只引用给定范围内的编号",
                }
            )

    # 串城检查：答案里提到的城市，必须是目标城市或引用资料里出现过的
    target = state.get("city") or ""
    allowed = {target} if target else set()
    for item in citations:
        if item.get("city"):
            allowed.add(item["city"])
    for city in find_cities_in_text(answer):
        if city in allowed:
            continue
        if city in ALL_CITIES:
            issues.append(
                {
                    "type": "wrong_city",
                    "detail": f"答案提到「{city}」，但目标城市是「{target or '未指定'}」",
                    "fix": f"删除与「{city}」相关的内容，或说明为何要跨城对比",
                }
            )
            break
    return issues


def _model_checks(answer: str, citations: List[dict], state: QueryGraphState) -> List[dict]:
    cite_text = "\n".join(
        f"[{c['index']}] {c.get('city', '')}·{c.get('doc_title', '')} · {c.get('section', '')}" for c in citations
    )
    prompt = render_safe(
        "answer_critic",
        "",
        city=state.get("city") or "未指定",
        slots=str(state.get("slots") or {}),
        query=state.get("original_query") or "",
        answer=answer,
        citations=cite_text,
        facts=(state.get("facts") or {}).get("text") or "（无）",
    )
    data, result = chat_json(prompt, tier="light", fallback={"passed": True, "issues": []})
    record_usage(state.get("trace_id", ""), result)
    if not result.ok:
        return []
    issues = data.get("issues") or []
    return [issue for issue in issues if isinstance(issue, dict)][:5]


def after_critic(state: QueryGraphState) -> str:
    """不通过且还有重试额度 → 重写；否则进入目标验收。"""
    critique = state.get("critique") or {}
    if critique.get("passed", True):
        return "goal"
    if (state.get("critic_retry") or 0) >= trip_config.critic_max_retry:
        return "goal"
    return "rewrite"
