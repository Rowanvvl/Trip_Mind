"""大模型调用。

只依赖 OpenAI 兼容接口，因此百炼 / DeepSeek / vLLM / Ollama 都能直接用，
换模型只改 `.env`，不改代码。

三个约定：
1. **未配置 key 时不抛异常**，返回 `disabled=True` 的空结果，让上游走降级分支
   （骨架阶段没有 key 也能把图跑通，这一点很重要）；
2. **结构化输出必须走 `chat_json`**，它做代码块剥离、括号配平、尾逗号修复，
   比在每个节点里写 `json.loads(resp)` 可靠得多；
3. 温度按节点类型区分：判定类接近 0，生成类 0.3（`lm_config` 里统一定义）。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from common.config import lm_config
from common.logging.logger import logger

_client = None


@dataclass
class LLMResult:
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    disabled: bool = False
    error: str = ""
    model: str = ""
    cost_ms: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.disabled and not self.error

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "disabled": self.disabled,
            "error": self.error,
            "model": self.model,
            "cost_ms": self.cost_ms,
        }


def get_client():
    """懒加载 OpenAI 客户端。没装 SDK 或没配 key 时返回 None。"""
    global _client
    if _client is not None:
        return _client
    if not lm_config.enabled:
        logger.warning("[llm] 未配置 OPENAI_API_KEY，模型调用将降级")
        return None
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("[llm] 未安装 openai SDK，模型调用将降级")
        return None
    _client = OpenAI(
        api_key=lm_config.api_key,
        base_url=lm_config.base_url,
        timeout=lm_config.request_timeout,
        max_retries=lm_config.max_retries,
    )
    return _client


def reset_client() -> None:
    """测试或热更新 key 时用。"""
    global _client
    _client = None


def chat(
    messages: List[Dict[str, str]],
    tier: str = "main",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
    response_format: Optional[Dict[str, str]] = None,
) -> LLMResult:
    """同步对话。失败返回 `error`，不抛异常。"""
    client = get_client()
    model_name = model or lm_config.get_model(tier)
    if client is None:
        return LLMResult(disabled=True, model=model_name)

    temp = lm_config.temperature if temperature is None else temperature
    started = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temp,
            max_tokens=max_tokens,
            response_format=response_format,
        )
    except Exception as exc:
        logger.exception(f"[llm] 调用失败 model={model_name}")
        return LLMResult(error=f"{type(exc).__name__}: {exc}", model=model_name)

    cost_ms = int((time.perf_counter() - started) * 1000)
    content = (resp.choices[0].message.content or "").strip() if resp.choices else ""
    usage = getattr(resp, "usage", None)
    return LLMResult(
        text=content,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        model=model_name,
        cost_ms=cost_ms,
    )


def chat_stream(
    messages: List[Dict[str, str]],
    tier: str = "main",
    temperature: Optional[float] = None,
    model: Optional[str] = None,
) -> Iterator[str]:
    """流式对话，逐段 yield 文本。

    调用方负责把 delta 推给 SSE；本函数**不碰**任何前端逻辑。
    """
    client = get_client()
    model_name = model or lm_config.get_model(tier)
    if client is None:
        return
    temp = lm_config.answer_temperature if temperature is None else temperature
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temp,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                yield text
    except Exception:
        logger.exception(f"[llm] 流式调用失败 model={model_name}")
        return


# ---------------------------------------------------------------- 结构化输出
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Optional[dict | list]:
    """从模型输出里抠出 JSON。处理三种常见脏数据：代码块、前后废话、尾逗号。"""
    if not text:
        return None

    candidates: List[str] = []
    fenced = _FENCE_RE.findall(text)
    candidates.extend(fenced)
    candidates.append(text)

    for candidate in candidates:
        snippet = _balanced_json(candidate)
        if not snippet:
            continue
        for variant in (snippet, re.sub(r",\s*([}\]])", r"\1", snippet)):
            try:
                return json.loads(variant)
            except json.JSONDecodeError:
                continue
    return None


def _balanced_json(text: str) -> Optional[str]:
    """按括号配平截取第一个完整的 JSON 对象/数组。"""
    start = None
    for index, char in enumerate(text):
        if char in "{[":
            start = index
            break
    if start is None:
        return None

    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def chat_json(
    prompt: str,
    tier: str = "main",
    system: str = "",
    temperature: float = 0.0,
    fallback: Optional[dict] = None,
    use_json_mode: bool = True,
) -> tuple[dict, LLMResult]:
    """要求模型输出 JSON 并解析。

    返回 `(data, result)`：解析失败时 data 为 fallback（默认空 dict），
    并把原始文本放进 `result.text` 供 trace 排障。
    """
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    result = chat(
        messages,
        tier=tier,
        temperature=temperature,
        response_format={"type": "json_object"} if use_json_mode else None,
    )
    if not result.ok:
        return (fallback or {}), result

    parsed = extract_json(result.text)
    if parsed is None:
        logger.warning(f"[llm] JSON 解析失败，原文前 200 字：{result.text[:200]}")
        result.error = "json_decode_failed"
        return (fallback or {}), result

    if isinstance(parsed, list):
        # 统一包一层，节点不用判断类型
        return {"items": parsed}, result
    return parsed, result


def record_usage(trace_id: str | None, result: LLMResult) -> None:
    """把 token 记到 trace，供成本核算。"""
    if not trace_id or not result:
        return
    try:
        from utils.trace_utils import record_tokens

        record_tokens(trace_id, result.prompt_tokens, result.completion_tokens)
    except Exception:
        logger.debug("[llm] token 记账跳过")


def health() -> dict:
    """`/health` 用：只回报配置状态，不真的发请求（避免探活打满额度）。"""
    return {
        "enabled": lm_config.enabled,
        "main_model": lm_config.llm_model,
        "light_model": lm_config.light_model,
        "base_url": lm_config.base_url,
    }
