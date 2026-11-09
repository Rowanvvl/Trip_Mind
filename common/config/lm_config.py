"""大模型配置。

模型分级（框架文档 12.7）：`understand / critic / guard / judge / rerank_refine`
走 `LLM_LIGHT_MODEL`，`answer / plan_generate` 走主模型。
调用方通过 `get_model(tier)` 取模型名，不要在业务代码里硬编码。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    vl_model: str = field(default_factory=lambda: os.getenv("VL_MODEL", "qwen-vl-max"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", "qwen-max"))
    light_model: str = field(default_factory=lambda: os.getenv("LLM_LIGHT_MODEL", "qwen-plus"))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.2")))
    # 生成类节点温度可单独调高一点，让建议更自然
    answer_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_ANSWER_TEMPERATURE", "0.3")))
    request_timeout: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "60")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "2")))

    @property
    def enabled(self) -> bool:
        """没配 key 时上层可走降级分支，而不是抛异常。"""
        return bool(self.api_key)

    def get_model(self, tier: str = "main") -> str:
        """按档位取模型名。tier: main | light"""
        return self.light_model if tier == "light" else self.llm_model


lm_config = LLMConfig()

# 各节点应使用的档位，集中登记，便于一处调整
NODE_MODEL_TIER = {
    "node_input_guard": "light",
    "node_query_understand": "light",
    "node_search_hyde": "light",
    "node_sub_query_retrieve": "light",
    "node_answer_critic": "light",
    "node_goal_check": "light",
    "node_answer_output": "main",
    "node_plan_generate": "main",
    "node_plan_validate": "light",
}
