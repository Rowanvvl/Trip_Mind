"""Mongo 配置：会话、槽位、长期偏好、trace、行程记录。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class MongoConfig:
    uri: str = field(default_factory=lambda: os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017"))
    db_name: str = field(default_factory=lambda: os.getenv("MONGO_DB", "tripmind"))
    timeout_ms: int = field(default_factory=lambda: int(os.getenv("MONGO_TIMEOUT_MS", "3000")))

    # 集合名
    sessions: str = "chat_sessions"
    messages: str = "chat_messages"
    itineraries: str = "itineraries"
    profiles: str = "user_profiles"
    traces: str = "traces"
    feedback: str = "feedback"

    # 历史保留轮数（一轮 = 一问一答）
    history_keep_turns: int = field(default_factory=lambda: int(os.getenv("HISTORY_KEEP_TURNS", "3")))


mongo_config = MongoConfig()
