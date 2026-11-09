"""Mongo 封装：会话、消息、槽位快照、长期偏好、行程、trace、反馈。

六个集合，职责不重叠：
- `chat_sessions`  会话元信息（城市、最近活跃时间）
- `chat_messages`  原始消息 + 引用 + **槽位快照**（每轮存一份，便于回放）
- `user_profiles`  跨会话偏好（带 confirmed / confidence）
- `itineraries`    行程结果
- `traces`         请求轨迹
- `feedback`       用户反馈（喂评测集）

所有函数失败都返回空值而不是抛异常：**记忆是增强，不是主链路**。
Mongo 挂了应该降级成单轮问答，而不是整个服务不可用。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from common.config import mongo_config
from common.logging.logger import logger
from utils.format_utils import now_iso

_db = None
_unavailable = False


def get_db():
    global _db, _unavailable
    if _db is not None or _unavailable:
        return _db
    try:
        from pymongo import MongoClient

        client = MongoClient(mongo_config.uri, serverSelectionTimeoutMS=mongo_config.timeout_ms)
        client.admin.command("ping")
        _db = client[mongo_config.db_name]
        logger.info(f"[mongo] 已连接 {mongo_config.db_name}")
    except Exception as exc:
        _unavailable = True
        logger.warning(f"[mongo] 连接失败（{type(exc).__name__}: {exc}），记忆功能降级")
        _db = None
    return _db


def available() -> bool:
    return get_db() is not None


def _col(name: str):
    db = get_db()
    return db[name] if db is not None else None


# ---------------------------------------------------------------- 会话与消息
def upsert_session(session_id: str, **fields: Any) -> bool:
    col = _col(mongo_config.sessions)
    if col is None:
        return False
    try:
        col.update_one(
            {"session_id": session_id},
            {"$set": {**fields, "updated_at": now_iso()}, "$setOnInsert": {"created_at": now_iso()}},
            upsert=True,
        )
        return True
    except Exception:
        logger.exception("[mongo] upsert_session 失败")
        return False


def append_message(
    session_id: str,
    role: str,
    text: str,
    citations: Optional[List[dict]] = None,
    city: str = "",
    slots: Optional[dict] = None,
    facts: Optional[dict] = None,
    images: Optional[List[str]] = None,
    trace_id: str = "",
) -> bool:
    """写一条消息。`slots_snapshot` 让「为什么第三轮变聪明了/变笨了」可回溯。"""
    col = _col(mongo_config.messages)
    if col is None:
        return False
    doc = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "citations": citations or [],
        "city": city,
        "slots_snapshot": slots or {},
        "facts": facts or {},
        "image_urls": images or [],
        "trace_id": trace_id,
        "ts": now_iso(),
    }
    try:
        col.insert_one(doc)
        return True
    except Exception:
        logger.exception("[mongo] append_message 失败")
        return False


def get_history(session_id: str, limit_turns: int | None = None) -> List[dict]:
    """取最近若干轮消息，**时间正序**返回（prompt 需要正序）。"""
    col = _col(mongo_config.messages)
    if col is None:
        return []
    turns = limit_turns or mongo_config.history_keep_turns
    try:
        cursor = col.find({"session_id": session_id}).sort("_id", -1).limit(turns * 2)
        items = list(cursor)[::-1]
        return [
            {
                "role": item.get("role", "user"),
                "text": item.get("text", ""),
                "city": item.get("city", ""),
                "slots": item.get("slots_snapshot", {}),
                "ts": item.get("ts", ""),
            }
            for item in items
        ]
    except Exception:
        logger.exception("[mongo] get_history 失败")
        return []


def clear_history(session_id: str) -> int:
    col = _col(mongo_config.messages)
    if col is None:
        return 0
    try:
        return col.delete_many({"session_id": session_id}).deleted_count
    except Exception:
        logger.exception("[mongo] clear_history 失败")
        return 0


def list_sessions(limit: int = 20) -> List[dict]:
    col = _col(mongo_config.sessions)
    if col is None:
        return []
    try:
        return list(col.find({}, {"_id": 0}).sort("updated_at", -1).limit(limit))
    except Exception:
        logger.exception("[mongo] list_sessions 失败")
        return []


# ---------------------------------------------------------------- 长期偏好
def get_profile(user_id: str) -> Dict:
    col = _col(mongo_config.profiles)
    if col is None or not user_id:
        return {}
    try:
        doc = col.find_one({"user_id": user_id}, {"_id": 0})
        return doc or {}
    except Exception:
        logger.exception("[mongo] get_profile 失败")
        return {}


def save_profile(user_id: str, profile: Dict) -> bool:
    col = _col(mongo_config.profiles)
    if col is None or not user_id:
        return False
    try:
        payload = {k: v for k, v in profile.items() if k != "_id"}
        payload.update({"user_id": user_id, "updated_at": now_iso()})
        col.update_one({"user_id": user_id}, {"$set": payload}, upsert=True)
        return True
    except Exception:
        logger.exception("[mongo] save_profile 失败")
        return False


# ---------------------------------------------------------------- 行程
def save_itinerary(plan: Dict) -> bool:
    col = _col(mongo_config.itineraries)
    if col is None:
        return False
    try:
        payload = dict(plan)
        payload.setdefault("created_at", now_iso())
        col.update_one({"plan_id": payload.get("plan_id")}, {"$set": payload}, upsert=True)
        return True
    except Exception:
        logger.exception("[mongo] save_itinerary 失败")
        return False


def get_itinerary(plan_id: str) -> Dict:
    col = _col(mongo_config.itineraries)
    if col is None:
        return {}
    try:
        return col.find_one({"plan_id": plan_id}, {"_id": 0}) or {}
    except Exception:
        logger.exception("[mongo] get_itinerary 失败")
        return {}


# ---------------------------------------------------------------- trace 与反馈
def save_trace(trace: Dict) -> bool:
    col = _col(mongo_config.traces)
    if col is None:
        return False
    try:
        col.insert_one(dict(trace))
        return True
    except Exception:
        logger.debug("[mongo] save_trace 失败")
        return False


def get_trace(trace_id: str) -> Dict:
    col = _col(mongo_config.traces)
    if col is None:
        return {}
    try:
        return col.find_one({"trace_id": trace_id}, {"_id": 0}) or {}
    except Exception:
        return {}


def save_feedback(payload: Dict) -> bool:
    col = _col(mongo_config.feedback)
    if col is None:
        return False
    try:
        col.insert_one({**payload, "ts": now_iso()})
        return True
    except Exception:
        logger.exception("[mongo] save_feedback 失败")
        return False


def ensure_indexes() -> dict:
    """首次部署跑一次即可。没有索引时历史查询会随消息量线性变慢。"""
    db = get_db()
    if db is None:
        return {"ok": False, "error": "mongo_unavailable"}
    try:
        db[mongo_config.messages].create_index([("session_id", 1), ("_id", -1)])
        db[mongo_config.sessions].create_index([("session_id", 1)], unique=True)
        db[mongo_config.profiles].create_index([("user_id", 1)], unique=True)
        db[mongo_config.itineraries].create_index([("plan_id", 1)], unique=True)
        db[mongo_config.traces].create_index([("trace_id", 1)], unique=True)
        db[mongo_config.traces].create_index([("session_id", 1), ("started_at", -1)])
        return {"ok": True}
    except Exception as exc:
        logger.exception("[mongo] 建索引失败")
        return {"ok": False, "error": str(exc)}
