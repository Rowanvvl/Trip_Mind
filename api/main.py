"""FastAPI 入口。

启动命令：
```
uvicorn api.main:app --reload --port 8000
```

页面：
- `/`        问答页（`page/chat.html`）
- `/import`  导入页
- `/plan`    行程页

健康检查**不真的调外部接口**：探活打满地图额度是很常见的事故。
`/health` 只回报配置与连接状态，`/stats` 给缓存与工具指标。
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.import_service import router as import_router
from api.plan_service import router as plan_router
from api.query_service import router as query_router
from common.config import trip_config
from common.logging.logger import logger
from utils.path_util import PAGE_DIR

app = FastAPI(
    title="TripMind",
    description="旅游知识问答与行程规划服务",
    version="0.1.0",
)

# 本地开发放开跨域；上线要收窄到具体域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(import_router)
app.include_router(query_router)
app.include_router(plan_router)

PAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/page", StaticFiles(directory=str(PAGE_DIR)), name="page")


@app.get("/", include_in_schema=False)
async def index():
    chat_page = PAGE_DIR / "chat.html"
    if chat_page.exists():
        return FileResponse(chat_page)
    return JSONResponse({"service": "TripMind", "docs": "/docs"})


@app.get("/import", include_in_schema=False)
async def import_page():
    page = PAGE_DIR / "import.html"
    return FileResponse(page) if page.exists() else JSONResponse({"detail": "page 不存在"}, status_code=404)


@app.get("/plan", include_in_schema=False)
async def plan_page():
    page = PAGE_DIR / "plan.html"
    return FileResponse(page) if page.exists() else JSONResponse({"detail": "page 不存在"}, status_code=404)


@app.get("/health")
async def health() -> dict:
    """探活：只查配置与连接状态，不发外部请求。"""
    from utils.clients import milvus_utils, minio_utils, mongo_history_utils
    from utils.lm import embedding_health, llm_health, reranker_health

    return {
        "status": "ok",
        "llm": llm_health(),
        "embedding": embedding_health(),
        "reranker": reranker_health(),
        "milvus": milvus_utils.stats(),
        "mongo": {"available": mongo_history_utils.available()},
        "minio": {"available": minio_utils.available()},
        "config": {
            "guard_enabled": trip_config.guard_enabled,
            "trace_enabled": trip_config.trace_enabled,
            "evidence_min_coverage": trip_config.evidence_min_coverage,
        },
    }


@app.get("/stats")
async def stats() -> dict:
    """缓存命中率与工具调用指标。工具治理是否生效全看这两个数。"""
    from utils.cache_utils import cache_snapshot
    from utils.clients import milvus_utils
    from utils.geo.city_gazetteer import summarize
    from utils.tools import tool_metrics, tool_policies

    return {
        "cache": cache_snapshot(),
        "tools": tool_metrics(),
        "tool_policies": tool_policies(),
        "milvus": milvus_utils.stats(),
        "gazetteer": summarize(),
    }


@app.on_event("startup")
async def on_startup() -> None:
    """启动时建索引，失败不阻塞启动（降级可用比启动失败好）。"""
    logger.info("TripMind 启动")
    try:
        from utils.clients import mongo_history_utils

        mongo_history_utils.ensure_indexes()
    except Exception:
        logger.warning("Mongo 索引初始化跳过")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("TripMind 关闭")
