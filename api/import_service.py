"""导入服务：扫描、上传、任务进度、重试。

导入是长任务（一篇文档要解析+切分+编码+写库，几十秒起），所以全部走后台任务，
接口只返回 `task_id`，进度靠 `/import/status/{task_id}` 或 SSE 拿。

**单文件失败不影响整批**：每个文件是独立任务，一个坏文件只会让那个任务 failed，
其余照常入库。这一点在多文件扫描时尤其重要。
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from common.logging.logger import logger
from utils.path_util import data_dir, resolve_path
from utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    create_task,
    get_task_progress,
    is_task_finished,
    set_task_error,
    set_task_result,
    set_task_status,
)

router = APIRouter(prefix="/import", tags=["import"])


class ScanRequest(BaseModel):
    path: Optional[str] = Field(default=None, description="目录或文件路径，默认 data/")
    force: bool = Field(default=False, description="忽略增量判断，强制重导")


class ScanResponse(BaseModel):
    task_ids: List[str]
    files: int
    skipped: int = 0


class TaskResponse(BaseModel):
    task_id: str
    status: str
    percent: int
    running: List[str] = []
    done: List[str] = []
    error: Optional[str] = None
    result: dict = {}


def _run_import(task_id: str, source_path: str, force: bool) -> None:
    """后台跑导入图。异常必须被吞掉并写进任务状态，否则后台任务会静默消失。"""
    from processor.import_processor.main_graph import import_app
    from processor.import_processor.state import create_import_state

    create_task(task_id, graph="import")
    try:
        state = create_import_state(task_id=task_id, source_path=source_path, force=force)
        result = import_app.invoke(state)
    except Exception as exc:
        logger.exception(f"[import] 任务异常 {task_id}")
        set_task_error(task_id, f"{type(exc).__name__}: {exc}")
        return

    stats = result.get("stats") or {}
    if result.get("final_status") == "success":
        set_task_result(task_id, {"path": source_path, "stats": stats})
        set_task_status(task_id, TASK_STATUS_COMPLETED)
    else:
        set_task_error(task_id, result.get("error") or "导入失败")
        set_task_result(task_id, {"path": source_path, "stats": stats})


@router.post("/scan", response_model=ScanResponse)
async def scan(request: ScanRequest, background: BackgroundTasks) -> ScanResponse:
    """扫描目录下所有 Markdown 并逐个建任务。"""
    target = resolve_path(request.path) if request.path else data_dir()
    if not target.exists():
        raise HTTPException(status_code=400, detail=f"路径不存在：{target}")

    files = [target] if target.is_file() else sorted(target.rglob("*.md"))
    if not files:
        raise HTTPException(status_code=400, detail="目录下没有 Markdown 文件")

    task_ids: List[str] = []
    for file_path in files:
        task_id = uuid.uuid4().hex[:16]
        task_ids.append(task_id)
        background.add_task(_run_import, task_id, str(file_path), request.force)

    logger.info(f"[import] 提交 {len(files)} 个导入任务")
    return ScanResponse(task_ids=task_ids, files=len(files))


@router.post("/upload", response_model=ScanResponse)
async def upload(background: BackgroundTasks, file: UploadFile = File(...), force: bool = False) -> ScanResponse:
    """上传单个文件后立即导入。落盘位置：data/uploads/。"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名缺失")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".md", ".markdown", ".pdf"}:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{suffix}")

    upload_dir = data_dir("uploads")
    target = upload_dir / file.filename
    target.write_bytes(await file.read())

    task_id = uuid.uuid4().hex[:16]
    background.add_task(_run_import, task_id, str(target), force)
    return ScanResponse(task_ids=[task_id], files=1)


@router.get("/status/{task_id}", response_model=TaskResponse)
async def status(task_id: str) -> TaskResponse:
    """查询任务进度。任务不存在时返回 pending，而不是 404——前端轮询更省事。"""
    progress = get_task_progress(task_id)
    return TaskResponse(
        task_id=task_id,
        status=progress.get("status", "pending"),
        percent=progress.get("percent", 0),
        running=progress.get("running", []),
        done=progress.get("done", []),
        error=progress.get("error"),
        result=progress.get("result", {}),
    )


@router.post("/retry/{task_id}", response_model=ScanResponse)
async def retry(task_id: str, background: BackgroundTasks) -> ScanResponse:
    """重试失败任务。只允许对已结束且失败的任务重试。"""
    progress = get_task_progress(task_id)
    if not is_task_finished(task_id):
        raise HTTPException(status_code=409, detail="任务仍在运行")
    if progress.get("status") != TASK_STATUS_FAILED:
        raise HTTPException(status_code=409, detail="只有失败任务可以重试")

    source_path = (progress.get("result") or {}).get("path") or ""
    if not source_path:
        raise HTTPException(status_code=400, detail="任务缺少源文件路径，无法重试")

    background.add_task(_run_import, task_id, source_path, True)
    return ScanResponse(task_ids=[task_id], files=1)
