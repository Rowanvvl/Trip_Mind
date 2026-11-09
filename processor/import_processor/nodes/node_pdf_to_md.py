"""PDF → Markdown。

默认关闭（`MINERU_ENABLED=0`）：本项目语料以 Markdown 为主，PDF 只是兜底。
开启后调用 MinerU，产物落到 `output/`，并把正文读回 state。

**未接入时的行为**：明确报错并短路，而不是返回空字符串继续跑——
空字符串会让下游切出 0 个切片，最终表现成"导入成功但检索不到"，
这种静默失败排查成本远高于直接报错。
"""
from __future__ import annotations

from pathlib import Path

import httpx

from common.config import mineru_config
from common.logging.logger import logger, node_log
from processor.import_processor.state import ImportGraphState
from utils.path_util import ensure_dir


@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> dict:
    """调 MinerU 解析 PDF。

    输出：`markdown`、`markdown_path`、`stage`
    TODO(M1)：按实际 MinerU 版本对齐 `/file_parse` 的响应结构；
    当前实现按「返回 markdown 文本」的通用约定编写。
    """
    if not mineru_config.enabled:
        return {
            "error": "PDF 分支未启用（设置 MINERU_ENABLED=1 后可用），请先转换为 Markdown",
            "final_status": "failed",
            "stage": "pdf_to_md",
        }

    pdf_path = Path(state.get("source_path") or "")
    out_dir = ensure_dir(mineru_config.output_dir)

    try:
        with httpx.Client(timeout=mineru_config.timeout) as client:
            with pdf_path.open("rb") as handle:
                resp = client.post(
                    f"{mineru_config.base_url.rstrip('/')}/file_parse",
                    files={"file": (pdf_path.name, handle, "application/pdf")},
                    data={"backend": mineru_config.backend, "return_content_list": "false"},
                )
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        logger.exception("[pdf_to_md] MinerU 调用失败")
        return {"error": f"MinerU 调用失败：{exc}", "final_status": "failed", "stage": "pdf_to_md"}

    markdown = payload.get("md_content") or payload.get("markdown") or ""
    if not markdown:
        return {"error": "MinerU 未返回 Markdown 内容", "final_status": "failed", "stage": "pdf_to_md"}

    md_path = out_dir / f"{pdf_path.stem}.md"
    md_path.write_text(markdown, encoding="utf-8")

    return {
        "markdown": markdown,
        "markdown_path": str(md_path),
        "stage": "pdf_to_md",
        "stats": {**(state.get("stats") or {}), "pdf_to_md": {"chars": len(markdown)}},
    }
