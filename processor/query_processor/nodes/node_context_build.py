"""上下文组装。

**这里不是「按字符预算截断」，而是压缩式打包**——两者的差别在效果上很明显：
截断经常把一段话的后半句（往往是结论）切掉，模型拿到前半句就开始编。

做法：
1. 每段切片保留「小节标题 + 首句 + 一个关键判断句」；
2. 关键判断句按判断词打分选出（「不要」「适合」「只有」「关键是」…），
   这正好对应语料的写法；
3. 总长度超出预算时，**从尾部整段丢弃**（而不是截半句话），
   并保证至少留 2 段。

编号在这里生成，答案里的 `[n]` 与 citations 的下标严格对应；
生成之后 `node_answer_critic` 会回查这个映射，防止引用错位。
"""
from __future__ import annotations

import re
from typing import List

from common.config import trip_config
from common.logging.logger import node_log
from processor.query_processor.state import QueryGraphState
from utils.format_utils import truncate

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；!?;])")
# 语料里的判断词。命中越多，这句越可能是结论
_JUDGE_WORDS = (
    "不如", "不要", "别", "适合", "不适合", "只能", "除非", "关键是", "问题在于",
    "重点", "注意", "建议", "区别", "反而是", "本质上", "代价", "风险", "更好",
    "取决于", "没必要", "值得", "优先级",
)


@node_log("node_context_build")
def node_context_build(state: QueryGraphState) -> dict:
    """压缩并编号。

    输出：`context`、`citations`、`debug.context`
    """
    docs = list(state.get("reranked_docs") or [])
    budget = trip_config.context_char_budget

    blocks: List[str] = []
    citations: List[dict] = []
    used_chars = 0
    dropped = 0

    for index, doc in enumerate(docs, start=1):
        body = _key_sentences(doc.get("text", ""), trip_config.per_chunk_sentences)
        if not body:
            continue

        header_bits = [doc.get("city") or "", doc.get("doc_title") or "", doc.get("section") or ""]
        header = " · ".join([b for b in header_bits if b])
        block = f"[{index}] {header}\n{body}"

        if used_chars + len(block) > budget and len(blocks) >= 2:
            dropped = len(docs) - index + 1
            break

        blocks.append(block)
        used_chars += len(block)
        citations.append(
            {
                "index": index,
                "chunk_id": doc.get("chunk_id", ""),
                "doc_id": doc.get("doc_id", 0),
                "doc_title": doc.get("doc_title", ""),
                "section": doc.get("section", ""),
                "city": doc.get("city", ""),
                "content_type": doc.get("content_type", ""),
                "source_path": doc.get("source_path", ""),
                "updated_at": doc.get("updated_at", ""),
                "rerank_score": doc.get("rerank_score"),
            }
        )

    # 网络兜底结果（如果有）排在知识库引用之后，编号连续
    for doc in state.get("web_docs") or []:
        index = len(citations) + 1
        snippet = truncate(doc.get("snippet", ""), 200)
        if not snippet:
            continue
        blocks.append(f"[{index}] 〔网络〕{doc.get('title', '')}\n{snippet}")
        citations.append(
            {
                "index": index,
                "chunk_id": "",
                "doc_title": doc.get("title", "网络结果"),
                "section": "",
                "city": "",
                "content_type": "网络",
                "source_path": doc.get("url", ""),
                "updated_at": "",
            }
        )

    context = "\n\n".join(blocks) if blocks else "（知识库未收录与本次提问相关的资料）"

    return {
        "context": context,
        "citations": citations,
        "debug": {
            **(state.get("debug") or {}),
            "context": {
                "blocks": len(blocks),
                "chars": len(context),
                "dropped": dropped,
                "budget": budget,
            },
        },
    }


def _key_sentences(text: str, limit: int = 2) -> str:
    """取首句 + 判断词得分最高的句子。"""
    clean = (text or "").strip()
    if not clean:
        return ""

    # 去掉切片自带的标题行（`文档名 · 小节`），标题已在 block 头部展示
    lines = [ln for ln in clean.splitlines() if ln.strip()]
    if lines and "·" in lines[0] and len(lines[0]) < 60:
        lines = lines[1:]
    body = " ".join(lines).strip()
    if not body:
        return ""

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(body) if s.strip()]
    if len(sentences) <= limit:
        return body

    first = sentences[0]
    rest = sentences[1:]
    best = max(rest, key=lambda s: sum(1 for w in _JUDGE_WORDS if w in s))
    chosen = [first] if best is first else [first, best]
    return "".join(dict.fromkeys(chosen))
