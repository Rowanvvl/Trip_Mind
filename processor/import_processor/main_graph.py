"""导入图装配。

```
START → node_entry ─┬─(失败)→ END
                    ├─(.pdf)→ node_pdf_to_md ─┐
                    └─(.md)──────────────────┴→ node_md_normalize
→ node_md_img → node_meta_extract → node_document_split
→ node_chunk_enrich → node_geo_resolve → node_bge_embedding → node_import_milvus → END
```

线性为主，唯一分支在入口（PDF / Markdown）。中间不做并行：
解析与切分都是 CPU 密集且强依赖前一步输出，并行只会增加复杂度。
"""
from __future__ import annotations

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from processor.import_processor.nodes.node_bge_embedding import node_bge_embedding
from processor.import_processor.nodes.node_chunk_enrich import node_chunk_enrich
from processor.import_processor.nodes.node_document_split import node_document_split
from processor.import_processor.nodes.node_entry import after_entry, node_entry
from processor.import_processor.nodes.node_geo_resolve import node_geo_resolve
from processor.import_processor.nodes.node_import_milvus import node_import_milvus
from processor.import_processor.nodes.node_md_img import node_md_img
from processor.import_processor.nodes.node_md_normalize import node_md_normalize
from processor.import_processor.nodes.node_meta_extract import node_meta_extract
from processor.import_processor.nodes.node_pdf_to_md import node_pdf_to_md
from processor.import_processor.state import ImportGraphState


def build_import_graph():
    builder = StateGraph(ImportGraphState)

    builder.add_node("node_entry", node_entry)
    builder.add_node("node_pdf_to_md", node_pdf_to_md)
    builder.add_node("node_md_normalize", node_md_normalize)
    builder.add_node("node_md_img", node_md_img)
    builder.add_node("node_meta_extract", node_meta_extract)
    builder.add_node("node_document_split", node_document_split)
    builder.add_node("node_chunk_enrich", node_chunk_enrich)
    builder.add_node("node_geo_resolve", node_geo_resolve)
    builder.add_node("node_bge_embedding", node_bge_embedding)
    builder.add_node("node_import_milvus", node_import_milvus)

    builder.add_edge(START, "node_entry")
    builder.add_conditional_edges(
        "node_entry",
        after_entry,
        {"end": END, "node_pdf_to_md": "node_pdf_to_md", "node_md_normalize": "node_md_normalize"},
    )
    builder.add_edge("node_pdf_to_md", "node_md_normalize")
    builder.add_edge("node_md_normalize", "node_md_img")
    builder.add_edge("node_md_img", "node_meta_extract")
    builder.add_edge("node_meta_extract", "node_document_split")
    builder.add_edge("node_document_split", "node_chunk_enrich")
    builder.add_edge("node_chunk_enrich", "node_geo_resolve")
    builder.add_edge("node_geo_resolve", "node_bge_embedding")
    builder.add_edge("node_bge_embedding", "node_import_milvus")
    builder.add_edge("node_import_milvus", END)

    return builder.compile()


import_app = build_import_graph()
