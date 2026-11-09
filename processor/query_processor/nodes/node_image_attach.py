"""配图（答案之后）。

三个刻意的设计：
1. **放在答案之后**：图片要等答案出来才知道该配哪些景点，且不能占用首字延迟；
2. **失败静默**：任何异常都吞掉，图片是锦上添花，不能因为它让答案失败；
3. **标注来源**：语料图 / 高德 POI 图分开标，不能把 POI 图说成官方图。
"""
from __future__ import annotations

from common.config import trip_config
from common.logging.logger import logger, node_log
from processor.query_processor.state import QueryGraphState
from utils.geo.city_gazetteer import find_attractions_in_text
from utils.sse_utils import SSEEvent, push_to_session
from utils.tools import image_utils


@node_log("node_image_attach")
def node_image_attach(state: QueryGraphState) -> dict:
    """按答案提到的景点挂图。

    输出：`images`
    """
    if trip_config.image_max_per_answer <= 0:
        return {"images": []}

    answer = state.get("answer") or ""
    city = state.get("city") or ""
    places = find_attractions_in_text(answer, city) or find_attractions_in_text(
        state.get("original_query") or "", city
    )
    entities = [{"name": p.name, "city": p.city} for p in places][: trip_config.image_max_per_answer]

    if not entities:
        return {"images": []}

    result = image_utils.attach_safe(entities, city=city or None)
    images = result.get("items") or []

    if images:
        logger.info(f"[image_attach] 挂图 {len(images)} 张 city={city}")
        # 独立事件下发，不阻塞 final
        push_to_session(state.get("session_id") or "", SSEEvent.IMAGES, {"items": images})

    return {"images": images}
