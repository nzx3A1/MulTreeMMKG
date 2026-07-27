"""表格—图像嵌入混合型地层图的专用抽取能力。"""

from .graph import build_table_embedded_hybrid_graph
from .pipeline import (
    TABLE_EMBEDDED_HYBRID_SCHEMA_VERSION,
    TableEmbeddedHybridPipeline,
    is_table_embedded_hybrid_payload,
)
from .prompt import build_table_embedded_hybrid_prompt
from .segmented_vlm import (
    SEGMENT_ORDER,
    extract_segmented_table_visual,
    merge_segmented_table_payloads,
    validate_and_repair_pixel_geometry,
)

__all__ = [
    "TABLE_EMBEDDED_HYBRID_SCHEMA_VERSION",
    "TableEmbeddedHybridPipeline",
    "build_table_embedded_hybrid_graph",
    "build_table_embedded_hybrid_prompt",
    "is_table_embedded_hybrid_payload",
    "SEGMENT_ORDER",
    "extract_segmented_table_visual",
    "merge_segmented_table_payloads",
    "validate_and_repair_pixel_geometry",
]
