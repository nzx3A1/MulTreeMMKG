"""表格—图像嵌入混合型地层图的专用抽取能力。"""

from .graph import build_table_embedded_hybrid_graph
from .pipeline import (
    TABLE_EMBEDDED_HYBRID_SCHEMA_VERSION,
    TableEmbeddedHybridPipeline,
    is_table_embedded_hybrid_payload,
)
from .prompt import build_table_embedded_hybrid_prompt
from .ppstructure_geometry import (
    PPSTRUCTURE_GEOMETRY_SCHEMA_VERSION,
    PPStructureV3GeometryExtractor,
    apply_ppstructure_geometry,
    extract_ppstructure_geometry,
    geometry_prompt_catalog,
)
from .segmented_vlm import (
    NODE_ENRICHMENT_SCHEMA_VERSION,
    SEGMENT_ORDER,
    apply_node_enrichment,
    build_node_enrichment_prompt,
    enrich_table_node_names,
    extract_segmented_table_visual,
    merge_segmented_table_payloads,
    validate_and_repair_pixel_geometry,
)
from .visual_track_extraction import (
    VISUAL_TRACK_EXTRACTION_VERSION,
    VISUAL_TRACK_SLICE_SCHEMA_VERSION,
    VLM_TRACK_TYPES,
    apply_visual_track_slice_responses,
    build_adjacent_visual_track_slices,
    build_semantic_tracks,
    build_visual_track_slice_prompt,
    cache_curve_track_images,
    describe_adjacent_visual_track_slices,
    enrich_visual_track_primitives,
    sort_tracks_left_to_right,
    validate_vlm_track_types,
)
from .vlm_options import (
    DEFAULT_TABLE_EMBEDDED_HYBRID_VLM_TIMEOUT_SECS,
    TABLE_EMBEDDED_HYBRID_VLM_TIMEOUT_ENV,
    table_embedded_hybrid_vlm_timeout_secs,
)

__all__ = [
    "TABLE_EMBEDDED_HYBRID_SCHEMA_VERSION",
    "TableEmbeddedHybridPipeline",
    "build_table_embedded_hybrid_graph",
    "build_table_embedded_hybrid_prompt",
    "PPSTRUCTURE_GEOMETRY_SCHEMA_VERSION",
    "PPStructureV3GeometryExtractor",
    "apply_ppstructure_geometry",
    "extract_ppstructure_geometry",
    "geometry_prompt_catalog",
    "is_table_embedded_hybrid_payload",
    "NODE_ENRICHMENT_SCHEMA_VERSION",
    "SEGMENT_ORDER",
    "apply_node_enrichment",
    "build_node_enrichment_prompt",
    "enrich_table_node_names",
    "extract_segmented_table_visual",
    "merge_segmented_table_payloads",
    "validate_and_repair_pixel_geometry",
    "VISUAL_TRACK_EXTRACTION_VERSION",
    "VISUAL_TRACK_SLICE_SCHEMA_VERSION",
    "VLM_TRACK_TYPES",
    "apply_visual_track_slice_responses",
    "build_adjacent_visual_track_slices",
    "build_semantic_tracks",
    "build_visual_track_slice_prompt",
    "cache_curve_track_images",
    "describe_adjacent_visual_track_slices",
    "enrich_visual_track_primitives",
    "sort_tracks_left_to_right",
    "validate_vlm_track_types",
    "DEFAULT_TABLE_EMBEDDED_HYBRID_VLM_TIMEOUT_SECS",
    "TABLE_EMBEDDED_HYBRID_VLM_TIMEOUT_ENV",
    "table_embedded_hybrid_vlm_timeout_secs",
]
