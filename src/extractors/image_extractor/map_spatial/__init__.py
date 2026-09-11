"""地图与平面空间图片的分阶段抽取器公开接口。"""

from .assembler import DeterministicRelationAssembler
from .batch import MapSpatialClassificationProvider, extract_map_spatial_chunks
from .extractor import MapSpatialExtractor
from .graph import build_map_spatial_graph, normalize_map_spatial_result
from .prompt import (
    build_entity_prompt,
    build_legend_prompt,
    build_map_spatial_prompt,
    build_relation_prompt,
)
from .relation_schema import RelationSchemaGenerator
from .stages import LegendGrounder, LegendParser, MapEntityExtractor, SemanticRelationExtractor

__all__ = [
    "DeterministicRelationAssembler",
    "LegendGrounder",
    "LegendParser",
    "MapSpatialExtractor",
    "MapSpatialClassificationProvider",
    "MapEntityExtractor",
    "RelationSchemaGenerator",
    "SemanticRelationExtractor",
    "build_map_spatial_graph",
    "build_entity_prompt",
    "build_legend_prompt",
    "build_map_spatial_prompt",
    "build_relation_prompt",
    "extract_map_spatial_chunks",
    "normalize_map_spatial_result",
]
