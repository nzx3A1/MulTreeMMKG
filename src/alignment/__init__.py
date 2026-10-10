"""Stage 06 多模态实体共指对齐与关系融合。

本包把同一文档中来自文本、图片和表格的实体 mention 聚合为文档级节点，
随后重写全部关系端点并汇集重复事实，同时保留原始 Chunk 与证据。
"""

from .pipeline import MultimodalAlignmentPipeline

__all__ = ["MultimodalAlignmentPipeline"]
