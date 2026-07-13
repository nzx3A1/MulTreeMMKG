"""多模态知识抽取包的统一公开入口。"""

from .extractor_init import InitExtractor, collect_chunks

__all__ = [
    "InitExtractor",
    "collect_chunks",
]
