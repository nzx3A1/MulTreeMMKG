"""多模态知识抽取包的统一公开入口。"""

from .extractor_init import InitExtractor, collect_chunks, extract_document_multimodal, extract_document_text

__all__ = [
    "InitExtractor",
    "collect_chunks",
    "extract_document_multimodal",
    "extract_document_text",
]
