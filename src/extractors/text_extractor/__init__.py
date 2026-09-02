"""不依赖 Schema 的文本知识图谱抽取公开入口。"""
from .pipeline import extract_text_chunks_to_file
from .text_extractor import extract_from_text, extract_text_chunk_graph


__all__ = [
    "extract_from_text",
    "extract_text_chunk_graph",
    "extract_text_chunks_to_file",
]
