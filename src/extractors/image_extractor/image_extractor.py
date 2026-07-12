"""图片 Chunk 的项目适配抽取器。"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from model import Graph
from src.utils.llm_client import safe_json_loads

from ..schema_constrained import extract_chunk_graph
from ..schema_router import SchemaSelector


def _mapping(chunk: Any) -> Mapping[str, Any]:
    """把图片 Chunk 模型或字典统一转换为映射。"""

    if isinstance(chunk, Mapping):
        return chunk
    if hasattr(chunk, "to_dict"):
        return chunk.to_dict()
    if hasattr(chunk, "dict"):
        return chunk.dict()
    raise TypeError(f"不支持的图片 Chunk 类型：{type(chunk).__name__}")


def _image_paths(item: Mapping[str, Any]) -> list[str]:
    """规范化单图或多图路径字段。"""

    value = item.get("image_path") or item.get("image_paths") or item.get("path") or []
    if isinstance(value, (str, bytes)):
        return [str(value)]
    return [str(path) for path in value] if isinstance(value, Sequence) else []


def _context_text(item: Mapping[str, Any]) -> str:
    """组合图注、引用正文和章节上下文作为图片的文本证据。"""

    references = item.get("references") or []
    if isinstance(references, str):
        references = [references]
    return "\n".join(
        part
        for part in (
            str(item.get("caption") or ""),
            "\n".join(str(value) for value in references),
            str(item.get("context") or ""),
        )
        if part.strip()
    )


def _describe_images(vlm_client: Any, paths: Sequence[str], context: str) -> str:
    """对每张图片做一次简洁预分析，结果同时用于 Schema 路由和证据校验。"""

    descriptions: list[str] = []
    prompt = (
        "请对这张石油地质图片做简洁预分析，识别图件类型、可见地质对象、地层、井、构造、"
        "曲线或实验信息。只描述图中及图注明确可见内容。\n上下文：" + context
    )
    for path in paths:
        if hasattr(vlm_client, "describe_image"):
            descriptions.append(str(vlm_client.describe_image(path, prompt) or ""))
    return "\n".join(value for value in descriptions if value.strip())


def extract_from_images(
    chunks: Sequence[Any],
    llm_client: Any,
    vlm_client: Any,
    schema_selector: Any | None = None,
) -> list[Graph]:
    """先以图片预分析选择 Schema 树，再携带原图完成实体与关系抽取。"""

    selector = schema_selector or SchemaSelector()
    graphs: list[Graph] = []
    for chunk in chunks:
        item = _mapping(chunk)
        paths = _image_paths(item)
        context = _context_text(item)
        visual_description = _describe_images(vlm_client, paths, context)
        source_text = "\n".join(value for value in (context, visual_description) if value.strip())
        relevant = selector.select(
            source_text,
            modality="image",
            context={"section_title": item.get("section_title"), "caption": item.get("caption")},
        )

        def visual_json_call(system: str, payload: Mapping[str, Any]) -> Any:
            """将统一结构化 Prompt 与当前 Chunk 首张原图一起提交给视觉模型。"""

            prompt = f"{system}\n\n{json.dumps(payload, ensure_ascii=False)}"
            if paths and hasattr(vlm_client, "describe_image"):
                return safe_json_loads(vlm_client.describe_image(paths[0], prompt))
            return {}

        graphs.append(
            extract_chunk_graph(
                item,
                "image",
                source_text,
                relevant,
                llm_client,
                entity_call=visual_json_call,
                relation_call=visual_json_call,
                extractor_name="image_schema_extractor",
            )
        )
        graphs[-1].metadata.extra["image_paths"] = paths
        graphs[-1].metadata.extra["visual_description"] = visual_description
    return graphs


__all__ = ["extract_from_images"]
