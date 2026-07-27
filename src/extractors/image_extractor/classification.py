"""单图片分类信息提供器，隔离分类来源与抽取流水线。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from src.utils.json_io import read_json

from .schema_models import ImageExtractionTask, as_string_tuple
from .vlm_classification import VLMImageClassifier


@dataclass(frozen=True)
class ImageClassification:
    """单张图片路由所需的最小分类信息。"""

    code: str = ""
    type_name: str = ""


class ImageClassificationProvider(Protocol):
    """分类信息提供器协议，后续可接文件、数据库或分类模型结果。"""

    def resolve(self, chunk: Mapping[str, Any], image_path: str, image_index: int) -> ImageClassification:
        """返回指定 Chunk 中某张图片的分类。"""


def _read_classification(value: Any) -> ImageClassification:
    """中文说明：兼容嵌套 classification 或直接主分类字段。"""

    if not isinstance(value, Mapping):
        return ImageClassification()
    nested = value.get("classification")
    source = nested if isinstance(nested, Mapping) else value
    return ImageClassification(
        code=str(source.get("primary_code") or ""),
        type_name=str(source.get("primary_type") or ""),
    )


class InlineImageClassificationProvider:
    """优先读取单图分类列表，再回退到 ImageChunk 的统一分类。"""

    item_fields = ("image_classifications", "individual_image_classifications")

    def resolve(self, chunk: Mapping[str, Any], image_path: str, image_index: int) -> ImageClassification:
        """中文说明：按路径或图片序号匹配单图分类，保证多图 Chunk 可以分别路由。"""

        for field_name in self.item_fields:
            values = chunk.get(field_name)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                continue
            for item in values:
                if not isinstance(item, Mapping):
                    continue
                item_path = str(item.get("image_path") or "")
                item_index = item.get("image_index_in_chunk", item.get("image_index"))
                path_matches = bool(item_path) and item_path == image_path
                index_matches = item_index is not None and int(item_index) == image_index
                if path_matches or index_matches:
                    return _read_classification(item)
        return _read_classification(chunk.get("classification") or chunk.get("image_classification"))


def _path_key(value: str) -> str:
    """中文说明：规范化路径分隔符与大小写，便于匹配 Windows 分类记录。"""

    return value.replace("/", "\\").casefold()


class RecordImageClassificationProvider:
    """从逐图片分类记录提供路由信息，同时保留 Chunk 内联分类作为回退。"""

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        """中文说明：按父 Chunk+序号及图片路径建立索引，查询时不重复扫描全部记录。"""

        self._by_chunk_index: dict[tuple[str, int], ImageClassification] = {}
        self._by_path: dict[str, ImageClassification] = {}
        self._fallback = InlineImageClassificationProvider()
        for record in records:
            classification = _read_classification(record)
            chunk_id = str(record.get("parent_chunk_id") or record.get("chunk_id") or "")
            index = record.get("image_index_in_chunk", record.get("image_index"))
            image_path = str(record.get("image_path") or "")
            if chunk_id and index is not None:
                self._by_chunk_index[(chunk_id, int(index))] = classification
            if image_path:
                self._by_path[_path_key(image_path)] = classification

    @classmethod
    def from_json(cls, path: str | Path) -> "RecordImageClassificationProvider":
        """中文说明：读取逐图片分类 JSON，兼容根列表或包含 images 字段的对象。"""

        payload = read_json(path)
        records = payload.get("images", []) if isinstance(payload, Mapping) else payload
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TypeError("逐图片分类 JSON 必须是列表或包含 images 列表")
        return cls([record for record in records if isinstance(record, Mapping)])

    def resolve(self, chunk: Mapping[str, Any], image_path: str, image_index: int) -> ImageClassification:
        """中文说明：优先按 Chunk 与序号匹配，其次按路径匹配，最后回退内联分类。"""

        chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
        return (
            self._by_chunk_index.get((chunk_id, image_index))
            or self._by_path.get(_path_key(image_path))
            or self._fallback.resolve(chunk, image_path, image_index)
        )


class VLMImageClassificationProvider:
    """通过项目 VLM 对每张原图执行 A01-A20 真实视觉分类。"""

    def __init__(self, vlm_client: Any, classifier: VLMImageClassifier | None = None) -> None:
        """中文说明：保存模型客户端和逐图审计记录，分类失败时仍允许批处理继续。"""

        self.vlm_client = vlm_client
        self.classifier = classifier or VLMImageClassifier()
        self.records: list[dict[str, Any]] = []

    def resolve(self, chunk: Mapping[str, Any], image_path: str, image_index: int) -> ImageClassification:
        """中文说明：构造最小单图任务调用真实分类 API，并记录成功结果或可追踪错误。"""

        chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
        image_id = f"{chunk_id}:image:{image_index}"
        task = ImageExtractionTask(
            document_id=str(chunk.get("document_id") or chunk_id.split(":section:", 1)[0]),
            chunk_id=chunk_id,
            image_id=image_id,
            image_index=image_index,
            image_path=image_path,
            caption=str(chunk.get("caption") or ""),
            references=as_string_tuple(chunk.get("references")),
            section_id=str(chunk.get("section_id") or ""),
            section_title=str(chunk.get("section_title") or ""),
            raw_chunk=chunk,
        )
        try:
            result = self.classifier.classify(task, self.vlm_client)
        except Exception as exc:
            # 中文说明：分类失败统一路由到不处理的 A20，同时在 records 中保留原始错误供批量重试。
            self.records.append(
                {
                    "status": "failed",
                    "chunk_id": chunk_id,
                    "image_id": image_id,
                    "image_index": image_index,
                    "image_path": image_path,
                    "error": str(exc),
                }
            )
            return ImageClassification(code="A20", type_name="其他石油地质综合图")
        payload = result.to_dict()
        self.records.append(
            {
                "status": "completed",
                "chunk_id": chunk_id,
                "image_id": image_id,
                "image_index": image_index,
                "image_path": image_path,
                **payload,
            }
        )
        return ImageClassification(code=result.primary_code, type_name=result.primary_type)
