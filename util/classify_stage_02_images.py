"""递归提取 Stage 02 文档树中的图片，并生成图片分类前端可直接读取的数据集。

脚本会把多图 ImageChunk 拆成逐图片记录，调用项目现有 VLM 执行 A01-A20 分类，
同时保留父 Chunk、章节、图注和正文引用等溯源信息，支持在中断后复用已有结果续跑。
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


# 中文说明：支持从任意工作目录直接执行脚本，并复用项目内已有分类器和 JSON 适配逻辑。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.extractor_init import collect_chunks
from src.extractors.image_extractor.schema_models import ImageExtractionTask, as_string_tuple
from src.extractors.image_extractor.vlm_classification import IMAGE_TAXONOMY, TYPE_BY_CODE, VLMImageClassifier
from src.utils.json_io import read_json
from src.utils.vlm_client import VLMClient


DEFAULT_INPUT = PROJECT_ROOT / "output" / "stage_02_document_tree.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "stage_02_individual_images_classified.json"
VALID_CODES = set(TYPE_BY_CODE)


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """校验 JSON 对象类型，并在输入结构错误时指出具体位置。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必须是 JSON 对象，实际为 {type(value).__name__}")
    return value


def _is_image_chunk(chunk: Mapping[str, Any]) -> bool:
    """兼容多种模态字段写法，只选择图片 Chunk。"""

    value = chunk.get("modality") or chunk.get("chunk_type") or chunk.get("type") or ""
    modality = str(getattr(value, "value", value)).strip().lower().replace("_", "")
    return modality in {"image", "imagechunk"}


def _string_list(value: Any) -> list[str]:
    """把可选字符串或字符串序列规范化为无空值列表。"""

    if value is None:
        return []
    if isinstance(value, (str, Path)):
        text = str(value).strip()
        return [text] if text else []
    if not isinstance(value, Sequence):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _stable_parent_id(document_id: str, chunk_id: str) -> str:
    """将文档内 Chunk ID 扩展为跨文档稳定且适合前端展示的父 ID。"""

    prefix = f"{document_id}:section:"
    return chunk_id if chunk_id.startswith(prefix) else f"{prefix}{chunk_id}"


def collect_image_chunks(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """递归收集 Stage 02 全部章节的 ImageChunk，并补齐稳定溯源字段。"""

    chunks = [dict(chunk) for chunk in collect_chunks(payload) if _is_image_chunk(chunk)]
    document = _as_mapping(payload.get("document", payload), "document")
    document_id = str(document.get("id") or document.get("document_id") or "").strip()
    if not document_id:
        raise ValueError("Stage 02 文档树缺少 document.id")

    seen_ids: set[str] = set()
    for chunk in chunks:
        chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "").strip()
        if not chunk_id:
            raise ValueError("发现缺少 id 的 ImageChunk")
        parent_id = _stable_parent_id(document_id, chunk_id)
        if parent_id in seen_ids:
            raise ValueError(f"ImageChunk 稳定 ID 重复：{parent_id}")
        seen_ids.add(parent_id)
        chunk["source_chunk_id"] = chunk_id
        chunk["id"] = parent_id
        chunk["document_id"] = document_id
        chunk["image_path"] = _string_list(chunk.get("image_path"))
        chunk["references"] = _string_list(chunk.get("references")) or None
    return chunks


def _base_image_record(chunk: Mapping[str, Any], image_path: str, image_index: int) -> dict[str, Any]:
    """从一个父 Chunk 构造前端需要的单图片基础记录。"""

    parent_id = str(chunk["id"])
    return {
        "image_id": f"{parent_id}:asset:{image_index}",
        "parent_chunk_id": parent_id,
        "image_index_in_chunk": image_index,
        "image_path": image_path,
        "caption": str(chunk.get("caption") or ""),
        "document_id": str(chunk.get("document_id") or ""),
        "section_id": str(chunk.get("section_id") or ""),
        "section_title": str(chunk.get("section_title") or ""),
        "source_chunk_id": str(chunk.get("source_chunk_id") or ""),
        "source_order": int(chunk.get("order") or 0),
        "source_references": _string_list(chunk.get("references")) or None,
    }


def build_image_records(chunks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """按父 Chunk 顺序拆分物理图片，并生成稳定且唯一的逐图记录。"""

    records: list[dict[str, Any]] = []
    for chunk in chunks:
        for image_index, image_path in enumerate(_string_list(chunk.get("image_path"))):
            records.append(_base_image_record(chunk, image_path, image_index))
    for review_index, record in enumerate(records, start=1):
        record["review_index"] = review_index
    if len({record["image_id"] for record in records}) != len(records):
        raise ValueError("拆分后的 image_id 存在重复")
    return records


def _load_reusable_records(output_path: Path) -> dict[str, Mapping[str, Any]]:
    """读取上次成功分类记录，按 image_id 建立续跑索引。"""

    if not output_path.is_file():
        return {}
    try:
        payload = read_json(output_path)
    except Exception:
        return {}
    if not isinstance(payload, Mapping) or not isinstance(payload.get("images"), list):
        return {}

    reusable: dict[str, Mapping[str, Any]] = {}
    for raw_record in payload["images"]:
        if not isinstance(raw_record, Mapping):
            continue
        classification = raw_record.get("classification")
        code = classification.get("primary_code") if isinstance(classification, Mapping) else None
        image_id = str(raw_record.get("image_id") or "")
        if image_id and code in VALID_CODES and raw_record.get("status", "completed") == "completed":
            reusable[image_id] = raw_record
    return reusable


def _classification_task(chunk: Mapping[str, Any], record: Mapping[str, Any]) -> ImageExtractionTask:
    """把逐图记录转换为项目现有 VLM 分类器所需的最小任务对象。"""

    return ImageExtractionTask(
        document_id=str(chunk.get("document_id") or ""),
        chunk_id=str(chunk["id"]),
        image_id=str(record["image_id"]),
        image_index=int(record["image_index_in_chunk"]),
        image_path=str(record["image_path"]),
        caption=str(chunk.get("caption") or ""),
        references=as_string_tuple(chunk.get("references")),
        section_id=str(chunk.get("section_id") or ""),
        section_title=str(chunk.get("section_title") or ""),
        raw_chunk=chunk,
    )


def _failed_classification(message: str) -> dict[str, Any]:
    """为失败记录生成不冒充 A01-A20 结论的前端占位分类。"""

    return {
        "primary_code": "UNCLASSIFIED",
        "primary_type": "分类失败，待重试",
        "secondary_codes": [],
        "secondary_types": [],
        "confidence": 0.0,
        "reason": message,
        "visual_evidence": [],
        "basis": "vlm_visual_classification_failed",
        "source": "vlm_visual_classification_failed",
    }


def _parent_classification(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """以父 Chunk 内成功单图分类的多数类别形成可比较的父分类。"""

    completed = [record for record in records if record.get("status") == "completed"]
    if not completed:
        return _failed_classification("父 Chunk 内没有成功的单图分类结果")
    counts = Counter(str(record["classification"]["primary_code"]) for record in completed)
    first_order = {str(record["classification"]["primary_code"]): index for index, record in enumerate(completed)}
    code = min(counts, key=lambda item: (-counts[item], first_order[item]))
    matching = [record for record in completed if record["classification"]["primary_code"] == code]
    confidence = sum(float(record["classification"].get("confidence") or 0) for record in matching) / len(matching)
    return {
        "primary_code": code,
        "primary_type": TYPE_BY_CODE[code],
        "secondary_codes": [],
        "secondary_types": [],
        "confidence": round(confidence, 3),
        "reason": f"父 Chunk 共 {len(records)} 张图片，按成功单图分类的多数类别汇总。",
        "basis": "individual_image_majority_vote",
        "source": "individual_image_majority_vote",
    }


def _attach_parent_results(chunks: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    """回填父 Chunk 汇总类别，并标注单图是否与父类别不同。"""

    records_by_parent: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_parent.setdefault(str(record["parent_chunk_id"]), []).append(record)
    for chunk in chunks:
        children = records_by_parent.get(str(chunk["id"]), [])
        parent = _parent_classification(children)
        chunk["classification"] = parent
        for record in children:
            classification = record["classification"]
            classification["parent_primary_code"] = parent["primary_code"]
            classification["parent_primary_type"] = parent["primary_type"]
            classification["changed_from_parent_chunk"] = (
                record.get("status") == "completed"
                and classification["primary_code"] != parent["primary_code"]
            )


def build_output(
    input_path: Path,
    chunks: list[dict[str, Any]],
    records: list[dict[str, Any]],
    reused_count: int,
) -> dict[str, Any]:
    """组装页面可直接读取的分类数据集、父 Chunk 和统计信息。"""

    _attach_parent_results(chunks, records)
    completed = [record for record in records if record.get("status") == "completed"]
    failed = [record for record in records if record.get("status") == "failed"]
    counts = Counter(record["classification"]["primary_code"] for record in completed)
    return {
        "_stage": "stage_02_individual_images_classified",
        "_description": "从 Stage 02 文档树递归提取并经 VLM 逐图分类的前端展示数据。",
        "_produced_by": "util/classify_stage_02_images.py",
        "source_file": str(input_path.resolve()),
        "taxonomy": [{"code": code, "name": name} for code, name in IMAGE_TAXONOMY],
        "statistics": {
            "total_image_chunks": len(chunks),
            "total_images": len(records),
            "existing_image_files": sum(Path(record["image_path"]).is_file() for record in records),
            "completed": len(completed),
            "failed": len(failed),
            "reused": reused_count,
            "classification_counts": dict(sorted(counts.items())),
        },
        "chunks": chunks,
        "images": records,
    }


def write_output(path: Path, payload: Mapping[str, Any]) -> None:
    """以临时文件替换方式写入 UTF-8 JSON，避免中断留下半个文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def classify_stage_02_images(
    payload: Mapping[str, Any],
    *,
    input_path: Path,
    output_path: Path,
    classifier: Any,
    vlm_client: Any,
    force: bool = False,
) -> dict[str, Any]:
    """逐图执行分类，成功一张便保存一次检查点，并返回最终前端数据集。"""

    chunks = collect_image_chunks(payload)
    records = build_image_records(chunks)
    chunk_by_id = {str(chunk["id"]): chunk for chunk in chunks}
    reusable = {} if force else _load_reusable_records(output_path)
    reused_count = 0

    for index, record in enumerate(records, start=1):
        previous = reusable.get(str(record["image_id"]))
        if previous and str(previous.get("image_path")) == str(record["image_path"]):
            record["classification"] = dict(previous["classification"])
            record["status"] = "completed"
            reused_count += 1
            print(f"[{index}/{len(records)}] 复用：{record['image_id']}", flush=True)
            continue

        image_path = Path(str(record["image_path"]))
        try:
            if not image_path.is_file():
                raise FileNotFoundError(f"图片不存在：{image_path}")
            chunk = chunk_by_id[str(record["parent_chunk_id"])]
            result = classifier.classify(_classification_task(chunk, record), vlm_client)
            record["classification"] = result.to_dict()
            record["status"] = "completed"
            print(
                f"[{index}/{len(records)}] {result.primary_code} {result.primary_type}：{image_path.name}",
                flush=True,
            )
        except Exception as exc:  # 中文说明：保留失败项继续处理剩余图片，随后可直接续跑失败记录。
            record["status"] = "failed"
            record["error"] = str(exc)
            record["classification"] = _failed_classification(str(exc))
            print(f"[{index}/{len(records)}] 失败：{record['image_id']}：{exc}", file=sys.stderr, flush=True)

        # 中文说明：每次模型调用后更新检查点，网络中断时无需重复已完成的视觉请求。
        write_output(output_path, build_output(input_path, chunks, records, reused_count))

    result = build_output(input_path, chunks, records, reused_count)
    write_output(output_path, result)
    return result


def main() -> None:
    """解析命令行参数，调用真实 VLM 分类并打印最终统计。"""

    parser = argparse.ArgumentParser(description="分类 Stage 02 文档树中的全部图片并生成前端展示 JSON")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Stage 02 文档树 JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="图片分类前端数据 JSON")
    parser.add_argument("--force", action="store_true", help="忽略已有成功记录并重新分类全部图片")
    args = parser.parse_args()

    payload = _as_mapping(read_json(args.input), "Stage 02 输入")
    result = classify_stage_02_images(
        payload,
        input_path=args.input,
        output_path=args.output,
        classifier=VLMImageClassifier(),
        vlm_client=VLMClient(),
        force=args.force,
    )
    print(json.dumps(result["statistics"], ensure_ascii=False, indent=2))
    print(f"输出文件：{args.output.resolve()}")
    if result["statistics"]["failed"]:
        raise SystemExit("部分图片分类失败；修复服务或路径后直接重跑即可续跑失败项")


if __name__ == "__main__":
    main()
