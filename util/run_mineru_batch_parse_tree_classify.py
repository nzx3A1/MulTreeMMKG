"""批量解析 MinerU Markdown、构建文档树并汇总逐图片分类结果。

本脚本面向 ``data/mineru_output`` 中多篇 MinerU ``full.md``，直接复用项目的
AMarkdownParser、Document Tree Builder 和 VLMImageClassifier。所有产物均写入单独
输出目录，并在每张图片完成后更新分类汇总文件，网络中断后可安全续跑。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


# 中文说明：确保从任意工作目录执行时均可导入仓库内模块。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.image_extractor.schema_models import ImageExtractionTask, as_string_tuple
from src.extractors.image_extractor.vlm_classification import TYPE_BY_CODE, VLMImageClassifier
from src.parser.AMarkdownParser import AMarkdownParser
from src.skeleton.document_tree_builder import build_document_tree, document_tree_to_dict
from src.utils.vlm_client import VLMClient


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "mineru_output"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "mineru_batch_20260907"


def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    """以临时文件替换方式写入 JSON，防止中断损坏可续跑分类检查点。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _iter_sections(sections: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    """按深度优先顺序遍历序列化后的章节字典。"""

    for section in sections:
        yield section
        children = section.get("children")
        if isinstance(children, list):
            yield from _iter_sections(item for item in children if isinstance(item, Mapping))


def _image_chunks_from_tree(tree: Mapping[str, Any]) -> list[dict[str, Any]]:
    """从单篇阶段二目录树提取图片 Chunk，并补充文档和章节溯源字段。"""

    document = tree.get("document")
    if not isinstance(document, Mapping):
        raise ValueError("阶段二目录树缺少 document 对象")
    document_id = str(document.get("id") or "").strip()
    if not document_id:
        raise ValueError("阶段二目录树缺少 document.id")

    chunks: list[dict[str, Any]] = []
    sections = document.get("sections")
    for section in _iter_sections(item for item in (sections or []) if isinstance(item, Mapping)):
        for raw_chunk in section.get("chunks") or []:
            if not isinstance(raw_chunk, Mapping) or str(raw_chunk.get("modality") or "").lower() != "image":
                continue
            source_chunk_id = str(raw_chunk.get("id") or "").strip()
            if not source_chunk_id:
                raise ValueError(f"文档 {document_id} 存在缺少 ID 的 ImageChunk")
            chunk = dict(raw_chunk)
            chunk.update(
                {
                    "id": f"{document_id}:section:{source_chunk_id}",
                    "source_chunk_id": source_chunk_id,
                    "document_id": document_id,
                    "section_id": str(section.get("id") or ""),
                    "section_title": str(section.get("title") or ""),
                    "image_path": list(as_string_tuple(raw_chunk.get("image_path"))),
                    "references": list(as_string_tuple(raw_chunk.get("references"))) or None,
                }
            )
            chunks.append(chunk)
    return chunks


def _base_image_record(chunk: Mapping[str, Any], image_path: str, image_index: int) -> dict[str, Any]:
    """将父 ImageChunk 中的一张物理图片转换为稳定的逐图分类记录。"""

    parent_chunk_id = str(chunk["id"])
    return {
        "image_id": f"{parent_chunk_id}:asset:{image_index}",
        "parent_chunk_id": parent_chunk_id,
        "image_index_in_chunk": image_index,
        "image_path": image_path,
        "caption": str(chunk.get("caption") or ""),
        "document_id": str(chunk.get("document_id") or ""),
        "section_id": str(chunk.get("section_id") or ""),
        "section_title": str(chunk.get("section_title") or ""),
        "source_chunk_id": str(chunk.get("source_chunk_id") or ""),
        "source_order": int(chunk.get("order") or 0),
        "source_references": list(as_string_tuple(chunk.get("references"))) or None,
    }


def _failed_classification(error: str) -> dict[str, Any]:
    """返回明确的失败占位，避免把模型或文件错误伪装成 A01-A20 结果。"""

    return {
        "primary_code": "UNCLASSIFIED",
        "primary_type": "分类失败，待重试",
        "secondary_codes": [],
        "secondary_types": [],
        "confidence": 0.0,
        "reason": error,
        "visual_evidence": [],
        "basis": "vlm_visual_classification_failed",
        "source": "vlm_visual_classification_failed",
    }


def _parent_classification(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """以父 Chunk 内已完成图片的多数主类别生成父级汇总类别。"""

    completed = [record for record in records if record.get("status") == "completed"]
    if not completed:
        return _failed_classification("父 Chunk 内没有成功的单图分类结果")
    counts = Counter(str(record["classification"]["primary_code"]) for record in completed)
    first_order = {
        str(record["classification"]["primary_code"]): index for index, record in enumerate(completed)
    }
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


def _build_classification_output(
    input_dir: Path,
    chunks: list[dict[str, Any]],
    records: list[dict[str, Any]],
    reused_count: int,
) -> dict[str, Any]:
    """生成全批次逐图、父 Chunk 和分类计数汇总，并回填父级类别。"""

    by_parent: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_parent.setdefault(str(record["parent_chunk_id"]), []).append(record)
    for chunk in chunks:
        parent = _parent_classification(by_parent.get(str(chunk["id"]), []))
        chunk["classification"] = parent
        for record in by_parent.get(str(chunk["id"]), []):
            classification = record.get("classification")
            if isinstance(classification, dict):
                classification["parent_primary_code"] = parent["primary_code"]
                classification["parent_primary_type"] = parent["primary_type"]
                classification["changed_from_parent_chunk"] = (
                    record.get("status") == "completed"
                    and classification.get("primary_code") != parent["primary_code"]
                )

    completed = [record for record in records if record.get("status") == "completed"]
    failed = [record for record in records if record.get("status") == "failed"]
    category_counts = Counter(str(record["classification"]["primary_code"]) for record in completed)
    per_document = Counter(str(record["document_id"]) for record in records)
    return {
        "_stage": "stage_02_individual_images_classified_all",
        "_description": "批量 MinerU 文档树的逐图片 A01-A20 VLM 分类与父 ImageChunk 汇总。",
        "_produced_by": "util/run_mineru_batch_parse_tree_classify.py",
        "source_directory": str(input_dir.resolve()),
        "taxonomy": [{"code": code, "name": name} for code, name in TYPE_BY_CODE.items()],
        "statistics": {
            "total_image_chunks": len(chunks),
            "total_images": len(records),
            "existing_image_files": sum(Path(str(record["image_path"])).is_file() for record in records),
            "completed": len(completed),
            "failed": len(failed),
            "reused": reused_count,
            "classification_counts": dict(sorted(category_counts.items())),
            "document_image_counts": dict(sorted(per_document.items())),
        },
        "chunks": chunks,
        "images": records,
    }


def _load_reusable_records(output_path: Path) -> dict[str, Mapping[str, Any]]:
    """读取上次的成功逐图结果，用 image_id 和原始路径保证续跑正确性。"""

    if not output_path.is_file():
        return {}
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    images = payload.get("images") if isinstance(payload, Mapping) else None
    if not isinstance(images, list):
        return {}
    reusable: dict[str, Mapping[str, Any]] = {}
    for record in images:
        if not isinstance(record, Mapping) or record.get("status") != "completed":
            continue
        classification = record.get("classification")
        image_id = str(record.get("image_id") or "")
        if image_id and isinstance(classification, Mapping) and classification.get("primary_code") in TYPE_BY_CODE:
            reusable[image_id] = record
    return reusable


def _never_merge_table_images(_: Path, __: float) -> bool:
    """批量运行时禁用可选 Paddle 表格续图检测，避免其环境故障中断 Markdown 主解析。"""

    return False


def _make_section_ids_unique(stage01: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    """复制阶段一结果并仅给重复章节 ID 添加后缀，使目录树索引可安全建立。"""

    copied = json.loads(json.dumps(stage01, ensure_ascii=False))
    seen_ids: Counter[str] = Counter()
    remapped_count = 0

    def visit(sections: Any) -> None:
        """深度优先处理当前层及其子层章节，维持原始章节 ID 的可追溯性。"""

        nonlocal remapped_count
        if not isinstance(sections, list):
            return
        for section in sections:
            if not isinstance(section, dict):
                continue
            original_id = str(section.get("id") or "section")
            seen_ids[original_id] += 1
            if seen_ids[original_id] > 1:
                section["source_section_id"] = original_id
                section["id"] = f"{original_id}__{seen_ids[original_id]}"
                remapped_count += 1
            visit(section.get("children"))

    visit(copied.get("toc"))
    return copied, remapped_count


def parse_and_build(
    input_dir: Path,
    output_dir: Path,
    *,
    use_table_image_detector: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """解析所有 Markdown，并逐篇调用目录树构建器生成阶段一、二聚合产物。"""

    markdown_files = sorted(input_dir.rglob("*.md"))
    if not markdown_files:
        raise FileNotFoundError(f"没有找到 Markdown 文件：{input_dir}")
    # 中文说明：图片分类只依赖普通图片 Chunk；默认关闭可选 Paddle 表格续图检测，
    # 防止未安装/不兼容的表格模型阻断所有论文的 Markdown 解析。
    parser = AMarkdownParser(
        table_image_detector=None if use_table_image_detector else _never_merge_table_images
    )
    stage01_documents = [parser.parse_file(path) for path in markdown_files]
    builder_inputs = [_make_section_ids_unique(document) for document in stage01_documents]
    stage02_trees = [document_tree_to_dict(build_document_tree(document)) for document, _ in builder_inputs]
    _write_json_atomic(
        output_dir / "stage_01_mineru_parse_all.json",
        {
            "_stage": 1,
            "_description": "批量 MinerU Markdown 解析结果。",
            "_produced_by": "src/parser/AMarkdownParser.py",
            "input_directory": str(input_dir.resolve()),
            "document_count": len(stage01_documents),
            "documents": stage01_documents,
        },
    )
    _write_json_atomic(
        output_dir / "stage_02_document_trees_all.json",
        {
            "_stage": 2,
            "_description": "批量 Document -> Section -> Chunk 目录树。",
            "_produced_by": "src/skeleton/document_tree_builder.py::build_document_tree",
            "source_stage_01": str((output_dir / "stage_01_mineru_parse_all.json").resolve()),
            "document_count": len(stage02_trees),
            "section_id_remapped_count": sum(count for _, count in builder_inputs),
            "section_id_remap_note": "仅构建阶段二树时为重复章节 ID 添加 __序号 后缀；原始阶段一结果保持不变。",
            "documents": stage02_trees,
        },
    )
    return stage01_documents, stage02_trees


def classify_images(
    input_dir: Path,
    trees: list[dict[str, Any]],
    output_path: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    """对全部图片执行真实 VLM 分类，并在每张完成后保存可续跑的总汇总。"""

    chunks = [chunk for tree in trees for chunk in _image_chunks_from_tree(tree)]
    if len({str(chunk["id"]) for chunk in chunks}) != len(chunks):
        raise ValueError("批量图片 Chunk ID 重复，无法保证分类溯源")
    records = [
        _base_image_record(chunk, image_path, image_index)
        for chunk in chunks
        for image_index, image_path in enumerate(chunk["image_path"])
    ]
    if len({str(record["image_id"]) for record in records}) != len(records):
        raise ValueError("批量图片 ID 重复，无法保证分类续跑")

    reusable = {} if force else _load_reusable_records(output_path)
    chunk_by_id = {str(chunk["id"]): chunk for chunk in chunks}
    classifier = VLMImageClassifier()
    vlm_client = VLMClient()
    reused_count = 0
    for index, record in enumerate(records, start=1):
        prior = reusable.get(str(record["image_id"]))
        if prior and str(prior.get("image_path")) == str(record["image_path"]):
            record["status"] = "completed"
            record["classification"] = dict(prior["classification"])
            reused_count += 1
            print(f"[{index}/{len(records)}] 复用：{record['image_id']}", flush=True)
            continue
        try:
            image_path = Path(str(record["image_path"]))
            if not image_path.is_file():
                raise FileNotFoundError(f"图片不存在：{image_path}")
            chunk = chunk_by_id[str(record["parent_chunk_id"])]
            task = ImageExtractionTask(
                document_id=str(chunk["document_id"]),
                chunk_id=str(chunk["id"]),
                image_id=str(record["image_id"]),
                image_index=int(record["image_index_in_chunk"]),
                image_path=str(record["image_path"]),
                caption=str(record.get("caption") or ""),
                references=tuple(as_string_tuple(record.get("source_references"))),
                section_id=str(record.get("section_id") or ""),
                section_title=str(record.get("section_title") or ""),
                raw_chunk=chunk,
            )
            result = classifier.classify(task, vlm_client)
            record["status"] = "completed"
            record["classification"] = result.to_dict()
            print(f"[{index}/{len(records)}] {result.primary_code} {result.primary_type}：{image_path.name}", flush=True)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            record["classification"] = _failed_classification(str(exc))
            print(f"[{index}/{len(records)}] 失败：{record['image_id']}：{exc}", flush=True)
        _write_json_atomic(output_path, _build_classification_output(input_dir, chunks, records, reused_count))

    result = _build_classification_output(input_dir, chunks, records, reused_count)
    _write_json_atomic(output_path, result)
    return result


def main() -> int:
    """解析 CLI 参数，顺序完成批量解析、建树、逐图分类和最终汇总。"""

    cli = argparse.ArgumentParser(description="批量解析 MinerU Markdown、构建目录树并分类全部图片")
    cli.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="包含 MinerU full.md 的根目录")
    cli.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="批量产物输出目录")
    cli.add_argument("--force-classify", action="store_true", help="忽略已有成功逐图结果并重新调用 VLM")
    cli.add_argument(
        "--use-table-image-detector",
        action="store_true",
        help="启用 AMarkdownParser 的 Paddle 表格续图检测；默认关闭以保持批量任务稳定",
    )
    args = cli.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    _, trees = parse_and_build(
        input_dir,
        output_dir,
        use_table_image_detector=args.use_table_image_detector,
    )
    result = classify_images(
        input_dir,
        trees,
        output_dir / "stage_02_individual_images_classified_all.json",
        force=args.force_classify,
    )
    print(json.dumps(result["statistics"], ensure_ascii=False, indent=2))
    print(f"输出目录：{output_dir}")
    return 0 if not result["statistics"]["failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
