"""构建 MulTreeMMKG 的可复现 MVP 评测数据骨架。

脚本固定 16 篇文档的开发/测试划分，复用现有 Markdown 解析与目录树语义，
生成候选 Chunk、120 个分层样本和四类待人工标注模板。脚本不调用模型，
也不会把候选结果写成金标。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.AMarkdownParser import AMarkdownParser
from src.skeleton.document_tree_builder import build_document_tree


MANIFEST_PATH = Path("eval/dataset_split/manifest.json")
MVP_DIR = Path("eval/mvp")
SPLITS = ("development", "test")
MODALITIES = ("text", "image", "table")
DEFAULT_SAMPLING_CONFIG: dict[str, Any] = {
    "dataset_name": "MulTreeMMKG-MVP-16",
    "sampling_version": "mvp-sampling-v1",
    "seed": "MulTreeMMKG-2026-09-20",
    "targets": {"text": 60, "image": 30, "table": 30},
    "split_targets": {
        "development": {"text": 15, "image": 8, "table": 6},
        "test": {"text": 45, "image": 22, "table": 24},
    },
}
QA_COUNTS = {
    "single_hop": 15,
    "cross_modal": 15,
    "multi_hop": 10,
    "table_numeric": 10,
    "synthesis": 5,
    "unanswerable": 5,
}
QA_DEV_COUNTS = {
    "single_hop": 4,
    "cross_modal": 4,
    "multi_hop": 2,
    "table_numeric": 2,
    "synthesis": 1,
    "unanswerable": 2,
}


def canonical_json_bytes(value: Any) -> bytes:
    """将对象编码为稳定 JSON 字节，用于指纹和内容哈希。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """计算字节串的 SHA-256 十六进制摘要。"""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """分块计算文件 SHA-256，避免把大 PDF 一次性读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    """读取 UTF-8 JSON 文件。"""

    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    """以稳定缩进和换行写入 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL，忽略空行并要求每行是对象。"""

    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} 必须是 JSON 对象")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """按键排序写入稳定 JSONL 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def repository_path(path: Path, project_root: Path) -> str:
    """把仓库内路径转为正斜杠相对路径。"""

    return path.resolve().relative_to(project_root.resolve()).as_posix()


def resolve_relative(project_root: Path, raw_path: str) -> Path:
    """解析并校验仓库相对路径，拒绝绝对路径和越界路径。"""

    pure = PurePosixPath(str(raw_path).replace("\\", "/"))
    if pure.is_absolute() or ":" in pure.parts[0]:
        raise ValueError(f"规范清单不允许绝对路径：{raw_path}")
    resolved = (project_root / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"路径越出仓库根目录：{raw_path}") from exc
    return resolved


def split_fingerprint(documents: Sequence[Mapping[str, Any]]) -> str:
    """依据有序文档 ID、划分和 PDF 哈希计算固定划分指纹。"""

    rows = [
        {
            "id": str(document["id"]),
            "split": str(document["split"]),
            "source_sha256": str(document["integrity"]["pdf_sha256"]),
        }
        for document in sorted(documents, key=lambda item: str(item["id"]))
    ]
    return sha256_bytes(canonical_json_bytes(rows))


def infer_mineru_dir(document: Mapping[str, Any]) -> str:
    """根据 PDF 相对路径推导对应 MinerU 输出目录。"""

    source = PurePosixPath(str(document["source_path"]).replace("\\", "/"))
    base = PurePosixPath("testData/mineru_output") if source.parts[0].casefold() == "testdata" else PurePosixPath("data/mineru_output")
    return (base / source.stem).as_posix()


def refresh_manifest(manifest: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    """把旧清单升级为只含相对路径和内容哈希的 16 文档规范清单。"""

    refreshed = deepcopy(dict(manifest))
    refreshed["name"] = "MulTreeMMKG-MVP-16"
    refreshed["manifest_format"] = "2.0"
    refreshed["version"] = "1.0"
    refreshed.pop("created_at", None)
    refreshed["split_policy"] = {
        "development_documents": 4,
        "test_documents": 12,
        "unit": "document",
        "test_usage": "Only evaluate after prompts, schema rules and fusion thresholds are frozen.",
    }
    documents: list[dict[str, Any]] = []
    for raw in refreshed.get("documents", []):
        document = deepcopy(dict(raw))
        document.pop("absolute_path", None)
        document["source_path"] = str(document["source_path"]).replace("\\", "/")
        document["mineru_dir"] = str(document.get("mineru_dir") or infer_mineru_dir(document))
        document["markdown_path"] = str(document.get("markdown_path") or f"{document['mineru_dir']}/full.md")
        document["parser_profile"] = "legacy" if document["source_path"].casefold().startswith("testdata/") else "current"

        mineru_dir = resolve_relative(project_root, document["mineru_dir"])
        content_lists = sorted(mineru_dir.glob("*_content_list_v2.json"))
        document["content_list_path"] = repository_path(content_lists[0], project_root) if content_lists else None
        pdf_path = resolve_relative(project_root, document["source_path"])
        markdown_path = resolve_relative(project_root, document["markdown_path"])
        if not pdf_path.is_file() or not markdown_path.is_file():
            raise FileNotFoundError(f"文档来源不完整：{document['id']}")
        document["integrity"] = {
            "pdf_bytes": pdf_path.stat().st_size,
            "pdf_sha256": sha256_file(pdf_path),
            "markdown_bytes": markdown_path.stat().st_size,
            "markdown_sha256": sha256_file(markdown_path),
        }
        if document["parser_profile"] == "legacy":
            blocks = dict(document.get("mineru_blocks") or {})
            blocks.setdefault("images", None)
            blocks.setdefault("charts", None)
            blocks.setdefault("tables", None)
            blocks.setdefault("equations", None)
            document["mineru_blocks"] = blocks
        documents.append(document)
    refreshed["documents"] = sorted(documents, key=lambda item: str(item["id"]))
    refreshed["split_fingerprint"] = split_fingerprint(refreshed["documents"])
    return refreshed


def validate_manifest(manifest: Mapping[str, Any], project_root: Path, *, verify_integrity: bool = True) -> list[dict[str, Any]]:
    """校验 16 文档、4/12 划分、相对路径和可选内容哈希。"""

    documents = [dict(item) for item in manifest.get("documents", [])]
    if len(documents) != 16:
        raise ValueError(f"清单必须包含 16 篇文档，实际为 {len(documents)}")
    ids = [str(item.get("id") or "") for item in documents]
    if len(set(ids)) != len(ids):
        raise ValueError("清单存在重复 document_id")
    expected_ids = [f"GEO-{index:03d}" for index in range(1, 17)]
    if sorted(ids) != expected_ids:
        raise ValueError("文档 ID 必须完整覆盖 GEO-001 至 GEO-016")
    counts = Counter(str(item.get("split")) for item in documents)
    if counts != Counter({"development": 4, "test": 12}):
        raise ValueError(f"划分必须为 4 development / 12 test，实际为 {dict(counts)}")

    source_paths: set[str] = set()
    for document in documents:
        for key in ("source_path", "mineru_dir", "markdown_path"):
            if not document.get(key):
                raise ValueError(f"{document['id']} 缺少 {key}")
            resolve_relative(project_root, str(document[key]))
        source = str(document["source_path"])
        if source in source_paths:
            raise ValueError(f"PDF 重复出现在清单中：{source}")
        source_paths.add(source)
        pdf_path = resolve_relative(project_root, source)
        markdown_path = resolve_relative(project_root, str(document["markdown_path"]))
        if not pdf_path.is_file() or not markdown_path.is_file():
            raise FileNotFoundError(f"{document['id']} 的 PDF 或 Markdown 不存在")
        if verify_integrity:
            integrity = document.get("integrity") or {}
            if sha256_file(pdf_path) != integrity.get("pdf_sha256"):
                raise ValueError(f"{document['id']} PDF 哈希不匹配")
            if sha256_file(markdown_path) != integrity.get("markdown_sha256"):
                raise ValueError(f"{document['id']} Markdown 哈希不匹配")
    if verify_integrity and manifest.get("split_fingerprint") != split_fingerprint(documents):
        raise ValueError("split_fingerprint 与文档划分或 PDF 哈希不一致")
    return sorted(documents, key=lambda item: str(item["id"]))


def iter_raw_sections(sections: Sequence[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    """按深度优先顺序遍历阶段一章节字典。"""

    for section in sections:
        yield dict(section)
        yield from iter_raw_sections(section.get("children") or [])


def ensure_unique_section_ids(stage01: dict[str, Any]) -> None:
    """在评测适配层稳定消解重复章节 ID，不改变生产解析器。"""

    counts: Counter[str] = Counter()

    def visit(sections: Sequence[dict[str, Any]]) -> None:
        """递归重写重复 ID，并继续处理子章节。"""

        for section in sections:
            base = str(section.get("id") or "section")
            counts[base] += 1
            if counts[base] > 1:
                section["id"] = f"{base}~{counts[base]}"
            visit(section.get("children") or [])

    visit(stage01.get("toc") or [])


def extract_asset_paths(raw: Mapping[str, Any]) -> list[str]:
    """从阶段一图表条目中提取可能存在的图片资源路径。"""

    values: list[Any] = []
    for key in ("path", "table_path"):
        value = raw.get(key)
        values.extend(value if isinstance(value, list) else [value] if value else [])
    content = raw.get("content")
    if isinstance(content, str) and Path(content).suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp"}:
        values.append(content)
    return [str(value) for value in values if value]


def raw_table_assets(stage01: Mapping[str, Any]) -> dict[str, list[str]]:
    """按本地 TableChunk ID 保存被目录树模型忽略的表格图片路径。"""

    assets: dict[str, list[str]] = {}
    for section in iter_raw_sections(stage01.get("toc") or []):
        section_id = str(section.get("id") or "section")
        tables = section.get("table") or section.get("tables") or []
        if isinstance(tables, Mapping) or isinstance(tables, str):
            tables = [tables]
        for index, table in enumerate(tables):
            row = table if isinstance(table, Mapping) else {"content": table}
            assets[f"{section_id}:table:{index}"] = extract_asset_paths(row)
    return assets


def flatten_content_list(payload: Any) -> Iterable[tuple[int | None, Mapping[str, Any]]]:
    """兼容按页数组和扁平数组两类 MinerU content_list 结构。"""

    if not isinstance(payload, list):
        return
    if payload and all(isinstance(item, list) for item in payload):
        for page_idx, page in enumerate(payload):
            for item in page:
                if isinstance(item, Mapping):
                    yield page_idx, item
        return
    for item in payload:
        if isinstance(item, Mapping):
            page_idx = item.get("page_idx")
            yield int(page_idx) if isinstance(page_idx, int) else None, item


def load_visual_provenance(document: Mapping[str, Any], project_root: Path) -> dict[str, dict[str, Any]]:
    """从 MinerU content_list 建立资源路径到页码、坐标框的索引。"""

    raw_path = document.get("content_list_path")
    if not raw_path:
        return {}
    content_list_path = resolve_relative(project_root, str(raw_path))
    mineru_dir = resolve_relative(project_root, str(document["mineru_dir"]))
    index: dict[str, dict[str, Any]] = {}
    for inferred_page, item in flatten_content_list(read_json(content_list_path)):
        content = item.get("content") if isinstance(item.get("content"), Mapping) else {}
        source = content.get("image_source") if isinstance(content.get("image_source"), Mapping) else {}
        asset = source.get("path") or item.get("img_path") or item.get("image_path")
        if not asset:
            continue
        asset_path = Path(str(asset))
        absolute = asset_path if asset_path.is_absolute() else mineru_dir / asset_path
        try:
            relative = repository_path(absolute, project_root)
        except ValueError:
            continue
        page_idx = item.get("page_idx", inferred_page)
        row = {
            "page_idx": int(page_idx) if isinstance(page_idx, int) else None,
            "bbox": list(item.get("bbox")) if isinstance(item.get("bbox"), list) else None,
            "mineru_type": str(item.get("type") or "unknown"),
        }
        index[relative.casefold()] = row
        index[Path(relative).name.casefold()] = row
    return index


def normalize_asset_path(raw_path: str, project_root: Path, markdown_path: Path) -> tuple[str, Path | None]:
    """把 Chunk 中的资源路径规范为仓库相对路径，并保留不存在状态。"""

    candidate = Path(raw_path)
    absolute = candidate if candidate.is_absolute() else markdown_path.parent / candidate
    absolute = absolute.resolve()
    try:
        return repository_path(absolute, project_root), absolute
    except ValueError:
        return str(raw_path).replace("\\", "/"), None


def build_asset_records(paths: Sequence[str], project_root: Path, markdown_path: Path) -> list[dict[str, Any]]:
    """生成资源相对路径、存在性、大小和哈希记录。"""

    records: list[dict[str, Any]] = []
    for raw_path in paths:
        relative, absolute = normalize_asset_path(raw_path, project_root, markdown_path)
        exists = bool(absolute and absolute.is_file())
        records.append({
            "path": relative,
            "exists": exists,
            "bytes": absolute.stat().st_size if exists and absolute else None,
            "sha256": sha256_file(absolute) if exists and absolute else None,
        })
    return records


def build_document_chunks(document_meta: Mapping[str, Any], project_root: Path) -> list[dict[str, Any]]:
    """解析一篇 Markdown，输出带全局 ID 和可核验溯源的三模态 Chunk。"""

    markdown_path = resolve_relative(project_root, str(document_meta["markdown_path"]))
    parser = AMarkdownParser(use_llm_basic_info=False, table_image_detector=lambda *_args: False)
    stage01 = parser.parse_file(markdown_path)
    stage01["document_id"] = str(document_meta["id"])
    ensure_unique_section_ids(stage01)
    table_assets = raw_table_assets(stage01)
    document = build_document_tree(stage01)
    visual_index = load_visual_provenance(document_meta, project_root)

    rows: list[dict[str, Any]] = []
    document_order = 0
    for section in document.iter_sections():
        for chunk in section.chunks:
            modality = str(chunk.modality.value)
            if modality not in MODALITIES:
                continue
            local_id = str(chunk.id)
            if modality == "text":
                content = str(chunk.text)
                caption = None
                raw_assets: list[str] = []
            elif modality == "image":
                content = str(chunk.caption or "")
                caption = chunk.caption
                raw_assets = [str(item) for item in chunk.image_path]
            else:
                content = str(chunk.markdown)
                caption = chunk.caption
                raw_assets = table_assets.get(local_id, []) or [str(item) for item in chunk.table_path]
            assets = build_asset_records(raw_assets, project_root, markdown_path)
            visual = next(
                (
                    visual_index.get(str(asset["path"]).casefold())
                    or visual_index.get(Path(str(asset["path"])).name.casefold())
                    for asset in assets
                    if visual_index.get(str(asset["path"]).casefold())
                    or visual_index.get(Path(str(asset["path"])).name.casefold())
                ),
                None,
            )
            content_payload = {"content": content, "caption": caption, "assets": assets}
            global_id = f"{document_meta['id']}::{local_id}"
            rows.append({
                "global_chunk_id": global_id,
                "document_id": str(document_meta["id"]),
                "split": str(document_meta["split"]),
                "chunk_id": local_id,
                "section_id": str(section.id),
                "section_title": section.title,
                "modality": modality,
                "document_order": document_order,
                "section_order": int(section.order),
                "chunk_order": int(chunk.order),
                "content": content,
                "caption": caption,
                "assets": assets,
                "content_sha256": sha256_bytes(canonical_json_bytes(content_payload)),
                "provenance": {
                    "markdown_path": str(document_meta["markdown_path"]),
                    "content_list_path": document_meta.get("content_list_path"),
                    "page_idx": visual.get("page_idx") if visual else None,
                    "bbox": visual.get("bbox") if visual else None,
                    "mineru_type": visual.get("mineru_type") if visual else None,
                    "status": "mineru_page_bbox" if visual else "section_markdown_only",
                },
            })
            document_order += 1
    return rows


def build_chunk_manifest(documents: Sequence[Mapping[str, Any]], project_root: Path) -> list[dict[str, Any]]:
    """构建整个语料的 Chunk 清单并校验全局 ID 唯一性。"""

    rows: list[dict[str, Any]] = []
    for document in documents:
        rows.extend(build_document_chunks(document, project_root))
    global_ids = [str(row["global_chunk_id"]) for row in rows]
    if len(global_ids) != len(set(global_ids)):
        duplicates = [key for key, count in Counter(global_ids).items() if count > 1]
        raise ValueError(f"全局 Chunk ID 重复：{duplicates[:5]}")
    return rows


def stable_candidate_rank(row: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    """按版本、种子和 Chunk 内容生成平台无关的采样排序键。"""

    token = "|".join([
        str(config["sampling_version"]),
        str(config["seed"]),
        str(row["document_id"]),
        str(row["modality"]),
        str(row["global_chunk_id"]),
        str(row["content_sha256"]),
    ])
    return sha256_bytes(token.encode("utf-8"))


def select_stratified_samples(chunks: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """按固定配额和稳定排序抽取开发/测试三模态样本。"""

    selected: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        for modality in MODALITIES:
            candidates = [
                dict(row) for row in chunks
                if row.get("split") == split and row.get("modality") == modality
            ]
            target = int(config["split_targets"][split][modality])
            if len(candidates) < target:
                raise ValueError(
                    f"{split}/{modality} 候选不足：需要 {target}，实际 {len(candidates)}；"
                    "请调整明确配额，不能用其他模态静默回填。"
                )
            for row in candidates:
                row["_rank"] = stable_candidate_rank(row, config)
            by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in candidates:
                by_document[str(row["document_id"])].append(row)
            for values in by_document.values():
                values.sort(key=lambda item: (item["_rank"], item["global_chunk_id"]))

            chosen: list[dict[str, Any]] = []
            chosen_ids: set[str] = set()
            for document_id in sorted(by_document):
                if len(chosen) >= target:
                    break
                row = by_document[document_id][0]
                chosen.append(row)
                chosen_ids.add(str(row["global_chunk_id"]))
            remainder = sorted(candidates, key=lambda item: (item["_rank"], item["global_chunk_id"]))
            for row in remainder:
                if len(chosen) >= target:
                    break
                if str(row["global_chunk_id"]) not in chosen_ids:
                    chosen.append(row)
                    chosen_ids.add(str(row["global_chunk_id"]))

            for selection_rank, row in enumerate(chosen, start=1):
                sample = {key: value for key, value in row.items() if key != "_rank"}
                sample.update({
                    "sample_id": f"{split}-{modality}-{selection_rank:03d}",
                    "selection_rank": selection_rank,
                    "candidate_population": len(candidates),
                    "sampling_version": config["sampling_version"],
                    "sampling_seed": config["seed"],
                })
                selected[split].append(sample)
    return selected


def sample_set_fingerprint(selected: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    """对有序样本 ID、Chunk ID 和内容哈希计算样本集指纹。"""

    rows = [
        {
            "sample_id": row["sample_id"],
            "global_chunk_id": row["global_chunk_id"],
            "content_sha256": row["content_sha256"],
        }
        for split in SPLITS
        for row in selected[split]
    ]
    return sha256_bytes(canonical_json_bytes(rows))


def pending_extraction_rows(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """为每个抽样 Chunk 生成空的待人工抽取标注记录。"""

    return [{
        "record_id": f"extract::{sample['global_chunk_id']}",
        "annotation_status": "PENDING",
        "document_id": sample["document_id"],
        "split": sample["split"],
        "global_chunk_id": sample["global_chunk_id"],
        "chunk_id": sample["chunk_id"],
        "modality": sample["modality"],
        "entities": [],
        "relations": [],
        "attributes": [],
        "annotator": None,
        "reviewer": None,
        "notes": None,
    } for sample in samples]


def pending_fusion_rows(split: str) -> list[dict[str, Any]]:
    """生成不包含虚构实体对的融合标注占位记录。"""

    return [{
        "record_id": f"fusion-slot::{split}::001",
        "annotation_status": "PENDING",
        "split": split,
        "left_mention": None,
        "right_mention": None,
        "label": None,
        "is_hard_negative": None,
        "evidence_chunk_ids": [],
        "annotator": None,
        "reviewer": None,
        "notes": "实体提及产生后再创建具体候选对；本记录不能计入金标。",
    }]


def pending_final_triple_rows(split: str, count: int) -> list[dict[str, Any]]:
    """生成端点为空的最终三元组人工核验槽位。"""

    return [{
        "record_id": f"triple-review::{split}::{index:03d}",
        "annotation_status": "PENDING",
        "split": split,
        "triple": None,
        "review_score": None,
        "schema_legal": None,
        "evidence": [],
        "conflicts_preserved": None,
        "annotator": None,
        "reviewer": None,
        "notes": None,
    } for index in range(1, count + 1)]


def pending_qa_rows(split: str, counts: Mapping[str, int]) -> list[dict[str, Any]]:
    """按指定题型数量生成待人工编写的问答槽位。"""

    rows: list[dict[str, Any]] = []
    for category, count in counts.items():
        for index in range(1, int(count) + 1):
            rows.append({
                "record_id": f"qa::{split}::{category}::{index:03d}",
                "annotation_status": "PENDING",
                "split": split,
                "category": category,
                "question": None,
                "canonical_answer": None,
                "accepted_aliases": [],
                "answerable": None,
                "supporting_chunk_ids": [],
                "supporting_entity_ids": [],
                "supporting_relation_ids": [],
                "minimal_evidence_path": [],
                "required_modalities": [],
                "annotator": None,
                "reviewer": None,
                "notes": None,
            })
    return rows


def write_pending_annotations(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """仅覆盖纯 PENDING 模板，避免重建时破坏已完成人工标注。"""

    existing = read_jsonl(path)
    if any(row.get("annotation_status") != "PENDING" for row in existing):
        print(f"保留已有人工作业的标注文件：{path}")
        return
    write_jsonl(path, rows)


def build_annotations(selected: Mapping[str, Sequence[Mapping[str, Any]]], project_root: Path) -> None:
    """为开发集和测试集生成四层待人工标注文件。"""

    for split in SPLITS:
        directory = project_root / MVP_DIR / "annotations" / split
        write_pending_annotations(directory / "extraction.jsonl", pending_extraction_rows(selected[split]))
        write_pending_annotations(directory / "fusion.jsonl", pending_fusion_rows(split))
        triple_count = 25 if split == "development" else 75
        write_pending_annotations(directory / "final_triples.jsonl", pending_final_triple_rows(split, triple_count))
        qa_counts = QA_DEV_COUNTS if split == "development" else {
            category: QA_COUNTS[category] - QA_DEV_COUNTS[category] for category in QA_COUNTS
        }
        write_pending_annotations(directory / "qa.jsonl", pending_qa_rows(split, qa_counts))


def build_dataset(project_root: Path = PROJECT_ROOT, *, refresh: bool = False) -> dict[str, Any]:
    """执行清单校验、Chunk 构建、稳定采样和标注模板生成。"""

    manifest_path = project_root / MANIFEST_PATH
    manifest = read_json(manifest_path)
    if refresh:
        manifest = refresh_manifest(manifest, project_root)
        write_json(manifest_path, manifest)
    documents = validate_manifest(manifest, project_root)
    chunks = build_chunk_manifest(documents, project_root)

    config_path = project_root / MVP_DIR / "sampling_config.json"
    config = read_json(config_path) if config_path.exists() else deepcopy(DEFAULT_SAMPLING_CONFIG)
    selected = select_stratified_samples(chunks, config)
    config = deepcopy(dict(config))
    config["candidate_counts"] = dict(Counter(str(row["modality"]) for row in chunks))
    config["sample_set_fingerprint"] = sample_set_fingerprint(selected)

    write_jsonl(project_root / MVP_DIR / "chunks.jsonl", chunks)
    write_json(config_path, config)
    for split in SPLITS:
        write_jsonl(project_root / MVP_DIR / "samples" / f"{split}.jsonl", selected[split])
    build_annotations(selected, project_root)
    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "samples": sum(len(rows) for rows in selected.values()),
        "modalities": dict(Counter(row["modality"] for rows in selected.values() for row in rows)),
        "split_fingerprint": manifest["split_fingerprint"],
        "sample_set_fingerprint": config["sample_set_fingerprint"],
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="构建 MulTreeMMKG MVP 评测数据集")
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="首次迁移时刷新相对路径、文件哈希和划分指纹",
    )
    return parser.parse_args()


def main() -> None:
    """运行构建流程并输出精简统计。"""

    args = parse_args()
    summary = build_dataset(refresh=args.refresh_manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
