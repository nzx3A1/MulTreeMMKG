"""按 Chunk 对照文本抽取 Graph 与参考标注，计算实体和关系的严格及表面规范化 F1。"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLES = PROJECT_ROOT / "eval/dataset/dev/chunk/text_chunk.json"
DEFAULT_REFERENCE = PROJECT_ROOT / "eval/dataset/dev/chunk/text_chunk_result.json"
DEFAULT_PREDICTION = PROJECT_ROOT / "eval/抽取评测/chunk抽取/text_extraction.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "text_f1_report.json"


def read_json(path: Path) -> Any:
    """读取 UTF-8 JSON 并保留原始结构供后续严格校验。"""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def source_info(path: Path) -> dict[str, str]:
    """记录输入文件的仓库相对路径和 SHA-256，便于复现本次评分。"""
    resolved = path.resolve()
    try:
        location = resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        location = str(resolved)
    return {"path": location, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def strict_name(value: str) -> str:
    """严格匹配只去除字段两端空白，不改写大小写或内部字符。"""
    return value.strip()


def normalized_name(value: str) -> str:
    """辅助匹配统一 Unicode 宽窄字符、英文字母大小写和全部空白。"""
    return "".join(char for char in unicodedata.normalize("NFKC", value).casefold() if not char.isspace())


def require_name(value: Any, label: str) -> str:
    """校验实体名或关系字段为非空字符串。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 必须是非空字符串")
    return value


def index_records(records: Any, id_field: str, label: str) -> dict[str, dict]:
    """按样本 ID 建立索引，拒绝缺失、重复或非对象记录。"""
    if not isinstance(records, list):
        raise ValueError(f"{label} 必须是 JSON 数组")
    indexed: dict[str, dict] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{label} 第 {index} 条必须是对象")
        sample_id = require_name(record.get(id_field), f"{label} 第 {index} 条的 {id_field}")
        if sample_id in indexed:
            raise ValueError(f"{label} 中存在重复样本 ID：{sample_id}")
        indexed[sample_id] = record
    return indexed


def reference_items(record: dict, normalize: Callable[[str], str]) -> tuple[set[str], set[tuple[str, str, str]]]:
    """读取名称列表与有方向的中文关系三元组。"""
    if record.get("modality") != "text":
        raise ValueError(f"参考模态不是 text：{record.get('sample_id')}")
    entities = record.get("entities")
    relations = record.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, list):
        raise ValueError(f"参考实体和关系必须是列表：{record['sample_id']}")
    names = {
        normalize(require_name(name, f"{record['sample_id']} 参考实体"))
        for name in entities
    }
    triples: set[tuple[str, str, str]] = set()
    for relation in relations:
        if not isinstance(relation, list) or len(relation) != 3:
            raise ValueError(f"参考关系必须是三元组：{record['sample_id']}")
        head, predicate, tail = (
            normalize(require_name(value, f"{record['sample_id']} 参考关系字段"))
            for value in relation
        )
        triples.add((head, predicate, tail))
    return names, triples


def predicted_items(graph: dict, normalize: Callable[[str], str]) -> tuple[set[str], set[tuple[str, str, str]]]:
    """通过关系端点 ID 读取预测实体名称和关系标签。"""
    metadata = graph.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("modality") != "text":
        raise ValueError("预测 Graph 缺少 text 模态元数据")
    entities = graph.get("entities")
    relations = graph.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, list):
        raise ValueError(f"预测 Graph 实体或关系不是列表：{metadata.get('chunk_id')}")
    by_id: dict[str, str] = {}
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError(f"预测实体不是对象：{metadata.get('chunk_id')}")
        entity_id = require_name(entity.get("id"), "预测实体 ID")
        name = normalize(require_name(entity.get("name"), "预测实体名称"))
        if entity_id in by_id and by_id[entity_id] != name:
            raise ValueError(f"预测实体 ID 对应不同名称：{entity_id}")
        by_id[entity_id] = name
    triples: set[tuple[str, str, str]] = set()
    for relation in relations:
        if not isinstance(relation, dict):
            raise ValueError(f"预测关系不是对象：{metadata.get('chunk_id')}")
        source_id = require_name(relation.get("source_id"), "关系头实体 ID")
        target_id = require_name(relation.get("target_id"), "关系尾实体 ID")
        if source_id not in by_id or target_id not in by_id:
            raise ValueError(f"关系端点未包含在同一 Graph：{metadata.get('chunk_id')}")
        label = relation.get("relation_name") or relation.get("type_zh") or relation.get("type")
        triples.add((by_id[source_id], normalize(require_name(label, "预测关系名称")), by_id[target_id]))
    return set(by_id.values()), triples


def metrics(tp: int, fp: int, fn: int) -> dict[str, int | float | None]:
    """根据 TP、FP、FN 计算 P/R/F1；双方皆空时指标记为 null。"""
    if tp == fp == fn == 0:
        precision = recall = f1 = None
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def compare(reference: set, prediction: set) -> dict[str, int | float | None]:
    """对一个 Chunk 的集合做一对一精确匹配并返回混淆计数。"""
    return metrics(
        len(reference & prediction),
        len(prediction - reference),
        len(reference - prediction),
    )


def evaluate(samples_path: Path, reference_path: Path, prediction_path: Path) -> dict:
    """检查三份文件的样本覆盖，并逐 Chunk 汇总严格和规范化 Micro-F1。"""
    samples = index_records(read_json(samples_path), "sample_id", "Chunk 输入")
    references = index_records(read_json(reference_path), "sample_id", "参考标注")
    prediction_file = read_json(prediction_path)
    if not isinstance(prediction_file, dict) or prediction_file.get("_status") != "completed":
        raise ValueError("文本抽取结果未标记 completed，拒绝计算 F1")
    if prediction_file.get("pending_sample_ids"):
        raise ValueError("文本抽取结果仍有待处理样本")
    graphs = prediction_file.get("graphs")
    if not isinstance(graphs, list):
        raise ValueError("文本抽取结果缺少 graphs 数组")
    predictions = index_records(
        [
            {**graph, "sample_id": graph.get("metadata", {}).get("chunk_id")}
            for graph in graphs
            if isinstance(graph, dict)
        ],
        "sample_id",
        "预测 Graph",
    )
    if len(predictions) != len(graphs):
        raise ValueError("预测 graphs 中存在非对象记录")
    expected = set(samples)
    for label, indexed in (("参考标注", references), ("预测 Graph", predictions)):
        if set(indexed) != expected:
            missing = sorted(expected - set(indexed))
            extra = sorted(set(indexed) - expected)
            raise ValueError(f"{label} 的样本 ID 与输入不一致：缺少={missing}，多出={extra}")
    if any(record.get("modality", "text") != "text" for record in samples.values()):
        raise ValueError("Chunk 输入含非文本模态")
    if prediction_file.get("statistics", {}).get("model_error_count", 0):
        raise ValueError("文本抽取结果存在未解决的模型请求错误")
    if prediction_file.get("statistics", {}).get("invalid_response_count", 0):
        raise ValueError("文本抽取结果存在未解决的模型响应格式错误")

    totals = {
        mode: {
            kind: {"tp": 0, "fp": 0, "fn": 0}
            for kind in ("entities", "relations")
        }
        for mode in ("strict", "normalized")
    }
    per_chunk = []
    duplicate_predicted_names = 0
    duplicate_predicted_ids = 0
    rejected_candidates = 0
    for sample_id in samples:
        reference = references[sample_id]
        graph = predictions[sample_id]
        metadata = graph["metadata"]
        if metadata.get("document_id") != sample_id.split("::", 1)[0]:
            raise ValueError(f"预测 Graph 文档 ID 与样本 ID 不一致：{sample_id}")
        raw_names = [entity.get("name") for entity in graph["entities"]]
        duplicate_predicted_names += len(raw_names) - len(set(raw_names))
        raw_ids = [entity.get("id") for entity in graph["entities"]]
        duplicate_predicted_ids += len(raw_ids) - len(set(raw_ids))
        rejected_candidates += int(metadata.get("extra", {}).get("validation", {}).get("rejected_count", 0))
        row: dict[str, Any] = {"sample_id": sample_id}
        for mode, normalize in (("strict", strict_name), ("normalized", normalized_name)):
            ref_entities, ref_relations = reference_items(reference, normalize)
            pred_entities, pred_relations = predicted_items(graph, normalize)
            row[mode] = {
                "entities": compare(ref_entities, pred_entities),
                "relations": compare(ref_relations, pred_relations),
            }
            for kind in ("entities", "relations"):
                for key in ("tp", "fp", "fn"):
                    totals[mode][kind][key] += row[mode][kind][key]
            if mode == "strict":
                row["strict_differences"] = {
                    "entity_fp": sorted(pred_entities - ref_entities),
                    "entity_fn": sorted(ref_entities - pred_entities),
                    "relation_fp": [list(item) for item in sorted(pred_relations - ref_relations)],
                    "relation_fn": [list(item) for item in sorted(ref_relations - pred_relations)],
                }
        per_chunk.append(row)
    summary = {
        mode: {kind: metrics(**totals[mode][kind]) for kind in ("entities", "relations")}
        for mode in ("strict", "normalized")
    }
    return {
        "_status": "completed",
        "reference_status": "unverified_reference",
        "inputs": {
            "samples": source_info(samples_path),
            "reference": source_info(reference_path),
            "prediction": source_info(prediction_path),
        },
        "matching": {
            "unit": "within_same_chunk",
            "entity": "name_only",
            "relation": "directed_head_name_relation_name_tail_name",
            "strict": "trim_outer_whitespace_only",
            "normalized": "Unicode_NFKC_casefold_remove_all_whitespace",
            "duplicates": "deduplicate_identical_names_or_triples_within_each_chunk",
            "both_empty": "precision_recall_f1_null",
        },
        "summary": {
            "chunk_count": len(samples),
            "reference_entity_count": sum(len(item["entities"]) for item in references.values()),
            "reference_relation_count": sum(len(item["relations"]) for item in references.values()),
            "prediction_entity_count": sum(len(item["entities"]) for item in predictions.values()),
            "prediction_relation_count": sum(len(item["relations"]) for item in predictions.values()),
            "duplicate_predicted_entity_names": duplicate_predicted_names,
            "duplicate_predicted_entity_ids": duplicate_predicted_ids,
            "prediction_validation_rejected_candidates": rejected_candidates,
            **summary,
        },
        "limitations": [
            "参考文件无标注者及人工复核信息，分数仅表示与当前参考标注的一致程度，不代表专家金标准准确率。",
            "参考标注仅含实体名称和中文关系三元组，无法计算实体类型、属性、证据或语义同义关系的 F1。",
            "关系标签按 relation_name 优先、type_zh 和 type 依次回退；谓词同义词不会自动合并。",
        ],
        "per_chunk": per_chunk,
    }


def main() -> int:
    """读取命令行路径、计算 F1 并保存可追溯的 JSON 报告。"""
    parser = argparse.ArgumentParser(description="计算 dev 文本 Chunk 的实体与关系 F1")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate(args.samples, args.reference, args.prediction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for mode in ("strict", "normalized"):
        entity = report["summary"][mode]["entities"]
        relation = report["summary"][mode]["relations"]
        print(f"{mode}: 实体 P/R/F1={entity['precision']:.4f}/{entity['recall']:.4f}/{entity['f1']:.4f}；"
              f"关系 P/R/F1={relation['precision']:.4f}/{relation['recall']:.4f}/{relation['f1']:.4f}")
    print(f"报告：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
