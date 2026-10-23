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
DEFAULT_PREDICTION = PROJECT_ROOT / "eval/抽取评测/chunk抽取/LLM抽取/text_schema_extrction.json"
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


def reference_items(record: dict, normalize: Callable[[str], str]) -> tuple[set[str], set[tuple[str, str]]]:
    """读取参考实体名称和有方向的关系源、目标实体名称。"""
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
    pairs: set[tuple[str, str]] = set()
    for relation in relations:
        if not isinstance(relation, list) or len(relation) != 3:
            raise ValueError(f"参考关系必须是三元组：{record['sample_id']}")
        head = normalize(require_name(relation[0], f"{record['sample_id']} 参考关系源实体"))
        require_name(relation[1], f"{record['sample_id']} 参考关系类型")
        tail = normalize(require_name(relation[2], f"{record['sample_id']} 参考关系目标实体"))
        pairs.add((head, tail))
    return names, pairs


def collect_attribute_texts(value: Any) -> set[str]:
    """递归提取实体属性中的字符串值，忽略属性名和其他类型。"""
    if isinstance(value, str):
        return {value} if value.strip() else set()
    if isinstance(value, dict):
        return set().union(*(collect_attribute_texts(child) for child in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(collect_attribute_texts(child) for child in value)) if value else set()
    return set()


def predicted_items(
    graph: dict, normalize: Callable[[str], str]
) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    """读取预测实体名称、属性文本和有方向的关系端点名称。"""
    metadata = graph.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("modality") != "text":
        raise ValueError("预测 Graph 缺少 text 模态元数据")
    entities = graph.get("entities")
    relations = graph.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, list):
        raise ValueError(f"预测 Graph 实体或关系不是列表：{metadata.get('chunk_id')}")
    by_id: dict[str, str] = {}
    attribute_texts: set[str] = set()
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError(f"预测实体不是对象：{metadata.get('chunk_id')}")
        entity_id = require_name(entity.get("id"), "预测实体 ID")
        name = normalize(require_name(entity.get("name"), "预测实体名称"))
        if entity_id in by_id and by_id[entity_id] != name:
            raise ValueError(f"预测实体 ID 对应不同名称：{entity_id}")
        by_id[entity_id] = name
        for attribute_field in ("attribute", "attributes"):
            attribute_texts.update(collect_attribute_texts(entity.get(attribute_field)))
    pairs: set[tuple[str, str]] = set()
    for relation in relations:
        if not isinstance(relation, dict):
            raise ValueError(f"预测关系不是对象：{metadata.get('chunk_id')}")
        source_id = require_name(relation.get("source_id"), "关系头实体 ID")
        target_id = require_name(relation.get("target_id"), "关系尾实体 ID")
        if source_id not in by_id or target_id not in by_id:
            raise ValueError(f"关系端点未包含在同一 Graph：{metadata.get('chunk_id')}")
        pairs.add((by_id[source_id], by_id[target_id]))
    return set(by_id.values()), {normalize(text) for text in attribute_texts}, pairs


def metrics(tp: int, fp: int, fn: int, recall_tp: int | None = None) -> dict[str, int | float | None]:
    """计算 P/R/F1；实体召回可使用包含属性命中的独立 TP 数。"""
    recall_tp = tp if recall_tp is None else recall_tp
    if tp == fp == fn == recall_tp == 0:
        precision = recall = f1 = None
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = recall_tp / (recall_tp + fn) if recall_tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "recall_tp": recall_tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def compare(reference: set, prediction: set, recall_prediction: set | None = None) -> dict[str, int | float | None]:
    """比较集合并允许召回候选范围包含额外的属性命中实体。"""
    recall_prediction = prediction if recall_prediction is None else recall_prediction
    tp = len(reference & prediction)
    recall_tp = len(reference & recall_prediction)
    return metrics(
        tp,
        len(prediction - reference),
        len(reference - recall_prediction),
        recall_tp=recall_tp,
    )


def evaluate(samples_path: Path, reference_path: Path, prediction_path: Path) -> dict:
    """检查三份文件的样本覆盖，并逐 Chunk 汇总严格和规范化 Micro-F1。"""
    samples = index_records(read_json(samples_path), "sample_id", "Chunk 输入")
    references = index_records(read_json(reference_path), "sample_id", "参考标注")
    prediction_file = read_json(prediction_path)
    if not isinstance(prediction_file, dict) or prediction_file.get("_status") not in {"completed", "completed_with_warnings"}:
        raise ValueError("文本抽取结果未完成，拒绝计算 F1")
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
    if set(references) != expected:
        missing = sorted(expected - set(references))
        extra = sorted(set(references) - expected)
        raise ValueError(f"参考标注的样本 ID 与输入不一致：缺少={missing}，多出={extra}")
    extra_predictions = sorted(set(predictions) - expected)
    if extra_predictions:
        raise ValueError(f"预测 Graph 含输入中不存在的样本 ID：{extra_predictions}")
    missing_predictions = sorted(expected - set(predictions))
    if any(record.get("modality", "text") != "text" for record in samples.values()):
        raise ValueError("Chunk 输入含非文本模态")
    if prediction_file.get("statistics", {}).get("model_error_count", 0):
        raise ValueError("文本抽取结果存在未解决的模型请求错误")
    if prediction_file.get("statistics", {}).get("invalid_response_count", 0):
        raise ValueError("文本抽取结果存在未解决的模型响应格式错误")

    totals = {
        mode: {
            kind: {"tp": 0, "recall_tp": 0, "fp": 0, "fn": 0}
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
        graph = predictions.get(sample_id)
        if graph is None:
            metadata = {}
        else:
            metadata = graph["metadata"]
            if metadata.get("document_id") != sample_id.split("::", 1)[0]:
                raise ValueError(f"预测 Graph 文档 ID 与样本 ID 不一致：{sample_id}")
            raw_names = [entity.get("name") for entity in graph["entities"]]
            duplicate_predicted_names += len(raw_names) - len(set(raw_names))
            raw_ids = [entity.get("id") for entity in graph["entities"]]
            duplicate_predicted_ids += len(raw_ids) - len(set(raw_ids))
            rejected_candidates += int(metadata.get("extra", {}).get("validation", {}).get("rejected_count", 0))
        row: dict[str, Any] = {"sample_id": sample_id, "prediction_missing": graph is None}
        for mode, normalize in (("strict", strict_name), ("normalized", normalized_name)):
            ref_entities, ref_relations = reference_items(reference, normalize)
            if graph is None:
                pred_entities = attribute_texts = pred_relations = set()
            else:
                pred_entities, attribute_texts, pred_relations = predicted_items(graph, normalize)
            attribute_entity_hits = {
                name for name in ref_entities
                if any(name in attribute_text for attribute_text in attribute_texts)
            }
            recall_entities = pred_entities | attribute_entity_hits
            row[mode] = {
                "entities": compare(ref_entities, pred_entities, recall_prediction=recall_entities),
                "relations": compare(ref_relations, pred_relations),
                "entity_attribute_recall_hits": sorted(attribute_entity_hits - pred_entities),
            }
            for kind in ("entities", "relations"):
                for key in ("tp", "recall_tp", "fp", "fn"):
                    totals[mode][kind][key] += row[mode][kind][key]
            if mode == "strict":
                row["strict_differences"] = {
                    "entity_fp": sorted(pred_entities - ref_entities),
                    "entity_fn": sorted(ref_entities - recall_entities),
                    "entity_attribute_recall_hits": sorted(attribute_entity_hits - pred_entities),
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
            "entity_precision": "predicted_entity_name_matches_reference_name",
            "entity_recall": "predicted_entity_name_or_entity_attribute_value_contains_reference_name",
            "relation": "directed_source_entity_name_and_target_entity_name_only; predicate_ignored",
            "strict": "trim_outer_whitespace_only",
            "normalized": "Unicode_NFKC_casefold_remove_all_whitespace",
            "duplicates": "deduplicate_identical_entity_names_or_endpoint_pairs_within_each_chunk",
            "both_empty": "precision_recall_f1_null",
        },
        "summary": {
            "chunk_count": len(samples),
            "missing_prediction_chunk_count": len(missing_predictions),
            "missing_prediction_chunk_ids": missing_predictions,
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
            "实体属性文本只用于实体召回命中判断，不计算属性本身的准确率；属性值包含参考实体名称即算命中。",
            "关系只比较有方向的源、目标实体名称，忽略关系类型；实体端点名称仍按对应严格或规范化规则匹配。",
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
