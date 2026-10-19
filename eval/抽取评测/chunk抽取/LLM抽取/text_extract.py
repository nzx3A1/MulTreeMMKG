"""逐文本 Chunk 使用提示词和项目 LLM 抽取实体、关系，并保存统一 Graph 结果。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.model_config import settings
from model.base import SourceModality
from model.graph import Entity, Graph, Relation
from src.utils.llm_client import LLMClient, safe_json_loads
from src.utils.vlm_client import VLMClient

DATA_DIR = PROJECT_ROOT / "eval/dataset/dev/chunk"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def read_samples(path: Path, modality: str) -> list[dict[str, Any]]:
    """读取并校验指定模态的 Chunk 列表及样本唯一性。"""
    samples = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(samples, list):
        raise ValueError(f"{path} 必须是 JSON 数组")
    seen: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Chunk 必须是 JSON 对象")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or "::" not in sample_id or sample_id in seen:
            raise ValueError(f"Chunk ID 无效或重复：{sample_id}")
        seen.add(sample_id)
        if not isinstance(sample.get("section"), dict) or not isinstance(sample.get("content"), dict):
            raise ValueError(f"Chunk 缺少 section 或 content：{sample_id}")
        content = sample["content"]
        key = "text" if modality == "text" else "image_path" if modality == "image" else "markdown"
        value = content.get(key)
        if modality == "image":
            valid = isinstance(value, list) and bool(value) and all(isinstance(item, str) and item.strip() for item in value)
        else:
            valid = isinstance(value, str) and bool(value.strip())
        if not valid:
            raise ValueError(f"Chunk 缺少有效 {key}：{sample_id}")
    return samples


def resolve_image(value: str, input_path: Path) -> Path:
    """解析图片绝对路径或相对于输入文件的路径，并检查文件存在。"""
    image = Path(value)
    if not image.is_absolute():
        image = input_path.parent / image
    image = image.resolve()
    if not image.is_file():
        raise FileNotFoundError(f"图片不存在：{image}")
    return image


def image_paths(sample: dict[str, Any], modality: str, input_path: Path) -> list[Path]:
    """确定图像或表格 Chunk 中需要直接交给视觉模型的图片。"""
    content = sample["content"]
    if modality == "image":
        values = content["image_path"]
    elif modality == "table":
        markdown = content["markdown"].strip()
        values = [markdown] if "\n" not in markdown and Path(markdown).suffix.lower() in IMAGE_SUFFIXES else []
        values += content.get("table_path") or []
    else:
        values = []
    return list(dict.fromkeys(resolve_image(value, input_path) for value in values))


def make_prompt(sample: dict[str, Any], modality: str) -> str:
    """为单个 Chunk 构造不依赖参考答案或其他 Chunk 的抽取提示词。"""
    section = sample["section"]
    content = sample["content"]
    body = content.get("text") if modality == "text" else content.get("markdown") if modality == "table" else "请直接观察附带图片。"
    if modality == "table" and isinstance(body, str) and Path(body.strip()).suffix.lower() in IMAGE_SUFFIXES:
        body = "表格内容见附带图片。"
    payload = {
        "sample_id": sample["sample_id"],
        "section_title": section.get("section_title"),
        "caption": content.get("caption"),
        "content": body,
    }
    return (
        "你是地学论文知识图谱抽取助手。只根据本次提供的一个 Chunk 的正文或图片及其章节标题、图表题抽取。"
        "不要使用外部知识补足缺失事实。图表只抽取清晰可辨的文字、数值或关系；不能辨认时留空。"
        "请只输出一个 JSON 对象，不要代码围栏、解释或额外字段。格式："
        '{"entities":[{"name":"实体名","type":"EnglishType","type_zh":"中文类型","provenance":"本 Chunk 中的简短证据"}],'
        '"relations":[{"source":"头实体名","target":"尾实体名","relation_name":"中文关系名",'
        '"type":"ENGLISH_RELATION_TYPE","type_zh":"中文关系名","provenance":"本 Chunk 中的简短证据"}]}。'
        "关系的 source 和 target 必须与 entities 中的 name 完全一致；方向按原文。"
        "实体应是有地学或实验意义的具体对象，避免泛词和臆造。无可抽取内容时两个数组均为空。\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def call_model(sample: dict[str, Any], modality: str, input_path: Path, llm: LLMClient, vlm: VLMClient) -> dict[str, Any]:
    """对一个 Chunk 发起一次逻辑模型调用并严格检查 JSON 顶层结构。"""
    prompt = make_prompt(sample, modality)
    pictures = image_paths(sample, modality, input_path)
    if modality == "image" and not pictures:
        raise ValueError(f"图像 Chunk 没有图片：{sample['sample_id']}")
    if pictures:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url": VLMClient.image_to_data_url(path)}} for path in pictures)
        answer = vlm.chat([{"role": "user", "content": content}])
    else:
        answer = llm.chat([{"role": "user", "content": prompt}])
    parsed = safe_json_loads(answer)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("entities"), list) or not isinstance(parsed.get("relations"), list):
        raise ValueError(f"模型未返回 entities/relations 数组：{sample['sample_id']}")
    return parsed


def stable_id(prefix: str, *parts: str) -> str:
    """由样本及抽取内容生成可复现的实体或关系 ID。"""
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def require_text(value: Any, label: str) -> str:
    """要求模型关键字段为非空字符串并去除两端空白。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 必须是非空字符串")
    return value.strip()


def to_graph(sample: dict[str, Any], modality: str, result: dict[str, Any]) -> dict[str, Any]:
    """将单次模型结果转换为与既有 text_extraction.json 一致的 Graph 结构。"""
    sample_id = sample["sample_id"]
    entities: dict[str, Entity] = {}
    for item in result["entities"]:
        if not isinstance(item, dict):
            raise ValueError(f"实体不是对象：{sample_id}")
        name = require_text(item.get("name"), "实体名称")
        if name in entities:
            continue
        entities[name] = Entity(
            id=stable_id("ent", sample_id, name), name=name, official_name=name,
            type=require_text(item.get("type"), "实体英文类型"),
            type_zh=require_text(item.get("type_zh"), "实体中文类型"),
            provenance=str(item.get("provenance") or "").strip(),
        )
    relations: dict[tuple[str, str, str], Relation] = {}
    rejected_relations = 0
    for item in result["relations"]:
        if not isinstance(item, dict):
            raise ValueError(f"关系不是对象：{sample_id}")
        source = require_text(item.get("source"), "关系头实体")
        target = require_text(item.get("target"), "关系尾实体")
        name = require_text(item.get("relation_name"), "关系名称")
        if source not in entities or target not in entities:
            rejected_relations += 1
            continue
        key = (source, name, target)
        if key in relations:
            continue
        relations[key] = Relation(
            id=stable_id("rel", sample_id, *key), type=require_text(item.get("type"), "关系英文类型"),
            relation_name=name, type_zh=str(item.get("type_zh") or name).strip(),
            source_id=entities[source].id, source_name=source, source_type=entities[source].type,
            target_id=entities[target].id, target_name=target, target_type=entities[target].type,
            provenance=str(item.get("provenance") or "").strip(),
        )
    graph = Graph.from_chunk(
        document_id=sample_id.split("::", 1)[0], chunk_id=sample_id,
        modality=SourceModality(modality), entities=entities.values(), relations=relations.values(),
        stage=f"stage_04_{modality}_llm_only",
    )
    graph.metadata.extra = {"validation": {"rejected_count": rejected_relations}}
    return graph.to_dict(exclude={"metadata": {"raw_response"}})


def save_checkpoint(
    output_path: Path, input_path: Path, samples: list[dict[str, Any]], graphs: dict[str, dict],
    status: str, issue: str | None, request_count: int, recovered_count: int,
) -> dict[str, Any]:
    """按现有抽取文件的外层格式原子保存进度及待处理样本。"""
    ids = [sample["sample_id"] for sample in samples]
    records = [graphs[sample_id] for sample_id in ids if sample_id in graphs]
    payload = {
        "_status": status,
        "input": str(input_path.resolve()),
        "statistics": {
            "chunk_count": len(ids), "graph_count": len(records),
            "entity_count": sum(len(item["entities"]) for item in records),
            "relation_count": sum(len(item["relations"]) for item in records),
            "model_request_count": request_count,
            "model_error_count": 1 if issue else 0,
            "invalid_response_count": 1 if issue and issue.startswith("ValueError") else 0,
            "recovered_invalid_response_count": recovered_count, "retry_attempt_count": 0,
        },
        "pending_sample_ids": [sample_id for sample_id in ids if sample_id not in graphs],
        "review_issue": issue,
        "graphs": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return payload


def run(input_path: Path, output_path: Path, modality: str, *, dry_run: bool = False) -> dict[str, Any]:
    """逐 Chunk 抽取并保存检查点，已有成功 Graph 可断点续跑。"""
    samples = read_samples(input_path, modality)
    for sample in samples:
        image_paths(sample, modality, input_path)
    if dry_run:
        return {"_status": "validated_no_extraction", "statistics": {"chunk_count": len(samples)}}
    ids = {sample["sample_id"] for sample in samples}
    graphs: dict[str, dict] = {}
    request_count = 0
    recovered_count = 0
    if output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8-sig"))
        if previous.get("input") != str(input_path.resolve()):
            raise ValueError("已有结果的输入路径不同，请更换输出路径")
        request_count = int(previous.get("statistics", {}).get("model_request_count", 0))
        recovered_count = int(previous.get("statistics", {}).get("recovered_invalid_response_count", 0))
        if previous.get("_status") == "partial_model_error" and previous.get("statistics", {}).get("invalid_response_count"):
            recovered_count += 1
        for graph in previous.get("graphs", []):
            metadata = graph.get("metadata", {})
            sample_id = metadata.get("chunk_id")
            if sample_id not in ids or metadata.get("modality") != modality or sample_id in graphs:
                raise ValueError(f"已有 Graph 的样本 ID 或模态无效：{sample_id}")
            graphs[sample_id] = graph
    llm = LLMClient()
    vlm = VLMClient()
    save_checkpoint(output_path, input_path, samples, graphs, "running", None, request_count, recovered_count)
    for index, sample in enumerate(samples, 1):
        sample_id = sample["sample_id"]
        if sample_id in graphs:
            continue
        try:
            request_count += 1
            response = call_model(sample, modality, input_path, llm, vlm)
            graphs[sample_id] = to_graph(sample, modality, response)
        except Exception as exc:
            issue = f"{type(exc).__name__}: {exc}"
            return save_checkpoint(output_path, input_path, samples, graphs, "partial_model_error", issue, request_count, recovered_count)
        save_checkpoint(output_path, input_path, samples, graphs, "running", None, request_count, recovered_count)
        print(f"{modality} {index}/{len(samples)}：{sample_id}", flush=True)
    return save_checkpoint(output_path, input_path, samples, graphs, "completed", None, request_count, recovered_count)


def main() -> int:
    """解析文本 Chunk 的输入输出路径并执行 LLM 抽取。"""
    parser = argparse.ArgumentParser(description="逐文本 Chunk 仅用提示词和项目 LLM 抽取")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "text_chunk.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "text_extraction.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(args.input, args.output, "text", dry_run=args.dry_run)
    print(f"模型={settings.llm.model}；状态={result['_status']}；{args.output.resolve()}")
    return 0 if result["_status"] in {"completed", "validated_no_extraction"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
