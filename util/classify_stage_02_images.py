"""递归提取 Stage 02 文档树中的图片，并生成图片分类前端可直接读取的数据集。

脚本会把多图 ImageChunk 拆成逐图片记录，调用项目现有 VLM 执行 A01-A20 分类，
同时保留父 Chunk、章节、图注和正文引用等溯源信息，支持在中断后复用已有结果续跑。
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import os
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


# 中文说明：支持从任意工作目录直接执行脚本，并复用项目内已有分类器和 JSON 适配逻辑。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.json_io import read_json
from src.utils.llm_client import safe_json_loads
from util.classify_image_chunks import TAXONOMY as IMAGE_TAXONOMY


# 中文说明：复用既有 A01-A20 名称，但不在测试导入阶段加载依赖 Pydantic v2 的抽取包。
TYPE_BY_CODE = {item["code"]: item["name"] for item in IMAGE_TAXONOMY}


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

    document = _as_mapping(payload.get("document", payload), "document")
    document_id = str(document.get("id") or document.get("document_id") or "").strip()
    if not document_id:
        raise ValueError("Stage 02 文档树缺少 document.id")

    chunks: list[dict[str, Any]] = []

    def visit(section: Mapping[str, Any], inherited_title: str = "") -> None:
        """深度优先遍历章节，并把章节上下文补到每个图片 Chunk。"""

        section_id = str(section.get("id") or section.get("section_id") or "")
        section_title = str(section.get("title") or inherited_title or "")
        for raw_chunk in section.get("chunks") or []:
            chunk = _as_mapping(raw_chunk, f"section[{section_id}].chunks[]")
            if not _is_image_chunk(chunk):
                continue
            normalized = dict(chunk)
            normalized.setdefault("document_id", document_id)
            normalized.setdefault("section_id", section_id)
            normalized.setdefault("section_title", section_title)
            chunks.append(normalized)
        for child in section.get("children") or []:
            visit(_as_mapping(child, f"section[{section_id}].children[]"), section_title)

    for section in document.get("sections") or []:
        visit(_as_mapping(section, "document.sections[]"))

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


@dataclass(frozen=True)
class _ClassificationTask:
    """VLM 分类器实际读取的最小任务字段，避免脚本依赖模型包的版本细节。"""

    document_id: str
    chunk_id: str
    image_id: str
    image_index: int
    image_path: str
    caption: str = ""
    references: tuple[str, ...] = ()
    section_id: str = ""
    section_title: str = ""
    raw_chunk: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class _ClassificationResult:
    """保存并序列化一次 VLM 图片分类结果。"""

    primary_code: str
    primary_type: str
    secondary_codes: tuple[str, ...]
    confidence: float
    reason: str
    visual_evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """输出前端和后续图片抽取阶段都能识别的分类字段。"""

        return {
            "primary_code": self.primary_code,
            "primary_type": self.primary_type,
            "secondary_codes": list(self.secondary_codes),
            "secondary_types": [TYPE_BY_CODE[code] for code in self.secondary_codes],
            "confidence": self.confidence,
            "reason": self.reason,
            "visual_evidence": list(self.visual_evidence),
            "basis": "vlm_visual_classification",
            "source": "vlm_visual_classification",
        }


def _build_vlm_prompt(task: _ClassificationTask) -> str:
    """构造要求模型严格依据图片视觉主体单选 A01-A20 的提示词。"""

    taxonomy = "\n".join(f"- {code} {name}" for code, name in IMAGE_TAXONOMY)
    references = json.dumps(list(task.references), ensure_ascii=False)
    return f"""
你是石油地质论文图片分类专家。请读取随消息提供的当前图片，在 A01-A20 中选择唯一主类别。

图片 ID：{task.image_id}
图题：{task.caption or "无"}
正文参考（仅用于消歧，不得覆盖图片视觉主体）：
{references}

分类体系：
{taxonomy}

判定规则：
1. 先判断当前这张图片的主版式：照片/显微图、地图、剖面、示意模型或坐标图表。必须以当前图片可见主体为准；父 Chunk 图注可能同时描述多个子图，只能用于消歧，不能把别的子图内容带入当前图片。
2. 图表分流是硬规则：只要主体是带坐标轴、曲线、散点、柱形、饼形或统计符号的图表，不能因为出现“孔隙、储层、岩石”等词就判为 A12、A19 或 A13。
3. A13 只保留“图像型”的孔隙结构实验：CT 二维切片/三维体、核磁成像、三维孔喉或 node-link 节点连线空间网络；A13 不包含普通坐标图表。即使横轴写着孔隙半径，只要主体是曲线和频率轴，就判 A15（统计分布与组成图），不能判 A13。普通孔隙度—渗透率关系图判 A16，碳氧同位素/稀土/色谱等地球化学图判 A14，随时间或阶段变化的曲线判 A17。
4. A12 仅限扫描电镜等电子显微形貌照片：应看到微观纹理和标尺，不能有明显坐标轴、回归线或统计图例。A11 仅限光学薄片/显微照片。图表不是 A11/A12；三维图的 X/Y/Z 方向标不等于统计坐标轴。
5. 箱体、箭头、阶段 I/II/III/IV、流体运移或成岩反应组成的概念示意图判 A09；油气藏/源储盖/运移富集模式判 A08。没有井间横向对比的“模式图”绝不能判 A04；没有真实显微纹理的流程图不能判 A13。
6. A03 必须用于纵向地层柱状/综合柱状表：看到地层系统、层组/段、深度轴、岩性柱、沉积相和储层栏的组合时，优先判 A03，即使图中有一条相对海平面曲线。A04 必须有地质剖面或多井连井对比的横向层位关系；A05 必须有地震反射同相轴或地球物理剖面；A06 必须有多道随深度变化的测井曲线；A07 必须是平面等值线、色斑或参数空间分布。
7. 多面板同时含薄片/显微照片、岩心照片和孔隙标注时，若当前图片仍是完整图版，判 A19；不要仅因其中一个面板是薄片就判 A11。典型判例：孔隙度—渗透率散点/拟合线=A16；碳氧同位素交会图=A14；孔径—频率分布曲线=A15；白云石化/孔隙发育过程框图=A09；SEM 灰度形貌图=A12；CT 切片或 node-link 三维孔网图=A13。
8. secondary_codes 只保留当前图片中有独立视觉面板支持的辅助类别，不要把父图注中的其他子图类别列为副类；visual_evidence 写 1 至 4 条当前图片直接可见的证据，不要输出思维过程。
9. confidence 是 0 到 1 的数值；版式与语义冲突或图像模糊时降低置信度。只输出 JSON 对象，不要 Markdown 或额外文字。

输出格式：
{{
  "primary_code": "A01-A20 中的一个值",
  "primary_type": "对应中文名称",
  "secondary_codes": [],
  "confidence": 0.0,
  "reason": "简短判定结论",
  "visual_evidence": ["直接可见证据1", "直接可见证据2"]
}}
""".strip()


class _VLMClassifier:
    """使用项目 VLMClient 完成逐图分类，并独立校验结构化响应。"""

    def classify(self, task: _ClassificationTask, vlm_client: Any) -> _ClassificationResult:
        """调用视觉模型并校验类别、置信度、理由和视觉证据字段。"""

        if vlm_client is None or not hasattr(vlm_client, "describe_image"):
            raise TypeError("VLM 客户端需要支持 describe_image 方法")
        response = vlm_client.describe_image(
            task.image_path,
            _build_vlm_prompt(task),
            task_name=f"石油地质图片大类分类:{task.image_id}",
            response_format={"type": "json_object"},
            max_tokens=int(os.getenv("IMAGE_CLASSIFICATION_VLM_MAX_TOKENS", "1536")),
        )
        payload = response if isinstance(response, Mapping) else safe_json_loads(str(response or ""))
        if not isinstance(payload, Mapping):
            raise ValueError("VLM 响应不是 JSON 对象")
        primary_code = str(payload.get("primary_code") or "").strip().upper()
        if primary_code not in VALID_CODES:
            raise ValueError(f"VLM 返回无效主类别：{primary_code!r}")
        try:
            confidence = float(payload.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise ValueError("VLM 响应缺少有效 confidence") from exc
        reason = str(payload.get("reason") or "").strip()
        if not reason:
            raise ValueError("VLM 响应缺少 reason")
        evidence_raw = payload.get("visual_evidence")
        if not isinstance(evidence_raw, list):
            raise ValueError("VLM 响应的 visual_evidence 必须是数组")
        evidence = tuple(str(item).strip() for item in evidence_raw if str(item).strip())
        if not evidence:
            raise ValueError("VLM 响应缺少可核验视觉证据")
        secondary_raw = payload.get("secondary_codes")
        secondary_codes: list[str] = []
        if isinstance(secondary_raw, list):
            for raw_code in secondary_raw:
                code = str(raw_code).strip().upper()
                if code in VALID_CODES and code != primary_code and code not in secondary_codes:
                    secondary_codes.append(code)
        return _ClassificationResult(
            primary_code=primary_code,
            primary_type=TYPE_BY_CODE[primary_code],
            secondary_codes=tuple(secondary_codes),
            confidence=round(max(0.0, min(1.0, confidence)), 3),
            reason=reason,
            visual_evidence=evidence,
        )


def _classification_task(chunk: Mapping[str, Any], record: Mapping[str, Any]) -> ImageExtractionTask:
    """把逐图记录转换为项目现有 VLM 分类器所需的最小任务对象。"""

    return _ClassificationTask(
        document_id=str(chunk.get("document_id") or ""),
        chunk_id=str(chunk["id"]),
        image_id=str(record["image_id"]),
        image_index=int(record["image_index_in_chunk"]),
        image_path=str(record["image_path"]),
        caption=str(chunk.get("caption") or ""),
        references=tuple(_string_list(chunk.get("references"))),
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
            classification = record.get("classification")
            if not isinstance(classification, dict):
                # 中文说明：检查点中尚未处理的图片只保留 pending，不伪造 A01-A20 分类。
                continue
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

    # 中文说明：仅在真正执行联网分类时加载 VLMClient，离线结构测试无需模型依赖。
    from src.utils.vlm_client import VLMClient

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
        classifier=_VLMClassifier(),
        vlm_client=VLMClient(),
        force=args.force,
    )
    print(json.dumps(result["statistics"], ensure_ascii=False, indent=2))
    print(f"输出文件：{args.output.resolve()}")
    if result["statistics"]["failed"]:
        raise SystemExit("部分图片分类失败；修复服务或路径后直接重跑即可续跑失败项")


if __name__ == "__main__":
    main()
