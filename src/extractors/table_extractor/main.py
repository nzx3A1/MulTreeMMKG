"""第三阶段章节总结 JSON 到表格知识图谱的独立执行入口。

本模块负责读取 ``stage_03_document_summary.json``、递归收集章节中的表格
Chunk，并把它们交给表格抽取目录中已有的识别、质量门和 Graph 装配流程。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

# 中文说明：支持直接执行本文件，将项目根目录加入模块搜索路径。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import Graph
from src.extractors.extractor_init import collect_chunks
from src.extractors.table_extractor.table_extractor import extract_from_tables
from src.utils.json_io import read_json
from src.utils.llm_client import LLMClient


DEFAULT_INPUT_PATH = PROJECT_ROOT / "output" / "stage_03_document_summary.json"
"""第三阶段章节总结 JSON 的默认输入路径。"""

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "stage_04_table_extraction.json"
"""通过质量门的表格 Graph 输出路径。"""

DEFAULT_WORK_DIR = PROJECT_ROOT / "output" / "table_extraction"
"""逐表识别产物和诊断文件目录。"""


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """校验 JSON 顶层是对象，并给出便于定位输入问题的错误信息。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必须是 JSON 对象，实际为 {type(value).__name__}")
    return value


def _is_table_chunk(chunk: Mapping[str, Any]) -> bool:
    """兼容不同模态字段写法，只保留表格 Chunk。"""

    value = chunk.get("modality") or chunk.get("chunk_type") or chunk.get("type") or ""
    modality = str(getattr(value, "value", value)).strip().lower()
    aliases = {"tablechunk": "table"}
    return aliases.get(modality.replace("_", ""), modality) == "table"


def extract_tables_from_summary_file(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    work_dir: str | Path = DEFAULT_WORK_DIR,
    llm_client: Any | None = None,
    vlm_client: Any | None = None,
    engine: str = "auto",
    rapid_model: str = "unitable",
    device: str = "auto",
    ocr_device: str = "cpu",
    ocr_backend: str = "onnxruntime",
    ocr_limit_side_len: int = 1600,
    use_llm_semantic: bool = False,
    minimum_score: float = 0.55,
    show_progress: bool = True,
) -> list[Graph]:
    """读取第三阶段总结文件，识别全部表格 Chunk 并保存抽取产物。

    章节树由公共 ``collect_chunks`` 递归展开，因此每个表格任务都带有文档、章节、
    总结和 Schema 上下文。实际图片识别、HTML 网格解析、质量门判断与知识图谱装配
    均复用 ``extract_from_tables``；单表失败不会阻断其他表格。
    """

    # 中文说明：公共收集器补齐文档和章节溯源字段，此处只筛选并规范化表格模态。
    payload = _as_mapping(read_json(input_path), "第三阶段总结输入")
    table_chunks: list[dict[str, Any]] = []
    for chunk in collect_chunks(payload):
        if not _is_table_chunk(chunk):
            continue
        normalized = dict(chunk)
        normalized["modality"] = "table"
        table_chunks.append(normalized)

    # 中文说明：仅在启用 LLM 语义补充时创建客户端，默认流程保持确定性规则抽取。
    client = llm_client
    if use_llm_semantic and client is None:
        client = LLMClient()

    # 中文说明：即使没有表格也调用公共管线，以写出一个标准空结果供下游读取。
    return extract_from_tables(
        table_chunks,
        client,
        vlm_client=vlm_client,
        output_path=output_path,
        work_dir=work_dir,
        engine=engine,
        rapid_model=rapid_model,
        device=device,
        ocr_device=ocr_device,
        ocr_backend=ocr_backend,
        ocr_limit_side_len=ocr_limit_side_len,
        use_llm_semantic=use_llm_semantic,
        minimum_score=max(0.0, min(1.0, minimum_score)),
        show_progress=show_progress,
    )


def _build_cli() -> argparse.ArgumentParser:
    """构建从第三阶段总结文件启动表格抽取的命令行参数。"""

    parser = argparse.ArgumentParser(description="从第三阶段章节总结 JSON 执行表格抽取")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="第三阶段章节总结 JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="表格 Graph JSON")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help="逐表识别和诊断产物目录")
    parser.add_argument("--engine", choices=("auto", "rapidtable", "mineru"), default="auto", help="图片表格识别引擎")
    parser.add_argument("--rapid-model", choices=("unitable", "slanetplus", "ppstructure_zh"), default="unitable", help="RapidTable 结构模型")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="RapidTable 推理设备")
    parser.add_argument("--ocr-device", choices=("auto", "cpu", "cuda"), default="cpu", help="OCR 推理设备")
    parser.add_argument("--ocr-backend", choices=("onnxruntime", "torch"), default="onnxruntime", help="OCR 执行后端")
    parser.add_argument("--ocr-limit-side-len", type=int, default=1600, help="OCR 检测最长边")
    parser.add_argument("--minimum-score", type=float, default=0.55, help="HTML 质量门最低分")
    parser.add_argument("--use-llm-semantic", action="store_true", help="规则无法确定主键时调用文本模型")
    parser.add_argument("--quiet", action="store_true", help="不显示逐表进度")
    return parser


def main() -> int:
    """命令行入口：从第三阶段总结文件启动表格识别和 Graph 抽取。"""

    args = _build_cli().parse_args()
    graphs = extract_tables_from_summary_file(
        args.input,
        args.output,
        work_dir=args.work_dir,
        engine=args.engine,
        rapid_model=args.rapid_model,
        device=args.device,
        ocr_device=args.ocr_device,
        ocr_backend=args.ocr_backend,
        ocr_limit_side_len=args.ocr_limit_side_len,
        use_llm_semantic=args.use_llm_semantic,
        minimum_score=args.minimum_score,
        show_progress=not args.quiet,
    )
    print(f"表格抽取完成：Graph={len(graphs)}，输出文件：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_WORK_DIR",
    "extract_tables_from_summary_file",
    "main",
]
