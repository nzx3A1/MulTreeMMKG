"""对单篇 MinerU Markdown 运行抽取前流程，并将三阶段 JSON 保存到原文目录。

用法：python eval/script/preExtractors.py
流程依次为 Markdown 解析、Document/Section/Chunk 骨架搭建和自底向上总结。
"""
from __future__ import annotations

import argparse
import json
import site
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# 中文说明：支持从任意工作目录直接执行本脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 中文说明：使用当前 Conda 环境的 CUDA 13 依赖，避免用户级 PyTorch CUDA 12.8 抢先加载同名 DLL。
USER_SITE = Path(site.getusersitepackages()).resolve()
sys.path[:] = [entry for entry in sys.path if Path(entry).resolve() != USER_SITE]
DEFAULT_MARKDOWN = PROJECT_ROOT / "eval" / "dataset" / "dev" / "mineru_output" / "File1039" / "full.md"

from src.parser.AMarkdownParser import AMarkdownParser
from src.skeleton.document_tree_builder import build_document_tree, document_tree_to_dict
from src.summarizer.bottom_up_summarizer import summarize_document_tree


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """以 UTF-8 原子写入阶段结果，避免进程中断留下不完整的 JSON。"""

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _never_merge_table_images(_path: Path, _threshold: float) -> bool:
    """显式关闭 PaddleOCR 检测时跳过表格续图判断。"""

    return False


def _unique_section_ids(stage01: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """仅在建树输入中修正重复章节 ID，保留原始解析结果与原 ID 溯源。"""

    copied = json.loads(json.dumps(stage01, ensure_ascii=False))
    all_ids: set[str] = set()
    seen: Counter[str] = Counter()
    remaps: list[dict[str, str]] = []

    def collect(sections: list[dict[str, Any]]) -> None:
        """先收集全部原 ID，避免新生成的后缀与原有章节 ID 冲突。"""

        for section in sections:
            all_ids.add(str(section.get("id") or "section"))
            collect(section.get("children") or [])

    def visit(sections: list[dict[str, Any]]) -> None:
        """按章节顺序重命名重复 ID，并记录源 ID 供检查。"""

        for section in sections:
            original = str(section.get("id") or "section")
            seen[original] += 1
            if seen[original] > 1:
                suffix = seen[original]
                candidate = f"{original}__{suffix}"
                while candidate in all_ids:
                    suffix += 1
                    candidate = f"{original}__{suffix}"
                section["source_section_id"] = original
                section["id"] = candidate
                all_ids.add(candidate)
                remaps.append({"source_section_id": original, "section_id": candidate})
            visit(section.get("children") or [])

    collect(copied.get("toc") or [])
    visit(copied.get("toc") or [])
    return copied, remaps


def run_pre_extraction(markdown_path: str | Path = DEFAULT_MARKDOWN, *, detect_table_continuations: bool = True) -> dict[str, Path]:
    """对一个 Markdown 执行解析、建树、总结，并将各阶段结果写入其所在目录。"""

    source = Path(markdown_path).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".md":
        raise FileNotFoundError(f"请输入实际存在的 Markdown 文件：{source}")

    output_dir = source.parent
    paths = {
        "parse": output_dir / "stage_01_mineru_parse.json",
        "tree": output_dir / "stage_02_document_tree.json",
        "summary": output_dir / "stage_03_document_summary.json",
    }
    # 中文说明：默认使用解析器的 PaddleOCR 表格续图检测；可显式关闭。
    detector = None if detect_table_continuations else _never_merge_table_images
    stage01 = AMarkdownParser(table_image_detector=detector).parse_file(source)
    _write_json(paths["parse"], stage01)
    print(f"Markdown 解析完成：{paths['parse']}", flush=True)

    builder_input, remaps = _unique_section_ids(stage01)
    stage02 = document_tree_to_dict(build_document_tree(builder_input, source_json=paths["parse"]))
    stage02["section_id_remapped_count"] = len(remaps)
    stage02["section_id_remaps"] = remaps
    _write_json(paths["tree"], stage02)
    print(f"文档骨架搭建完成：{paths['tree']}", flush=True)

    stage03 = summarize_document_tree(stage02)
    stage03["_stage"] = 3
    stage03["_description"] = "已添加章节与全文摘要、schemaKeys 和 ConceptCategories 的多模态文档目录树。"
    stage03["_produced_by"] = "src/summarizer/bottom_up_summarizer.py::summarize_document_tree"
    _write_json(paths["summary"], stage03)
    print(f"文档总结完成：{paths['summary']}", flush=True)
    return paths


def main() -> int:
    """读取单篇 Markdown 路径并运行全部抽取前步骤。"""

    cli = argparse.ArgumentParser(description="单篇 MinerU Markdown 的抽取前解析、建树与总结")
    cli.add_argument("markdown", nargs="?", type=Path, default=DEFAULT_MARKDOWN, help=f"MinerU 的 .md 文件；默认：{DEFAULT_MARKDOWN}")
    cli.add_argument("--no-detect-table-continuations", action="store_true", help="关闭默认开启的 PaddleOCR 表格续图检测")
    args = cli.parse_args()
    run_pre_extraction(args.markdown, detect_table_continuations=not args.no_detect_table_continuations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
