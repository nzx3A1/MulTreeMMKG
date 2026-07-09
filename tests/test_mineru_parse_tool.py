"""MinerU 工具脚本的本地单 PDF 场景测试。"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from util.mineru_parse_tool import extract_zip_to_output


def test_extract_zip_to_output_normalizes_mineru_files(tmp_path: Path):
    """本地 zip 解压后应自动生成 full.md、content_list.json 和 layout.json。"""

    zip_path = tmp_path / "paper_001.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("nested/full.md", "# 测试论文\n")
        archive.writestr("nested/demo_content_list.json", json.dumps([{"type": "text", "text": "正文"}], ensure_ascii=False))
        archive.writestr("nested/demo_middle.json", json.dumps({"pdf_info": []}, ensure_ascii=False))

    output_dir = extract_zip_to_output(zip_path, "paper_001", output_root=tmp_path / "mineru_output")

    assert (output_dir / "full.md").exists()
    assert (output_dir / "content_list.json").exists()
    assert (output_dir / "layout.json").exists()
