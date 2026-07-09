"""MinerU VLM 精准解析工具。

该工具面向单个 PDF 文件完成完整 MinerU 云端解析流程：
申请上传地址、上传 PDF、轮询解析结果、下载 zip、校验 zip、解压到
data/mineru_output/<paper_id>/，并把 MinerU 常见文件名归一化为
full.md、content_list.json、layout.json。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.app_config import settings


RESULT_URL_TEMPLATE = "https://mineru.net/api/v4/extract-results/batch/{batch_id}"


@dataclass(frozen=True)
class MinerUParseResult:
    """记录单个 PDF 经 MinerU 解析后的本地落地信息。"""

    paper_id: str
    source_pdf: str
    batch_id: str
    output_dir: Path
    zip_path: Path
    state: str


class MinerUParseError(RuntimeError):
    """MinerU 解析流程失败时抛出的业务异常。"""


def _project_relative(path: Path) -> str:
    """将项目内路径转成相对路径，便于 index.json 跨机器复用。"""

    try:
        return str(path.resolve().relative_to(settings.project_root))
    except ValueError:
        return str(path)


def _headers(token: str) -> dict[str, str]:
    """生成 MinerU API 请求头。"""

    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


def _safe_reset_dir(target_dir: Path, root_dir: Path) -> None:
    """安全清空目标目录，确保删除范围始终限制在 MinerU 输出根目录内。"""

    target_dir = target_dir.resolve()
    root_dir = root_dir.resolve()
    try:
        target_dir.relative_to(root_dir)
    except ValueError as exc:
        raise MinerUParseError(f"目标目录不在 MinerU 输出根目录内: {target_dir}") from exc
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)


def _request_upload_url(pdf_path: Path, paper_id: str, token: str, timeout_secs: float) -> tuple[str, str]:
    """向 MinerU 申请单文件上传 URL，并返回 batch_id 与上传地址。"""

    payload = {
        "files": [{"name": pdf_path.name, "data_id": paper_id}],
        "model_version": "vlm",
        "enable_formula": True,
        "enable_table": True,
        "language": "ch",
    }
    response = requests.post(settings.model.mineru.batch_url, headers=_headers(token), json=payload, timeout=timeout_secs)
    data = response.json()
    if response.status_code != 200 or data.get("code") != 0:
        raise MinerUParseError(f"申请 MinerU 上传 URL 失败: HTTP {response.status_code}, {data}")
    urls = data.get("data", {}).get("file_urls") or []
    batch_id = data.get("data", {}).get("batch_id")
    if not batch_id or not urls:
        raise MinerUParseError(f"MinerU 返回缺少 batch_id 或 file_urls: {data}")
    return str(batch_id), str(urls[0])


def _upload_pdf(pdf_path: Path, upload_url: str, timeout_secs: float) -> None:
    """把 PDF 二进制内容上传到 MinerU 返回的对象存储地址。"""

    with pdf_path.open("rb") as file:
        response = requests.put(upload_url, data=file, timeout=max(timeout_secs, 300))
    if response.status_code != 200:
        raise MinerUParseError(f"上传 PDF 到 MinerU 失败: HTTP {response.status_code}")


def _query_batch(batch_id: str, token: str, timeout_secs: float) -> dict[str, Any]:
    """查询 MinerU 批量任务状态并返回第一条解析结果。"""

    url = RESULT_URL_TEMPLATE.format(batch_id=batch_id)
    response = requests.get(url, headers=_headers(token), timeout=timeout_secs)
    data = response.json()
    if response.status_code != 200 or data.get("code") != 0:
        raise MinerUParseError(f"查询 MinerU 解析状态失败: HTTP {response.status_code}, {data}")
    results = data.get("data", {}).get("extract_result") or []
    if not results:
        raise MinerUParseError(f"MinerU 尚未返回解析结果: {data}")
    return dict(results[0])


def _wait_until_done(batch_id: str, token: str, timeout_secs: float, poll_interval: float, max_wait_secs: float) -> dict[str, Any]:
    """轮询 MinerU 任务直到 done 或 failed，超时则抛出异常。"""

    deadline = time.monotonic() + max_wait_secs
    while time.monotonic() < deadline:
        result = _query_batch(batch_id, token, timeout_secs)
        state = result.get("state")
        if state == "done":
            return result
        if state in {"failed", "error"}:
            raise MinerUParseError(f"MinerU 解析失败: {result.get('err_msg') or result}")
        time.sleep(poll_interval)
    raise MinerUParseError(f"MinerU 解析超时: batch_id={batch_id}")


def _download_zip(url: str, zip_path: Path, timeout_secs: float, retry_times: int = 3) -> None:
    """下载 MinerU zip 结果包；代理失败时自动尝试直连下载。"""

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = zip_path.with_suffix(".zip.part")
    sessions: list[requests.Session] = []
    default_session = requests.Session()
    sessions.append(default_session)
    direct_session = requests.Session()
    direct_session.trust_env = False
    sessions.append(direct_session)

    last_error: Exception | None = None
    for session in sessions:
        for attempt in range(1, retry_times + 1):
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
                with session.get(url, stream=True, timeout=(30, max(timeout_secs, 300))) as response:
                    response.raise_for_status()
                    with tmp_path.open("wb") as file:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                file.write(chunk)
                _assert_valid_zip(tmp_path)
                tmp_path.replace(zip_path)
                return
            except Exception as exc:  # noqa: BLE001 - 这里需要保留最后一次下载异常供诊断。
                last_error = exc
                time.sleep(min(5 * attempt, 20))
    raise MinerUParseError(f"下载 MinerU zip 失败: {last_error}")


def _assert_valid_zip(zip_path: Path) -> None:
    """校验 zip 文件完整性，防止半下载文件被解压。"""

    with zipfile.ZipFile(zip_path) as archive:
        bad_file = archive.testzip()
        if bad_file:
            raise MinerUParseError(f"zip 文件损坏，异常条目: {bad_file}")


def _normalize_mineru_files(target_dir: Path) -> list[str]:
    """把 MinerU zip 内常见文件名复制为阶段二加载器固定读取的标准文件名。"""

    files = [path for path in target_dir.rglob("*") if path.is_file()]
    for path in files:
        name = path.name
        if name == "full.md":
            if path.resolve() != (target_dir / "full.md").resolve():
                shutil.copy2(path, target_dir / "full.md")
        elif name.endswith("_content_list.json") or name == "content_list.json":
            if path.resolve() != (target_dir / "content_list.json").resolve():
                shutil.copy2(path, target_dir / "content_list.json")
        elif name.endswith("_middle.json") or name in {"layout.json", "middle.json"}:
            if path.resolve() != (target_dir / "layout.json").resolve():
                shutil.copy2(path, target_dir / "layout.json")
    return [name for name in ("full.md", "content_list.json", "layout.json") if not (target_dir / name).exists()]


def extract_zip_to_output(zip_path: Path, paper_id: str, output_root: Path | None = None, overwrite: bool = True) -> Path:
    """把 MinerU zip 解压到对应 paper_id 目录，并归一化核心文件名。"""

    output_root = output_root or settings.mineru_output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    _assert_valid_zip(zip_path)
    target_dir = output_root / paper_id
    if overwrite:
        _safe_reset_dir(target_dir, output_root)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_dir)
    missing = _normalize_mineru_files(target_dir)
    if missing:
        raise MinerUParseError(f"解压后缺少 MinerU 核心文件: {', '.join(missing)}")
    return target_dir


def _update_index(result: MinerUParseResult, output_root: Path) -> None:
    """更新 data/mineru_output/index.json，记录 PDF 与 paper_id 的对应关系。"""

    index_path = output_root / "index.json"
    rows: list[dict[str, Any]] = []
    if index_path.exists():
        loaded = json.loads(index_path.read_text(encoding="utf-8"))
        rows = loaded if isinstance(loaded, list) else []
    rows = [row for row in rows if row.get("paper_id") != result.paper_id]
    rows.append({
        "paper_id": result.paper_id,
        "source_pdf": Path(result.source_pdf).name,
        "state": result.state,
        "batch_id": result.batch_id,
        "output_dir": _project_relative(result.output_dir),
        "zip_path": _project_relative(result.zip_path),
        "missing": [],
    })
    rows.sort(key=lambda row: str(row.get("paper_id", "")))
    index_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_pdf_with_mineru(
    pdf_path: Path,
    paper_id: str | None = None,
    output_root: Path | None = None,
    token: str | None = None,
    poll_interval: float = 10.0,
    max_wait_secs: float = 1800.0,
) -> MinerUParseResult:
    """调用 MinerU VLM 精准解析单个 PDF，并自动下载、解压到对应目录。"""

    pdf_path = pdf_path.resolve()
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"PDF 文件不存在或后缀不是 .pdf: {pdf_path}")
    output_root = output_root or settings.mineru_output_dir
    paper_id = paper_id or pdf_path.stem
    token = token or settings.model.mineru.token
    if not token:
        raise MinerUParseError("未配置 MINERU_TOKEN 或 config.model_config.MinerUConfig.token")

    batch_id, upload_url = _request_upload_url(pdf_path, paper_id, token, settings.model.mineru.timeout_secs)
    _upload_pdf(pdf_path, upload_url, settings.model.mineru.timeout_secs)
    mineru_result = _wait_until_done(batch_id, token, settings.model.mineru.timeout_secs, poll_interval, max_wait_secs)
    zip_url = mineru_result.get("full_zip_url")
    if not zip_url:
        raise MinerUParseError(f"MinerU 完成但未返回 full_zip_url: {mineru_result}")

    zip_path = output_root / "_zips" / f"{paper_id}.zip"
    _download_zip(str(zip_url), zip_path, settings.model.mineru.timeout_secs)
    target_dir = extract_zip_to_output(zip_path, paper_id, output_root=output_root, overwrite=True)
    result = MinerUParseResult(
        paper_id=paper_id,
        source_pdf=str(pdf_path),
        batch_id=batch_id,
        output_dir=target_dir,
        zip_path=zip_path,
        state=str(mineru_result.get("state")),
    )
    _update_index(result, output_root)
    return result


def main() -> None:
    """命令行入口：解析单个 PDF 并把结果写入 MinerU 输出目录。"""

    parser = argparse.ArgumentParser(description="调用 MinerU VLM 精准解析单个 PDF")
    parser.add_argument("--pdf", required=True, help="待解析的 PDF 文件路径")
    parser.add_argument("--paper-id", default=None, help="输出目录名，默认使用 PDF 文件名")
    parser.add_argument("--output-root", default=None, help="MinerU 输出根目录，默认 data/mineru_output")
    parser.add_argument("--poll-interval", type=float, default=10.0, help="轮询间隔秒数")
    parser.add_argument("--max-wait-secs", type=float, default=1800.0, help="最大等待秒数")
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else None
    result = parse_pdf_with_mineru(
        pdf_path=Path(args.pdf),
        paper_id=args.paper_id,
        output_root=output_root,
        poll_interval=args.poll_interval,
        max_wait_secs=args.max_wait_secs,
    )
    print(json.dumps({
        "paper_id": result.paper_id,
        "batch_id": result.batch_id,
        "output_dir": str(result.output_dir),
        "zip_path": str(result.zip_path),
        "state": result.state,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
