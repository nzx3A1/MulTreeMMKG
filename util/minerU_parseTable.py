"""调用 MinerU 云端 API 解析本地表格图片，并打印结构化 JSON 结果。"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.model_config import settings as model_settings


DEFAULT_IMAGE_PATH = Path(__file__).with_name("Snipaste_2026-07-15_22-39-44.png")
APPLY_UPLOAD_URL = model_settings.mineru.batch_url
QUERY_RESULT_URL = "https://mineru.net/api/v4/extract-results/batch/{batch_id}"
DEFAULT_POLL_INTERVAL = 5.0
DEFAULT_MAX_WAIT = 600.0
DOWNLOAD_RETRIES = 3


class MinerUAPIError(RuntimeError):
    """表示 MinerU 请求失败、返回格式异常或解析任务执行失败。"""


def _pretty_json(data: Any) -> str:
    """将 API 或解析产物格式化为便于终端阅读的 UTF-8 JSON 文本。"""
    return json.dumps(data, ensure_ascii=False, indent=2)


def _request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    **kwargs: Any,
) -> dict[str, Any]:
    """发送 HTTP 请求并统一校验状态码、JSON 格式及 MinerU 业务状态码。"""
    try:
        response = session.request(
            method,
            url,
            headers=headers,
            timeout=timeout,
            **kwargs,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise MinerUAPIError(f"请求 MinerU 失败：{exc}") from exc

    try:
        result = response.json()
    except ValueError as exc:
        preview = response.text[:500]
        raise MinerUAPIError(f"MinerU 未返回合法 JSON：{preview}") from exc

    if not isinstance(result, dict):
        raise MinerUAPIError(f"MinerU 返回的 JSON 顶层不是对象：{result!r}")
    if result.get("code") != 0:
        raise MinerUAPIError(
            f"MinerU 接口返回错误：code={result.get('code')}, "
            f"msg={result.get('msg', '未知错误')}"
        )
    return result


def create_upload_task(
    session: requests.Session,
    image_path: Path,
    headers: dict[str, str],
    timeout: float,
) -> tuple[str, str]:
    """为一张表格图片申请签名上传地址，并返回批次 ID 与上传 URL。"""
    payload = {
        "files": [
            {
                "name": image_path.name,
                "data_id": image_path.stem,
                "is_ocr": True,
            }
        ],
        "model_version": "vlm",
        "enable_table": True,
        "enable_formula": True,
        "language": "ch",
    }
    result = _request_json(
        session,
        "POST",
        APPLY_UPLOAD_URL,
        headers=headers,
        timeout=timeout,
        json=payload,
    )
    print("\n[1/4] 申请上传地址返回的 JSON：")
    print(_pretty_json(result))

    data = result.get("data")
    if not isinstance(data, dict):
        raise MinerUAPIError("申请上传地址响应缺少 data 对象")

    batch_id = data.get("batch_id")
    file_urls = data.get("file_urls")
    if not isinstance(batch_id, str) or not batch_id:
        raise MinerUAPIError("申请上传地址响应缺少 batch_id")
    if not isinstance(file_urls, list) or len(file_urls) != 1:
        raise MinerUAPIError("申请上传地址响应中的 file_urls 数量不正确")
    if not isinstance(file_urls[0], str) or not file_urls[0]:
        raise MinerUAPIError("申请上传地址响应中的上传 URL 无效")
    return batch_id, file_urls[0]


def upload_image(
    session: requests.Session,
    image_path: Path,
    upload_url: str,
    timeout: float,
) -> None:
    """使用 MinerU 返回的签名 URL 上传图片；按官方要求不设置 Content-Type。"""
    try:
        with image_path.open("rb") as image_file:
            response = session.put(upload_url, data=image_file, timeout=timeout)
        response.raise_for_status()
    except (OSError, requests.RequestException) as exc:
        raise MinerUAPIError(f"上传表格图片失败：{exc}") from exc
    print(f"\n[2/4] 图片上传成功：{image_path}")


def wait_for_result(
    session: requests.Session,
    batch_id: str,
    headers: dict[str, str],
    request_timeout: float,
    poll_interval: float,
    max_wait: float,
) -> dict[str, Any]:
    """轮询批次任务，直至目标图片完成解析或达到最大等待时间。"""
    query_url = QUERY_RESULT_URL.format(batch_id=batch_id)
    deadline = time.monotonic() + max_wait
    poll_count = 0

    while True:
        poll_count += 1
        result = _request_json(
            session,
            "GET",
            query_url,
            headers=headers,
            timeout=request_timeout,
        )
        data = result.get("data")
        extract_results = data.get("extract_result") if isinstance(data, dict) else None
        if not isinstance(extract_results, list) or not extract_results:
            raise MinerUAPIError("查询响应缺少 data.extract_result")

        task = extract_results[0]
        if not isinstance(task, dict):
            raise MinerUAPIError("查询响应中的 extract_result 项格式错误")
        state = task.get("state")
        print(f"[3/4] 第 {poll_count} 次查询：state={state}")

        if state == "done":
            print("\n解析任务最终返回的 JSON：")
            print(_pretty_json(result))
            return task
        if state == "failed":
            raise MinerUAPIError(f"MinerU 解析失败：{task.get('err_msg', '未知原因')}")
        if time.monotonic() >= deadline:
            raise MinerUAPIError(f"等待 MinerU 解析超时（{max_wait:g} 秒）")

        time.sleep(poll_interval)


def download_content_list_json(
    session: requests.Session,
    full_zip_url: str,
    timeout: float,
) -> tuple[str, Any]:
    """重试下载 MinerU 结果 ZIP，并读取记录表格结构的 content_list JSON。"""
    response: requests.Response | None = None
    last_error: requests.RequestException | None = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        # 首次沿用系统网络设置；若代理导致 CDN TLS 中断，后续请求绕过代理直连重试。
        direct_session: requests.Session | None = None
        download_session = session
        if attempt > 1:
            direct_session = requests.Session()
            direct_session.trust_env = False
            download_session = direct_session
        try:
            candidate_response = download_session.get(
                full_zip_url,
                timeout=timeout,
            )
            candidate_response.raise_for_status()
            response = candidate_response
            break
        except requests.RequestException as exc:
            last_error = exc
            print(f"结果下载第 {attempt}/{DOWNLOAD_RETRIES} 次失败：{exc}")
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(2.0 * attempt)
        finally:
            if direct_session is not None:
                direct_session.close()

    if response is None:
        raise MinerUAPIError(f"下载 MinerU 解析结果失败：{last_error}") from last_error

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            json_names = [
                name
                for name in archive.namelist()
                if Path(name).name == "content_list.json"
                or Path(name).name.endswith("_content_list.json")
            ]
            if not json_names:
                available = ", ".join(archive.namelist())
                raise MinerUAPIError(
                    "结果 ZIP 中未找到 content_list JSON；"
                    f"压缩包内容：{available}"
                )
            json_name = sorted(json_names, key=lambda name: (len(name), name))[0]
            with archive.open(json_name) as json_file:
                content = json.load(json_file)
    except zipfile.BadZipFile as exc:
        raise MinerUAPIError("MinerU 下载结果不是有效的 ZIP 文件") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MinerUAPIError(f"读取 content_list JSON 失败：{exc}") from exc

    return json_name, content


def parse_table_image(
    image_path: Path,
    *,
    token: str,
    request_timeout: float,
    poll_interval: float,
    max_wait: float,
) -> Any:
    """执行申请上传、图片上传、结果轮询和结构化 JSON 下载的完整流程。"""
    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"表格图片不存在：{image_path}")
    if not token.strip():
        raise MinerUAPIError("未配置 MinerU Token，请设置 MINERU_TOKEN")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token.strip()}",
    }
    with requests.Session() as session:
        batch_id, upload_url = create_upload_task(
            session, image_path, headers, request_timeout
        )
        upload_image(session, image_path, upload_url, request_timeout)
        task = wait_for_result(
            session,
            batch_id,
            headers,
            request_timeout,
            poll_interval,
            max_wait,
        )

        full_zip_url = task.get("full_zip_url")
        if not isinstance(full_zip_url, str) or not full_zip_url:
            raise MinerUAPIError("解析完成，但响应中缺少 full_zip_url")
        json_name, content = download_content_list_json(
            session, full_zip_url, request_timeout
        )

    print(f"\n[4/4] 表格解析 JSON（{json_name}）：")
    print(_pretty_json(content))
    return content


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数，使脚本既可直接运行也可指定其他表格图片。"""
    parser = argparse.ArgumentParser(description="调用 MinerU 解析表格图片并打印 JSON")
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help=f"待解析图片路径（默认：{DEFAULT_IMAGE_PATH}）",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="轮询间隔秒数（默认：5）",
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=DEFAULT_MAX_WAIT,
        help="最大等待秒数（默认：600）",
    )
    return parser


def main() -> int:
    """解析命令行参数并运行表格图片解析，失败时输出明确原因。"""
    args = build_argument_parser().parse_args()
    try:
        parse_table_image(
            args.image,
            token=model_settings.mineru.token,
            request_timeout=model_settings.mineru.timeout_secs,
            poll_interval=max(args.poll_interval, 0.1),
            max_wait=max(args.max_wait, 1.0),
        )
    except (FileNotFoundError, MinerUAPIError) as exc:
        print(f"\n解析失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
