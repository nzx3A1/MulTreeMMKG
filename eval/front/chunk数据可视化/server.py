"""Chunk 数据集可视化页面的本地静态服务与受限数据接口。"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


FRONT_DIR = Path(__file__).resolve().parent
ROOT_DIR = FRONT_DIR.parents[2]
CHUNK_DIR = ROOT_DIR / "eval" / "dataset" / "dev" / "chunk"
ASSET_ROOT = ROOT_DIR / "eval" / "dataset" / "dev" / "mineru_output"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
CHUNK_FILES = {"text": "text_chunk.json", "image": "image_chunk.json", "table": "table_chunk.json"}
ANSWER_FILES = {"text": "text_chunk_result.json", "image": "image_chunk_result.json"}
EXTRACTION_FILES = {
    "text": ROOT_DIR / "eval" / "抽取评测" / "chunk抽取" / "text_extraction.json",
    "image": ROOT_DIR / "eval" / "抽取评测" / "chunk抽取" / "image" / "image_extraction.json",
    "table": ROOT_DIR / "eval" / "抽取评测" / "chunk抽取" / "table" / "table_extraction.json",
}
BASELINE_DIR = ROOT_DIR / "eval" / "抽取评测" / "chunk抽取" / "LLM抽取"
BASELINE_FILES: dict[str, Path] = {
    modality: BASELINE_DIR / f"{modality}_extraction.json"
    for modality in ("text", "image", "table")
}
ASSET_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class ChunkViewerHandler(BaseHTTPRequestHandler):
    """提供前端文件、Chunk JSON 和数据集图片资源。"""

    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        """发送响应内容并添加基础安全与缓存响应头。"""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value: object, status: int = 200) -> None:
        """将 Python 对象编码为 UTF-8 JSON 响应。"""
        self.send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def resolve_asset(self, raw_path: str) -> Path | None:
        """解析图片路径，并确保目标仍位于 MinerU 数据集目录内。"""
        if not raw_path:
            return None
        try:
            candidate = Path(raw_path).resolve(strict=True)
            candidate.relative_to(ASSET_ROOT.resolve(strict=True))
        except (OSError, RuntimeError, ValueError):
            return None
        if candidate.suffix.lower() not in ASSET_TYPES or not candidate.is_file():
            return None
        return candidate

    def do_GET(self) -> None:
        """按请求路由返回页面、Chunk、抽取答案或本地图片。"""
        request = urlsplit(self.path)
        params = parse_qs(request.query)
        if request.path in STATIC_FILES:
            filename, content_type = STATIC_FILES[request.path]
            try:
                self.send_bytes((FRONT_DIR / filename).read_bytes(), content_type)
            except OSError:
                self.send_json({"error": "页面资源读取失败"}, 500)
            return

        if request.path in {"/api/chunks", "/api/answers"}:
            modality = params.get("modality", [""])[0]
            files = CHUNK_FILES if request.path == "/api/chunks" else ANSWER_FILES
            filename = files.get(modality)
            if filename is None:
                self.send_json({"error": "未知数据类型或 Chunk 模态"}, 400)
                return
            try:
                body = (CHUNK_DIR / filename).read_bytes()
                json.loads(body)
            except (OSError, ValueError):
                self.send_json({"error": f"{filename} 无法读取"}, 500)
                return
            self.send_bytes(body, "application/json; charset=utf-8")
            return

        if request.path == "/api/extractions":
            modality = params.get("modality", [""])[0]
            method = params.get("method", [""])[0]
            if modality not in CHUNK_FILES or method not in {"algorithm", "baseline"}:
                self.send_json({"error": "未知抽取结果类型"}, 400)
                return
            path = EXTRACTION_FILES.get(modality) if method == "algorithm" else BASELINE_FILES[modality]
            if path is None:
                self.send_json({"error": "该模态尚无算法抽取结果"}, 404)
                return
            try:
                body = path.read_bytes()
                json.loads(body)
            except (OSError, ValueError):
                self.send_json({"error": f"抽取结果文件无法读取：{path.name}"}, 500)
                return
            self.send_bytes(body, "application/json; charset=utf-8")
            return

        if request.path == "/api/asset":
            asset = self.resolve_asset(params.get("path", [""])[0])
            if asset is None:
                self.send_json({"error": "图片不存在或路径不在数据集目录内"}, 404)
                return
            self.send_bytes(asset.read_bytes(), ASSET_TYPES[asset.suffix.lower()])
            return

        self.send_json({"error": "接口不存在"}, 404)


def main() -> None:
    """启动仅监听本机地址的 Chunk 数据可视化服务。"""
    parser = argparse.ArgumentParser(description="Chunk 数据集可视化")
    parser.add_argument("--port", type=int, default=8766, help="本地监听端口，默认 8766")
    parser.add_argument("--baseline-text", type=Path, help="纯 LLM 文本 Chunk 抽取结果 JSON")
    parser.add_argument("--baseline-image", type=Path, help="纯 LLM 图像 Chunk 抽取结果 JSON")
    parser.add_argument("--baseline-table", type=Path, help="纯 LLM 表格 Chunk 抽取结果 JSON")
    args = parser.parse_args()
    if args.baseline_text is not None:
        BASELINE_FILES["text"] = args.baseline_text
    if args.baseline_image is not None:
        BASELINE_FILES["image"] = args.baseline_image
    if args.baseline_table is not None:
        BASELINE_FILES["table"] = args.baseline_table
    server = ThreadingHTTPServer(("127.0.0.1", args.port), ChunkViewerHandler)
    print(f"打开 http://127.0.0.1:{args.port}")
    print("按 Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
