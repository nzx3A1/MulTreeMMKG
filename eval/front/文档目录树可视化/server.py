"""为 Stage 03 文档目录树页面提供本地静态文件、文档列表和图片访问。"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


FRONT_DIR = Path(__file__).resolve().parent
DEV_DIR = FRONT_DIR.parents[1] / "dataset" / "dev"
DATA_ROOTS = (DEV_DIR / "mineru" / "_output", DEV_DIR / "mineru_output")
SUMMARY_NAME = "stage_03_document_summary.json"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def find_documents() -> list[dict[str, str]]:
    """扫描两个兼容的数据目录，只列出确实存在 Stage 03 结果的文件夹。"""
    documents = []
    for root_index, root in enumerate(DATA_ROOTS):
        if not root.is_dir():
            continue
        for folder in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            summary = folder / SUMMARY_NAME
            if not folder.is_dir() or not summary.is_file():
                continue
            try:
                data = json.loads(summary.read_text(encoding="utf-8"))
                title = data.get("document", {}).get("title") or folder.name
            except (OSError, ValueError, TypeError, AttributeError):
                title = folder.name
            documents.append({
                "id": f"{root_index}:{folder.name}",
                "folder": folder.name,
                "title": str(title),
                "source": str(root.relative_to(DEV_DIR)).replace("\\", "/"),
            })
    return documents


def resolve_folder(document_id: str) -> Path | None:
    """验证文档编号并限制访问范围为已配置目录的直接子文件夹。"""
    try:
        index_text, folder_name = document_id.split(":", 1)
        index = int(index_text)
        root = DATA_ROOTS[index]
    except (ValueError, IndexError):
        return None
    if not folder_name or folder_name in {".", ".."} or Path(folder_name).name != folder_name:
        return None
    if "/" in folder_name or "\\" in folder_name:
        return None
    folder = root / folder_name
    if folder.is_dir() and (folder / SUMMARY_NAME).is_file():
        return folder
    return None


class ViewerHandler(BaseHTTPRequestHandler):
    """处理页面静态资源与受限的本地数据接口。"""

    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        """发送指定类型的响应，并禁止浏览器缓存动态数据。"""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value: object, status: int = 200) -> None:
        """把接口结果编码成 UTF-8 JSON。"""
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_bytes(body, "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        """按路由读取静态文件、文档结果或文档图片。"""
        request = urlsplit(self.path)
        params = parse_qs(request.query)
        if request.path in STATIC_FILES:
            filename, content_type = STATIC_FILES[request.path]
            self.send_bytes((FRONT_DIR / filename).read_bytes(), content_type)
            return
        if request.path == "/api/documents":
            self.send_json({"documents": find_documents()})
            return
        document_id = params.get("id", [""])[0]
        folder = resolve_folder(document_id)
        if folder is None:
            self.send_json({"error": "文档不存在"}, 404)
            return
        if request.path == "/api/document":
            try:
                body = (folder / SUMMARY_NAME).read_bytes()
                json.loads(body)
            except (OSError, ValueError):
                self.send_json({"error": "文档 JSON 无法读取"}, 500)
                return
            self.send_bytes(body, "application/json; charset=utf-8")
            return
        if request.path == "/api/image":
            filename = params.get("file", [""])[0]
            if not filename or Path(filename).name != filename or "/" in filename or "\\" in filename:
                self.send_json({"error": "图片路径无效"}, 400)
                return
            image = folder / "images" / filename
            content_type = IMAGE_TYPES.get(image.suffix.lower())
            if not content_type or not image.is_file():
                self.send_json({"error": "图片不存在"}, 404)
                return
            self.send_bytes(image.read_bytes(), content_type)
            return
        self.send_json({"error": "接口不存在"}, 404)


def main() -> None:
    """启动仅监听本机的轻量 HTTP 服务。"""
    parser = argparse.ArgumentParser(description="Stage 03 文档目录树可视化")
    parser.add_argument("--port", type=int, default=8765, help="本地监听端口，默认 8765")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), ViewerHandler)
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
