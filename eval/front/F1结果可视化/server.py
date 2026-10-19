"""为文本抽取 F1 报告提供本地页面和实时 JSON 接口。"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


FRONT_DIR = Path(__file__).resolve().parent
REPORT_PATH = FRONT_DIR.parents[1] / "抽取评测" / "F1计算" / "text_f1_report.json"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class ReportHandler(BaseHTTPRequestHandler):
    """只提供固定的页面文件和 F1 报告。"""

    def send_content(self, body: bytes, content_type: str, status: int = 200) -> None:
        """发送带有内容类型和禁用缓存头的响应。"""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """按固定路由返回静态文件或当前 F1 报告。"""
        path = urlsplit(self.path).path
        if path == "/api/report":
            try:
                body = REPORT_PATH.read_bytes()
                json.loads(body)
            except (OSError, ValueError):
                self.send_content("报告文件无法读取或 JSON 格式错误".encode("utf-8"), "text/plain; charset=utf-8", 500)
                return
            self.send_content(body, "application/json; charset=utf-8")
            return

        static_file = STATIC_FILES.get(path)
        if static_file is None:
            self.send_content(b"Not Found", "text/plain; charset=utf-8", 404)
            return
        filename, content_type = static_file
        try:
            body = (FRONT_DIR / filename).read_bytes()
        except OSError:
            self.send_content(b"File unavailable", "text/plain; charset=utf-8", 500)
            return
        self.send_content(body, content_type)


def main() -> None:
    """解析端口并启动仅监听本机的 HTTP 服务。"""
    parser = argparse.ArgumentParser(description="查看文本抽取 F1 评测报告")
    parser.add_argument("--port", type=int, default=8767, help="本地服务端口，默认 8767")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), ReportHandler)
    print(f"F1 结果页面：http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
