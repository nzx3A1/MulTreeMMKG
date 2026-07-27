"""直接调用项目 VLM 客户端生成图片描述。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401 - 直接运行测试文件时注入项目根目录。

from config.model_config import settings
from src.utils.vlm_client import VLMClient


DEFAULT_IMAGE_PATH = Path(__file__).parent / "test_data" / "images" / "img.png"
DEFAULT_PROMPT = "请详细描述这张图片中的主要内容、文字信息和关键视觉特征。"


def get_image_description(
    image_path: str | Path = DEFAULT_IMAGE_PATH,
    prompt: str = DEFAULT_PROMPT,
) -> str:
    """中文说明：使用 config 中的 VLM 配置调用客户端，并返回图片描述文本。"""

    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在：{path}")

    vlm_client = VLMClient(config=settings.vlm)
    return vlm_client.describe_image(
        image_path=path,
        prompt=prompt,
        task_name="VLM 图片描述测试",
    )


def parse_args() -> argparse.Namespace:
    """中文说明：读取可选图片路径和提示词，未传参时使用仓库测试图片。"""

    parser = argparse.ArgumentParser(description="调用 config 中配置的 VLM 模型描述图片")
    parser.add_argument(
        "image_path",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help=f"待描述图片路径，默认：{DEFAULT_IMAGE_PATH}",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="发送给 VLM 的图片描述要求",
    )
    return parser.parse_args()


def main() -> None:
    """中文说明：调用图片描述函数，并将模型返回结果直接打印到终端。"""

    # 中文说明：VLM 回复可能包含 GBK 无法表示的符号，统一使用 UTF-8 输出。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    description = get_image_description(args.image_path, args.prompt)
    print(description)


if __name__ == "__main__":
    main()
