"""岩石与微观储集空间图片的 VLM 文本描述抽取器。"""

from .extractor import RockMicroExtractor
from .prompt import build_rock_micro_description_prompt

__all__ = ["RockMicroExtractor", "build_rock_micro_description_prompt"]
