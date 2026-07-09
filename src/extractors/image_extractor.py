"""图像抽取器。

调用 VLM 读取论文中的图像（地质剖面图、显微照片、井位图等），
输出图像描述、所含实体与关系。

输出：stage_08_image_extraction.json
对应阶段：08 图像抽取
"""
from __future__ import annotations


def extract_from_images(images: list[dict], vlm_client, schema_cfg) -> list[dict]:
    """调用 VLM 从图像中抽取实体与关系。"""
    raise NotImplementedError
