from __future__ import annotations

import base64
import mimetypes
import os
import re
import sys
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

current_dir = Path(__file__).resolve().parent
explicitkg_root = current_dir.parent.parent
if str(explicitkg_root) not in sys.path:
    sys.path.append(str(explicitkg_root))

from modalprocessor.BaseModel import BaseModel
from modalprocessor.images.GeoVlmEcaExtractor import GeoVlmEcaExtractor


class ImageModalProcessor(BaseModel):
    """Image modal processor using the GeoVLM-ECA extraction workflow."""

    def __init__(self, config_path: str = None, rate_limit_qps: Optional[float] = None):
        """初始化图片处理器，并让 VLM 与文本 LLM 共用请求速率上限。"""
        super().__init__(config_path, rate_limit_qps=rate_limit_qps)
        self.extractor = GeoVlmEcaExtractor(
            llm_json_fn=self.call_openai_json,
            vlm_json_fn=self._call_vlm_json,
        )

    def process_image(
        self,
        image_paths: Union[str, List[str]],
        caption: str,
        references: str,
        context: str,
    ) -> Dict[str, Any]:
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        image_inputs: List[Dict[str, str]] = []
        valid_paths: List[str] = []
        for image_path in image_paths:
            image_path = str(image_path or "").strip()
            if not image_path:
                continue

            if self._is_remote_image_path(image_path):
                # Attempt to download the image to avoid VLM provider blocking CDN URLs
                try:
                    resp = requests.get(
                        image_path,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                        timeout=10
                    )
                    resp.raise_for_status()
                    encoded_image = base64.b64encode(resp.content).decode("utf-8")
                    mime_type = resp.headers.get('Content-Type') or mimetypes.guess_type(image_path)[0] or "image/jpeg"
                    image_inputs.append({
                        "kind": "base64",
                        "value": encoded_image,
                        "mime_type": mime_type,
                    })
                    valid_paths.append(image_path)
                    continue
                except Exception as e:
                    print(f"下载远程图像失败 {image_path}: {e}")
                    # Fallback to URL if download fails (though likely will be skipped or fail later)
                    image_inputs.append({"kind": "url", "value": image_path})
                    valid_paths.append(image_path)
                    continue

            if not os.path.exists(image_path):
                print(f"在 {image_path} 路径未找到图像文件")
                continue

            encoded_image = self._encode_image(image_path)
            if not encoded_image:
                print(f"图像编码失败: {image_path}")
                continue

            image_inputs.append({
                "kind": "base64",
                "value": encoded_image,
                "mime_type": mimetypes.guess_type(image_path)[0] or "image/jpeg",
            })
            valid_paths.append(image_path)

        if not image_inputs:
            return {"error": "没有有效的图像输入"}

        return self.extractor.extract(
            image_paths=valid_paths,
            image_inputs=image_inputs,
            caption=caption or "",
            references=references or "",
            context=context or "",
        )

    @staticmethod
    def _is_remote_image_path(path: str) -> bool:
        try:
            parsed = urlparse(path.strip())
            return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
        except Exception:
            return False

    @staticmethod
    def _encode_image(image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as exc:
            print(f"编码图像 {image_path} 时出错: {exc}")
            return ""

    def _call_vlm_json(self, prompt: str, image_inputs: List[Dict[str, str]]) -> Any:
        """
        辅助方法：调用视觉语言模型(VLM)并返回JSON格式结果
        扩展BaseModel的功能以支持多模态输入（文本+图像）
        
        Args:
            prompt (str): 提示词文本
            image_inputs (List[Dict[str, str]]): 图像输入列表（支持URL和base64）
            
        Returns:
            Any: 模型返回的JSON解析结果，出错时返回空字典
        """
        # 构建包含文本提示
        content = [{"type": "text", "text": prompt}]
        
        # 添加所有图像（支持远程URL与base64）
        for image_input in image_inputs:
            if not isinstance(image_input, dict):
                continue

            kind = image_input.get("kind")
            value = image_input.get("value")
            if not value:
                continue

            if kind == "url":
                image_url = value
            else:
                image_url = f"data:image/jpeg;base64,{value}"

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            })

        # 构建包含文本和图像的多模态消息
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        try:
            # 调用OpenAI的多模态模型API
            self._rate_limit()
            resp = self.client.chat.completions.create(
                model=self.model_name_vl,    # 使用配置中的模型名称（需支持视觉功能）
                messages=messages,        # 多模态消息
                temperature=self.temperature,  # 生成温度（控制随机性）
                timeout=self.timeout,      # 请求超时时间
                extra_body={"enable_thinking": False},
            )
            # 提取模型返回的内容并解析为JSON
            content = resp.choices[0].message.content or ""
            return self.safe_json_loads(content)
        except Exception as e:
            # 捕获所有异常并打印错误信息
            print(f"调用视觉语言模型出错: {e}")
            return {}

    @staticmethod
    def _should_skip_remote_vlm(image_inputs: List[Dict[str, str]]) -> bool:
        if not image_inputs:
            return False
        for image_input in image_inputs:
            if not isinstance(image_input, dict) or image_input.get("kind") != "url":
                return False
            value = image_input.get("value") or ""
            try:
                host = urlparse(value).netloc.lower()
            except Exception:
                return False
            if host != "cdn-mineru.openxlab.org.cn":
                return False
        return True
