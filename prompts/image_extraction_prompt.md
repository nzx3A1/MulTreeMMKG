# 图像实体关系抽取 Prompt

## 用途

供 `src/extractors/image_extractor.py` 与 `src/utils/vlm_client.py` 配合使用，
向 VLM 提供图像（地质剖面图、显微照片、井位图、图表等），
要求其在 schema 约束下输出图中所含实体与关系。

## 输入占位符

- `{image}`         base64 编码 / 路径 / URL
- `{image_caption}` 图题与上下文段落
- `{entity_schema}` 注入的实体类型定义
- `{relation_schema}` 注入的关系类型定义

## 输出格式

```json
{
  "image_description": "...",
  "entities": [...],
  "relations": [...]
}
```
