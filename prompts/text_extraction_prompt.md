# 文本实体关系抽取 Prompt

## 用途

供 `src/extractors/text_extractor.py` 调用，输入为多模态 chunk 的文本部分，
要求 LLM 在 `schema/entity_schema.json` 与 `schema/relation_schema.json` 约束下，
输出 JSON 形式的实体列表与关系列表。

## 输入占位符

- `{chunk_text}`    当前 chunk 的纯文本内容
- `{section_path}`  当前 chunk 所属章节路径（如“第 3 章 / 3.2 节 / 储层评价”）
- `{entity_schema}` 注入的实体类型定义
- `{relation_schema}` 注入的关系类型定义
- `{few_shot}` 可选的 few-shot 示例

## 输出格式

```json
{
  "entities": [
    {"name": "...", "type": "...", "attributes": {...}, "evidence": "..."}
  ],
  "relations": [
    {"head": "...", "relation": "...", "tail": "...", "evidence": "..."}
  ]
}
```
