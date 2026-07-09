# 表格实体关系抽取 Prompt

## 用途

供 `src/extractors/table_extractor.py` 调用，输入为表格的 Markdown / JSON 表示，
要求 LLM 在 schema 约束下识别表头语义并抽取实体与关系
（如：油田-产层-储集参数、井号-层位-孔隙度 等）。

## 输入占位符

- `{table_repr}`     表格的字符串表示（Markdown / JSON）
- `{table_caption}`  表格标题
- `{entity_schema}`  注入的实体类型定义
- `{relation_schema}` 注入的关系类型定义

## 输出格式

```json
{
  "entities": [...],
  "relations": [...],
  "table_summary": "..."
}
```
