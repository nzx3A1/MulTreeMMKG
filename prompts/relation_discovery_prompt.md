# 关系发现 Prompt

## 用途

供 `src/discovery/relation_discovery.py` 使用，在已有 schema 之外挖掘潜在新关系。
输入为文档全文/章节摘要 + 当前已对齐的实体集合，要求 LLM 给出候选关系及证据。

## 输入占位符

- `{paper_summary}`   文档全文摘要
- `{aligned_entities}` 已对齐实体列表
- `{existing_relations}` 当前 schema 中的关系类型
- `{allow_open}`      是否允许开放式关系（true / false）

## 输出格式

```json
{
  "candidate_relations": [
    {
      "head": "...",
      "relation": "...",
      "tail": "...",
      "evidence": "...",
      "confidence": 0.0
    }
  ]
}
```
