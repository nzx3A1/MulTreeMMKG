# 公式抽取 Prompt

## 用途

供 `src/extractors/formula_extractor.py` 使用，
输入为 LaTeX / MathML 形式的地质相关公式（达西定律、相渗曲线、孔隙度计算等），
要求 LLM 解析公式符号、物理含义并抽取出与其它实体的关系。

## 输入占位符

- `{formula_latex}`  公式的 LaTeX 表达式
- `{formula_context}` 公式所在段落上下文
- `{entity_schema}`   注入的实体类型定义

## 输出格式

```json
{
  "formula_id": "...",
  "latex": "...",
  "symbols": [{"symbol": "k", "meaning": "渗透率", "unit": "mD"}],
  "related_entities": [...],
  "description": "..."
}
```
