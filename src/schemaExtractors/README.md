# 基于 Schema 的文本抽取

本模块读取 [schema.md](schema.md) 中的实体类型和关系定义。实体类型与关系仍按本体校验；实体属性采用开放的中文名称，不受本体属性表限制。

## 运行

在项目根目录执行：

```powershell
python -m src.schemaExtractors.textTextractor.main --check-input
python -m src.schemaExtractors.textTextractor.main
python -m src.schemaExtractors.textTextractor.main --limit 2 --workers 1 --output output/schema_text_smoke.json
```

默认输入为 `output/stage_03_document_summary.json`，默认输出为 `output/stage_04_schema_text_extraction.json`。每个 Chunk 依次完成实体、关系和属性补充；不同 Chunk 可并发处理。成功结果按输入指纹恢复，提示词或校验规则变更会使旧检查点失效。

## 实体输出

```json
{
  "id": "entity_1792730783864496900",
  "name": "马家沟组储层",
  "official_name": "马家沟组储层",
  "type": "储层",
  "type_en": "Reservoir",
  "attribute": {
    "孔隙度": "8%",
    "厚度范围": "100～900 m",
    "孔隙类型": ["晶间孔", "溶蚀孔"]
  },
  "provenance": "马家沟组储层孔隙度为8%，厚度为100～900 m。"
}
```

- 正式实体 ID 由程序生成，格式为 `entity_` 加纳秒时间戳；模型实体阶段的 ID 仅用于临时识别。
- `attribute` 是唯一的实体属性容器。属性名须含中文，属性值可为非空字符串、数值、布尔值、列表或对象。区间、单位、条件及多组测量应完整保留。
- 实体不输出 `attributes` 或 `attribute_provenance`，属性补充阶段也无需逐项提交证据。实体自身的 `provenance` 仍须是当前 Chunk 的连续原文。
- 关系仍使用 `attributes` 保存关系限定词，关系自身的 `provenance` 仍须来自当前 Chunk。
- 统计中的 `entity_attribute_count` 汇总实体 `attribute` 的条目数；`attribute_completed_text_chunk_count` 统计完成属性补充阶段的 Chunk 数。

第三阶段只接受 `{"entity_attributes":[{"entity_id":"已有正式实体ID","attribute":{"孔隙度":"8%"}}]}` 格式；已有属性值冲突时保留原值并记录警告。

## 验证

```powershell
python -m pytest src/schemaExtractors/tests -q
```
