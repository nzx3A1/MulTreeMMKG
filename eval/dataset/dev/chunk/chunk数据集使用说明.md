# Dev Chunk 数据集说明

## 来源文档

候选 Chunk 来源于以下 5 篇文档的 Stage 02 文档树：`File1039`、`File1066`、`File108`、`File460`、`File565`。

候选池包含文本 Chunk 127 个、图像 Chunk 57 个、表格 Chunk 13 个、公式 Chunk 1 个。本轮计划抽取文本 30 个、图像 15 个、表格 8 个，共 53 个；公式暂不纳入。

## Chunk 结构

所有模态使用相同结构。`sample_id` 由文档 ID 和原始 Chunk ID 组成，格式为 `<document_id>::<chunk_id>`。

```json
{
  "sample_id": "File1039::2.1:image:0",
  "order": 2,
  "section": {
    "section_id": "2.1",
    "section_title": "地层特征",
    "section_level": 3,
    "parent_section_id": "2"
  },
  "content": {}
}
```

字段含义：

| 字段 | 含义 |
|---|---|
| `sample_id` | 样本唯一 ID，格式为 `<document_id>::<chunk_id>`。 |
| `order` | Chunk 在所属章节中的顺序，沿用 Stage 02 原值。 |
| `section.section_id` | 所属章节 ID。 |
| `section.section_title` | 所属章节标题。 |
| `section.section_level` | 章节层级。 |
| `section.parent_section_id` | 父章节 ID；没有父章节时为 `null`。 |
| `content` | Chunk 内容，按模态填写。 |

## 各模态的 content

- **文本**：`{"text": "文本内容"}`，内容来自 Stage 02 的 `text`。
- **图像**：`{"image_path": ["图片路径"], "caption": "图注", "references": null}`，字段来自 Stage 02 同名字段。
- **表格**：`{"markdown": "表格内容", "caption": "表题", "table_path": [], "references": null}`，字段来自 Stage 02 同名字段。
- **公式**：本轮不抽取；如后续纳入，使用 `{"formula": "公式内容"}`。

章节信息和内容均从对应文档的 `stage_02_document_tree.json` 中读取；资源路径沿用原 Chunk 路径。

## 已选样本

| 文件 | 数量 | 筛选重点 |
|---|---:|---|
| `text_chunk.json` | 30 | 五篇文档各 6 个；正文长度 304–831 字符，均含指代词和地质术语。 |
| `image_chunk.json` | 15 | 覆盖地图、地层柱、剖面、地震、测井、露头、薄片、统计图和演化示意图等视觉类型。 |
| `table_chunk.json` | 8 | 覆盖粒度参数、地层演化、年代测定、地球化学、压力实验、样品信息与物源对比等表格。 |

这批 Stage 02 表格的 `markdown` 字段实际是表格图片路径。复杂表格按图片中可见的分组表头、合并单元格和多列结构挑选；目前没有可直接解析的嵌套 HTML 表格。

图像按可见主体初步目视筛选，覆盖 A01–A11、A14–A17 等 15 种分类形态；这些类别只用于检查抽样多样性，不是图像分类金标准。
