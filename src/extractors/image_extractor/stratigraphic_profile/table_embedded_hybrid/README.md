# 表格嵌入混合型地层图抽取流程

本文档对应当前 `table_embedded_hybrid` 实现。该分支用于处理同时包含地层表格、测井曲线、岩性柱、沉积相、储层或井标注等内容的复合地层图。

当前方案的核心原则是：**PP-StructureV3 负责原图像素几何，VLM 负责语义识别，Python 程序负责坐标换算、关系推导和知识图谱装配。** VLM 返回的像素坐标不会参与后续计算。

## 1. 适用范围与入口

上层图片流水线先完成图片大类分类。图片被路由到 `stratigraphic_profile` 后，再由地层子分类器判断为以下三类之一：

- `table_embedded_hybrid`：表格嵌入混合型；由本目录实现处理。
- `three_dimensional_stratigraphic_model`：三维地层模型；由独立子包处理。
- `two_dimensional_stratigraphic_log`：二维地层剖面或柱状图；由独立子包处理。

本模块的公开入口位于 `__init__.py`，主要包括：

- `extract_segmented_table_visual`：执行 PP-StructureV3 几何识别、三段 VLM 语义识别和一次节点官方名规范化。
- `TableEmbeddedHybridPipeline`：生成可审计的结构化中间结果。
- `build_table_embedded_hybrid_graph`：确定性装配统一 `Graph`。
- `PPStructureV3GeometryExtractor`：生成并缓存可信像素几何。

正常生产调用由 `StratigraphicProfileExtractor` 分发，不需要直接拼接这些步骤。

## 2. 最新完整流程

```text
原始图片
  │
  ├─ 1. VLM 图片大类分类
  │      └─ 路由到 stratigraphic_profile
  │
  ├─ 2. VLM 地层子分类
  │      └─ table_embedded_hybrid
  │
  ├─ 3. PP-StructureV3 单图几何识别
  │      ├─ OCR 文本及 rec_boxes
  │      ├─ table cell_box_list
  │      ├─ content_bbox、轨道、单元格、OCR 框和表格线
  │      └─ 稳定的 pp_track_* / pp_cell_* / pp_ocr_* ID
  │
  ├─ 4. 三段 VLM 语义识别
  │      ├─ layout
  │      ├─ stratigraphy_lithology
  │      └─ facies_reservoir_wells
  │
  ├─ 5. PP 几何回填
  │      └─ VLM 只选择几何 ID，程序解析 bbox、top_y、bottom_y、pixel_y
  │
  ├─ 6. 第四次 VLM 节点规范化
  │      ├─ 只处理前三段已有节点 ID，不新增节点、关系或坐标
  │      ├─ 使用领域知识、图题和正文上下文补充 official_name
  │      └─ 程序按 PP/VLM 轨道映射写入 track_header
  │
  ├─ 7. 纵轴拟合与结构化解析
  │      ├─ 绝对深度/高程/时间轴，或 relative_sequence
  │      ├─ 区间、点标记、曲线和对象规范化
  │      └─ 层级、相邻层序及跨轨道区间对齐
  │
  ├─ 8. 确定性知识图谱装配
  │      ├─ 稳定实体和关系 ID
  │      ├─ 来源、视觉证据、显式/推导标记
  │      └─ Graph 引用完整性校验
  │
  └─ 9. 结果质量门禁与 JSON 落盘
```

默认真实客户端支持分段抽取，因此一张目标图通常产生 6 次逻辑 VLM 调用：1 次图片大类分类、1 次地层子分类、3 次内容识别和 1 次节点官方名规范化。缓存命中的调用会复用已有响应，不一定产生新的网络请求。

## 3. PP-StructureV3 几何阶段

实现文件：`ppstructure_geometry.py`

### 3.1 输入与标准化

`PPStructureV3GeometryExtractor` 对原图执行一次 PP-StructureV3，主要读取：

- `overall_ocr_res.rec_boxes` 与 `rec_texts`；
- `table_res_list.cell_box_list`；
- PP 输出中的原始画布尺寸。

程序将结果标准化为 `ppstructurev3.table_geometry.v1`，包含：

- `content_bbox`：有效内容区域；
- `tracks`：按表格列边界构建的轨道；
- `cells`：表格单元格及其所属轨道；
- `ocr_lines`：OCR 文本、像素框和所属轨道/单元格；
- `rule_lines`：由单元格边界聚合出的纵向和横向表格线；
- `quality`：OCR 行数、单元格数、轨道数等统计；
- `runtime`：模型配置及 `fresh_inference`、`memory_hit` 或 `disk_hit` 缓存状态。

若 PP 未识别到可用单元格，程序会以内容区域构造保守的单轨道几何，但不会让 VLM补写像素坐标。

### 3.2 几何缓存

默认缓存目录为：

```text
data/cache/ppstructurev3
```

缓存键同时包含原图绝对路径、文件大小、修改时间和 PP 模型配置。图片或模型配置变化后会自然生成新的缓存项。

可用环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `TABLE_PPSTRUCTURE_CACHE_DIR` | `data/cache/ppstructurev3` | 磁盘缓存目录 |
| `TABLE_PPSTRUCTURE_CACHE` | `true` | 是否启用磁盘缓存 |
| `TABLE_PPSTRUCTURE_FORCE_REFRESH` | `false` | 设为 `1` 时忽略磁盘缓存并重新推理 |
| `TABLE_PPSTRUCTURE_DEVICE` | `cpu` | 例如 `cpu` 或 `gpu:0` |
| `TABLE_PPSTRUCTURE_LAYOUT_MODEL` | `PP-DocLayout-M` | 版面检测模型 |
| `TABLE_PPSTRUCTURE_TEXT_DET_MODEL` | `PP-OCRv5_mobile_det` | 文本检测模型 |
| `TABLE_PPSTRUCTURE_TEXT_REC_MODEL` | `PP-OCRv5_mobile_rec` | 文本识别模型 |
| `TABLE_PPSTRUCTURE_ENABLE_MKLDNN` | `false` | CPU 默认关闭 oneDNN/MKLDNN 兼容风险路径 |

## 4. 三段 VLM 语义识别

实现文件：`segmented_vlm.py`

三段固定顺序为：

```python
SEGMENT_ORDER = (
    "layout",
    "stratigraphy_lithology",
    "facies_reservoir_wells",
)
```

各段职责互斥，合并时若同一 `primitives` 字段被多个分段同时输出，程序会直接报错。

| 分段 | 负责内容 | 主要输出 |
|---|---|---|
| `layout` | 图名、版式、纵轴语义和轨道角色 | `diagram_id`、`layout_family`、`coordinate_system`、`tracks` |
| `stratigraphy_lithology` | 地层、参考井段、岩性、地质特征 | `stratigraphic_intervals`、`reference_intervals`、`lithology_intervals`、`geological_feature_intervals` |
| `facies_reservoir_wells` | 沉积相、储层、油层、曲线、井、其他对象和图中明确关系 | `facies_intervals`、`reservoir_intervals`、`oil_layer_intervals`、`curve_tracks`、`curve_observations`、`point_markers`、`objects`、`explicit_relations` |

第二、三段会收到前序分段已识别的轨道或地层锚点，以减少同一对象使用不同 ID 的情况。

VLM 的约束如下：

- 只从 PP 几何目录中选择 `track_id`、`geometry_refs` 和 `calibration_ocr_ids`；
- 不得输出或估算 `content_bbox`、`bbox`、`pixel_y`、`top_y`、`bottom_y`；
- 不确定内容写入 `uncertainties`，不能凭纹理猜测岩性或地质事实；
- `explicit_relations` 只能连接本次已经抽取出的局部对象 ID。

分段最大输出长度可通过以下变量调整：

| 变量 | 默认值 |
|---|---:|
| `STRATIGRAPHIC_TABLE_LAYOUT_MAX_TOKENS` | `4096` |
| `STRATIGRAPHIC_TABLE_GEOLOGY_MAX_TOKENS` | `8192` |
| `STRATIGRAPHIC_TABLE_RESERVOIR_MAX_TOKENS` | `8192` |
| `STRATIGRAPHIC_TABLE_NODE_ENRICHMENT_MAX_TOKENS` | `12288` |

前三段完成 PP 几何回填后，第四次调用接收已确认节点、图题、正文引用和轨道表头。响应必须逐一覆盖已有局部 ID，只能填写 `official_name`、规范化依据和置信度；缺失、重复或未知 ID 会直接报错。`track_header` 不由第四次调用自由生成，而是由程序根据节点 `track_id` 与 PP 几何引用确定。

对于不声明 `supports_segmented_table_extraction` 的自定义客户端，外层仍保留一次完整 VLM 抽取调用的兼容路径，输出上限由 `STRATIGRAPHIC_PROFILE_VLM_MAX_TOKENS` 控制，并在其后执行同一节点官方名规范化调用。真实运行脚本和项目默认 `VLMClient` 均使用三段抽取加一次规范化流程。

## 5. 坐标回填与纵轴处理

VLM 语义分段合并为 `table_embedded_hybrid.v1` 后，`apply_ppstructure_geometry` 会执行以下操作：

1. 将 VLM 轨道角色映射到 PP 轨道 ID；
2. 根据 `geometry_refs` 查找单元格或 OCR 框；
3. 由这些 PP 框的并集计算区间的 `top_y`、`bottom_y` 或点标记的 `pixel_y`；
4. 写入 `coordinate_source: "PP-StructureV3"`；
5. 写入 `geometry_policy.vlm_pixel_coordinates_used: false`。

纵轴分为两种情况：

### 5.1 连续绝对纵轴

当轴类型是 `depth`、`elevation` 或 `time`，并且深度轨道中至少存在两个跨度足够、方向单调的可解析 OCR 刻度时，程序用最小二乘法拟合：

```text
vertical_value = slope × pixel_y + intercept
```

拟合结果记录 `slope`、`intercept`、`rmse`、轴单位和刻度锚点。所有区间与点标记随后都通过同一变换换算到公共纵轴。

### 5.2 相对层序

如果只有逐层厚度数字、刻度不足、刻度不单调或无法可靠判断绝对轴，程序退化为：

```json
{
  "kind": "relative_sequence",
  "unit": "relative",
  "increases": "downward"
}
```

内容顶部和底部分别映射到 `0.0` 与 `1.0`。这样仍可表达上下层序和区间重叠，但不会伪造绝对深度。

## 6. 中间结果与关系生成

实现文件：`pipeline.py`

`TableEmbeddedHybridPipeline.run` 输出 `table_embedded_hybrid.intermediate.v1`，主要字段为：

- `source`：文档、Chunk、图片 ID、图片序号、路径和图注；
- `diagram`：图名、版式、真实图片尺寸；
- `coordinate_system`：轴类型、内容框、纵轴变换和坐标来源；
- `ppstructure_geometry`：本图标准化 PP 几何；
- `tracks`：经过边界校验的语义轨道；
- `parsed`：规范化区间、曲线、井标记和对象；
- `alignment_relations`：层级、层序、纵轴对齐和图中明确关系；
- `quality`：数量统计、轴拟合误差、未解析轨道、丢弃关系和不确定性。

程序生成关系时遵循以下规则：

- `part_of`：仅由明确的 `parent_id` 生成，标记为显式关系；
- `directly_overlies`：只连接同一父层下纵轴相邻且间隙不超过阈值的层；
- `has_lithology`、`has_sedimentary_facies`、`contains_reservoir`、`contains_oil_layer`、`contains_geological_feature`、`characterizes`：由公共纵轴区间重叠推导，重叠比例阈值为 `0.15`；
- 若父层和更具体的直接子层同时匹配，优先连接子层，避免父子重复挂接；
- 点标记优先对齐储层，否则对齐所在的地层/参考区间；
- VLM 明确关系的端点必须存在，无法解析的关系进入 `quality.dropped_explicit_relations`，不会创建虚构节点。

## 7. 知识图谱装配与来源追踪

实现文件：`graph.py`

图谱装配阶段不再调用大模型。程序为剖面、地层、岩性、相、储层、油层、地质特征、曲线响应、测井曲线、井和其他可见对象生成跨运行稳定的 ID，并将中间结果中的关系转换为统一 `Relation`。

每个实体和关系都必须携带：

- `source_modality: image`；
- `source_image_path`；
- `source_image_id`；
- `source_chunk_id`；
- `image_index`；
- 非空 `provenance` 和 `visual_evidence`；
- 关系的 `explicit` 与 `inference_basis`。

当前分支不抽取事件，因此 `Graph.events` 必须为空。

## 8. 运行方式

### 8.1 环境准备

在项目根目录执行：

```powershell
conda activate treeSchemeKG
python -m pip install "paddleocr[doc-parser]==3.7.0" paddlepaddle==3.3.1
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONUTF8='1'
```

VLM 的地址、模型和密钥继续使用项目现有配置；本模块不会把 API Key 或图片 Base64 写入结果文件。

### 8.2 单图真实运行

```powershell
python util\run_table_embedded_hybrid_live_single.py
```

默认结果：

```text
src/extractors/image_extractor/stratigraphic_profile/table_embedded_hybrid/result/
  table_embedded_hybrid_live_api_single_extraction_result.json
```

可选参数：

```powershell
python util\run_table_embedded_hybrid_live_single.py `
  --output path\to\single_result.json `
  --no-resume
```

不传 `--no-resume` 时，脚本会按任务名、模型、图片路径、Prompt 哈希和响应内容复用已有的成功调用。

### 8.3 批量真实运行

```powershell
python util\run_table_embedded_hybrid_live_batch.py
```

默认输入：

```text
src/extractors/image_extractor/stratigraphic_profile/testImage/
  image_chunks_stratigraphic_profile.json
```

默认输出：

```text
src/extractors/image_extractor/stratigraphic_profile/table_embedded_hybrid/result/
  table_embedded_hybrid_live_batch_results.json
  table_embedded_hybrid_live_batch_nodes.json
  table_embedded_hybrid_live_batch_relations.json
```

自定义运行示例：

```powershell
python util\run_table_embedded_hybrid_live_batch.py `
  --source path\to\image_chunks.json `
  --output path\to\batch_results.json
```

仅预检大类和子分类：

```powershell
python util\run_table_embedded_hybrid_live_batch.py --classification-only
```

忽略已有断点并重新调用：

```powershell
python util\run_table_embedded_hybrid_live_batch.py --no-resume
```

批量脚本只对目标子类型执行完整抽取，非目标图片会保留分类状态，不会被错误送入本抽取器。

## 9. 结果质量门禁

单图结果只有同时满足以下条件才标记为 `completed`：

- 已得到合法的 `table_embedded_hybrid.v1` 语义结果；
- 已生成结构化中间结果；
- 图谱至少包含实体；
- 没有模型错误；
- `Graph.validate_references()` 无错误；
- 每个实体和关系通过逐图来源校验；
- 没有未解析端点导致的 `dropped_explicit_relations`；
- 事件数量为 `0`。

否则单图脚本标记为 `failed_quality_gate`；批量脚本会将相应图片标记为失败，并在 `validation` 中保留具体原因。

重点审计字段：

```text
structured_intermediate_result.coordinate_system
structured_intermediate_result.ppstructure_geometry.runtime
structured_intermediate_result.quality.axis_rmse
structured_intermediate_result.quality.unresolved_track_ids
structured_intermediate_result.quality.dropped_explicit_relations
structured_intermediate_result.quality.uncertainties
graph.metadata.extra.quality
validation
```

其中 `coordinate_system.vlm_pixel_coordinates_used` 和 `quality.vlm_pixel_coordinates_used` 均应为 `false`。

## 10. 测试

无需真实 API 的重点回归测试：

```powershell
$env:PYTHONPATH=(Get-Location).Path
pytest -q tests\test_table_embedded_hybrid_ppstructure.py
pytest -q tests\test_stratigraphic_profile_subclassifier.py
pytest -q tests\test_image_extractor_architecture.py
```

`test_table_embedded_hybrid_ppstructure.py` 重点验证：

- PP 原始 OCR/单元格字段的标准化；
- 同图几何缓存复用；
- PP 坐标完全替换 VLM 坐标；
- 逐层厚度不会生成伪绝对深度；
- 三段 VLM 只选择 PP 几何 ID，第四次 VLM 只能规范化已有节点官方名。

上述测试不等同于真实模型运行证据。需要确认 PP 模型下载、VLM 服务、真实响应质量和结果落盘时，仍应执行单图或批量真实运行脚本。

## 11. 文件职责

| 文件 | 职责 |
|---|---|
| `ppstructure_geometry.py` | PP-StructureV3 调用、几何标准化、缓存、ID 目录、纵轴 OCR 和坐标回填 |
| `segmented_vlm.py` | 三段抽取 Prompt、节点官方名规范化 Prompt、VLM 调用、响应校验和字段所有权合并 |
| `prompt.py` | 非分段客户端使用的完整语义抽取 Prompt |
| `layout.py` | 纵轴拟合、表格线检测和轨道重建 |
| `pipeline.py` | 图元规范化、纵轴换算、区间对齐、层级和层序关系生成 |
| `graph.py` | 稳定 ID、实体/关系装配、来源元数据和引用校验 |
| `batch.py` | 单图实体与关系的来源完整性校验 |
| `result/` | 单图、批量、节点和关系运行结果 |

## 12. 当前边界

- 不使用 mock 结果作为生产输入。
- 不让 VLM 生成或修补像素坐标。
- 不把逐层厚度误当作连续绝对深度。
- 不依据横向接近程度猜测跨轨道关系。
- 不为无法解析的显式关系端点创建新实体。
- 不执行额外的关系审查 LLM。
- 不抽取事件。
- `table_embedded_hybrid` 与另外两个地层子类型保持独立实现，不进行通用算法回退。
