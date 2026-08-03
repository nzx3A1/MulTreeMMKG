# 三维地层建模图抽取

本目录实现 `three_dimensional_stratigraphic_model` 子类型的完整抽取链：

1. 整图调用从图片、图题和正文参考联合识别三维模型、命名地层、井、断裂、分区、流体与上下文关系。
2. 层界冻结调用只识别命名地层的正面、侧面、顶面区域，并区分垂向薄层 `vertical_layer` 与横向岩相区 `lateral_facies_zone`，不允许提前判断岩性。
3. 岩性候选目录调用独立读取图例文字、颜色和纹理，也保留模型本体上明确写出的岩性词；地层代号不视为岩性证据。
4. 独立层界审查调用用已定位图例排除误框，补齐漏掉的命名层、内部薄层和横向岩相区；程序自动识别真实像素与 `0–1000` 归一化坐标。
5. 程序从各地层可见区域提取主底色，生成保留线纹的同色掩膜拼图，并按地层单元分批调用 VLM；颜色只划定归属范围，只有图例纹理、层内显式文字或跨面一致性才能证明岩性。
6. 同色批次逐项扫描一个地层单元内的全部图例纹理，可补回单层识别容易漏掉的 secondary/interbed 岩性；缺少 `matched_patterns`、低于阈值或颜色范围无法区分的候选不会进入 Graph。
7. 每个冻结层段继续使用“整图 + 多面局部放大 + 图例”的聚焦拼图独立识别，作为同色单元扫描的细粒度交叉证据。
8. 低于阈值、无可靠候选、存在冲突特征或过薄的层段自动触发独立盲审；两次分歧时追加裁决调用，不允许创造第三种岩性。
9. 全图一致性审查联合检查逐层结果和同色单元结果，只输出修正项及未解决对象，不重写完整结果；未识别内容保留为不确定性，不凭地质常识补齐。
10. 质量门通过后，内部层段只作为可追溯视觉证据保留在 `multipass_lithology`，不生成 `visual_layer_segment` 图节点；所有 `has_lithology` 均由具体 `stratigraphic_unit` 直接指向一种或多种岩性。
11. 只在相同 `column_id`、相同 `parent_unit_id` 内按 `order_bottom_to_top` 生成相邻地层的 `directly_overlies` / `directly_underlies`。
12. 图内明确空间关系与正文上下文关系分开处理；正文关系统一 `explicit=false`，并同时保存 `context_evidence` 和图内锚点。
13. 所有实体和关系都保存来源图片路径、图片 ID、Chunk ID、图内证据和证据范围；事件保持为空。

## 运行

从项目根目录执行：

```powershell
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONUTF8='1'
python -m src.extractors.image_extractor.stratigraphic_profile.three_dimensional_stratigraphic_model.batch
```

若真实响应已经记录，只需应用新确定性规则而不重复调用模型：

```powershell
python -m src.extractors.image_extractor.stratigraphic_profile.three_dimensional_stratigraphic_model.batch --rebuild-existing
```

默认输入为 `testData/stratigraphic_profile_subtype_mock.json`，只处理其中全部 `three_dimensional_stratigraphic_model` 记录。默认输出在 `result/`，包含一个批量主文件和每张图片唯一命名的独立结果。

同色识别默认每批处理 4 个地层单元，可用 `THREE_DIMENSIONAL_UNIT_COLOR_BATCH_SIZE` 调整；`THREE_DIMENSIONAL_UNIT_COLOR_ACCEPT_THRESHOLD` 控制候选入图阈值，`THREE_DIMENSIONAL_UNIT_COLOR_DISTANCE_THRESHOLD` 控制程序颜色掩膜的 RGB 距离。建议先调整批大小，不要为了增加召回而降低纹理证据阈值。

## 证据边界

- `visual`：图片中的标签、相交、包围、错断、箭头等直接证据。
- `visual_derived`：程序在同柱同父层级内按层序确定性推导的上下关系。
- `context`：图题或正文明确支持的三元组，始终标记为非显式关系。
- `manual_review`：针对具体图片的专项 OCR 复核；原始 VLM 响应和复核后响应同时保留，不覆盖原始证据。
