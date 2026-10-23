# 图片抽取与子图隔离

生产入口 `main.py` 和第四阶段 `InitExtractor` 默认先调用项目现有 VLM 检测原图，随后进行子图隔离，再交给 A01–A20 分类器和专业抽取器。原有执行方式无需变化：

```powershell
python -m src.extractors.image_extractor.main --input output/stage_03_document_summary.json --output output/stage_04_image_extraction.json
```

## 处理过程

1. 检测并排、上下、多行子图及嵌套位置图/放大图；即使多个子图属于相同类别，也分别拆分。
2. 按归一化检测框进行真实像素裁剪。嵌套图单独裁剪，同时在主图对应区域涂白。尽量保留坐标轴、标题、标尺和图例；共享图例需要扩大裁剪范围时，遮蔽范围内的其他主体。
3. 每张裁剪结果再次调用 VLM，确认只有一个独立面板且为单一类型。如果仍有子图，递归切分。单张综合地层表的多列、同一坐标轴的多条曲线不拆成独立子图。
4. 子图重新调用视觉分类器，禁止继承整图类别或外部分类型记录。已确认无需拆分的单图仍可使用外部分类型记录。
5. 按子图类别独立抽取，最后汇总回原图及所属 Chunk。同名实体按现有规则合并，关系和事件引用随之修复；不依据子图邻接位置编造跨子图关系。

检测或分类失败、不能确认单图单类型、越界/重复框、未隔离的重叠区域和超限分支均写入错误结果，不回退到混合原图抽取。几何校验失败会携带错误信息重新检测一次；合格兄弟子图继续处理。模型视觉判断仍有误差，`verified_single` 表示通过模型复核，不代表人工标注级保证。

## 最终汇总与来源

`graphs` 仍按原有接口每个 ImageChunk 返回一个 Graph；一个 Chunk 对应多张原图时，各原图分别保留汇总索引。

- `entities / relations / events`：子图抽取结果合并后的结构化结果。
- `metadata.extra.source_image_results`：按原图分组，记录原图 ID/路径、处理状态、子图 ID、子图数量和类别，以及最终 Graph 中的 `entity_ids / relation_ids / event_ids`。同名节点合并后这里引用合并后的 ID。岩石图片的文本描述也按原图保留在 `descriptions` 中。
- `metadata.extra.routes`：每个子图的路由、分类、状态和错误。
- `metadata.extra.panel_preprocessing`：原图检测及递归复核审计。
- 各图元素的 `metadata.image_panel`：原图、子图路径、相对原图的裁剪框及遮蔽框。同名实体合并记录保留各子图证据。

裁剪框采用 EXIF 朝向归一化后的原图像素坐标，`bbox_pixels` 为左上闭、右下开区间。专业抽取器已有坐标仍相对于当前输入子图，不能直接当作原图坐标。

输出目录的 `image_panels/<来源及文件内容哈希>/` 保存归一化原图、子图 PNG 及 `panels.json` 检测清单。旧版整图缓存，以及部分失败的结果，不能绕过本版子图检查；只有完成当前版本检查并满足原有实体/关系条件的结果才会复用。

## 程序接口

`extract_from_images(..., vlm_client=client)` 默认启用子图处理；可通过 `context_options` 设置：

| 选项 | 默认值 | 用途 |
| --- | --- | --- |
| `panel_output_dir` | 结果文件同目录的 `image_panels`，未指定结果文件时为 `output/image_panels` | 子图保存位置 |
| `panel_max_depth` | 3 | 最大递归切分深度 |
| `panel_max_panels` | 32 | 每张原图创建的子图总数上限，含中间节点 |
| `panel_max_calls` | 64 | 每张原图的检测/复核调用上限，不含后续分类和专业抽取 |
| `split_subfigures` | true | 离线测试或调用方明确提供独立单图时可关闭 |

地图和定量图表批处理便捷入口也默认启用，可通过 `split_subfigures=False` 显式关闭。`build_image_tasks` 保留原有离线展开/分类功能，不负责切图；需要子图处理时使用统一 `extract_from_images` 入口。
