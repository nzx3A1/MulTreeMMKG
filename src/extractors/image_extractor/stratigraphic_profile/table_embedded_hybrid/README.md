# 表格—图像嵌入混合型地层图

PP-StructureV3 提供原图像素几何，VLM 识别文字、图例和曲线轨道及其内容，确定性流水线装配知识图谱。

## 曲线图像属性

VLM 在整图语义识别阶段输出 `curve_tracks`（名称、单位、刻度等）和 `curve_observations`（层段定性响应）。流水线按测井曲线的 `track_id` 找到 PP 几何确定的完整语义轨道，将包含表头、刻度和全段曲线的矩形区域保存为 PNG。

同一轨道中多个测井曲线节点共享一张整轨图，不进行单条线分离。每个 `log_curve` 节点的 `attributes` 包含：

| 属性 | 含义 |
| --- | --- |
| `cropped_image_cache_path` | 裁剪 PNG 的绝对路径 |
| `cropped_image_bbox` | 原图像素框 `[left, top, right, bottom]`，右、下边界不包含 |
| `cropped_image_scope` | `full_track_including_header` |
| `cropped_image_coordinate_space` | `original_pixels` |
| `cropped_image_source_path` | 来源原图的绝对路径 |

图片写入 `data/cache/table_embedded_hybrid/visual_track_slices/`；可通过 `TABLE_VISUAL_TRACK_SLICE_CACHE_DIR` 指定缓存根目录。缓存文件名包含来源路径、文件大小、修改时间、处理版本、轨道 ID 和裁剪边界的摘要。原图或裁剪边界变化后使用新缓存；重复运行复用已有图片。

轨道缺失或边界非法会明确报错，不用整张原图代替裁剪结果。已有结果文件需重新经过 `TableEmbeddedHybridPipeline.run` 和图谱装配，才会获得这些属性。

## 图例识别与曲线处理边界

只有 `legend` 按左右最近文字轨道中实体较多的一侧投影切片，再调用 VLM 识别；数量相同选左侧。图例切片响应使用 `table_embedded_hybrid.legend_slice_description.v3`。

曲线不调用切片 VLM，也不执行本地像素追踪、刻度换算或样点聚合。相关实现已移除；整图 VLM 提供的曲线语义继续保留。
