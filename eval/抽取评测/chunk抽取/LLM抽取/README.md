# Chunk 抽取脚本与结果

本目录保留抽取入口、正式结果及图片抽取所需的来源图片、路径映射和续跑清单。

| 抽取方式 | 入口脚本 | 正式结果 |
| --- | --- | --- |
| 文本提示词抽取 | `text_extract.py` | `text_extraction.json` |
| 图片提示词抽取 | `image_extract.py` | `image_extraction.json` |
| 表格提示词抽取 | `table_extract.py` | `table_extraction.json` |
| 文本 Schema 约束抽取 | `text_schema_extract.py` | `text_schema_extrction.json` |

`text_schema_extrction.json` 沿用现有拼写，供可视化页面直接读取。

`image_extract.py` 保留原有 VLM 提示词抽取方法，可使用子图检测后选择性重跑：

```powershell
python image_extract.py --rerun-split-images --workers 3
```

脚本检测所有原图，但只重新抽取需要切分的样本；未切分样本保留旧 Graph。子图通过单图、单类型复核及分类后，分别使用原来的提示词抽取实体和关系，再合并回原样本 ID，合并同名实体并保留来源。不会切换成项目的专业抽取算法。

- `image_extraction.json`：最终结果，仍是一条样本对应一个 Graph。
- `image_extraction.panel_rerun.json`：逐原图检测审计、子图路径和状态、模型调用统计与续跑清单。
- `image_panels/`：正式结果或续跑清单引用的标准化原图、独立子图和遮蔽后的主图；应与结果一起保留。
- `image_path_overrides.json`：失效原图路径的修正映射及定位依据。
- Graph 中的 `metadata.extra.source_image_results`：各原图的子图 ID、分类、实体关系汇总索引。

完成的样本自动跳过，失败样本再次执行时重试。失败时旧 Graph 暂时保留，并在 `pending_sample_ids` 和 `panel_rerun.stale_previous_sample_ids` 中明确标识，不能视为本轮成功结果。

本地发现的失效图片路径通过 `image_path_overrides.json` 记录原路径、实际读取路径及 MinerU 定位依据，不改写评测输入，也不将新图伪装为旧文件。首次使用路径映射时传入：

```powershell
python image_extract.py --rerun-split-images --image-overrides image_path_overrides.json
```

后续续跑会复用已保存的映射；显式修改映射后，受影响样本会重新处理。

清理时可删除 `__pycache__/`、运行日志、已核对的旧结果备份和单次验证报告。逐图 `panels.json` 的审计及子图信息已完整并入总续跑清单时，也可删除重复报告；未被结果或续跑清单引用的裁剪图可清理。

新建图片重跑清单时，脚本会自动生成 `image_extraction.previous-时间.json`。结果及清单中的 `backup` 仅记录历史备份位置，清理备份不会影响正式结果加载和已有清单的续跑。
