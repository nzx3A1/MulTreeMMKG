# 二维平面地层—测井图抽取

本目录只处理 `two_dimensional_stratigraphic_log` 子类型，包括二维构造/地层剖面、地震时间剖面、单柱层序模式和多井测井对比图。

处理边界：

1. VLM 抄录图中实体、图内证据、归一化位置、从上到下层序、从左到右空间组和显式交切/对比关系。
2. 程序只在层序中相邻的两个节点之间生成 `directly_overlies` 和 `directly_underlies`，不生成跨层的 `above` / `below`。
3. 横向组只保留为版式审计信息，不生成 `left_of`、`right_of` 或 `adjacent_to` 三元组。
4. 交切、包含、跨井对比、轨迹沿层、岩性和测井曲线等业务关系继续保留。
5. 除图片/Chunk 根节点外，没有任何业务关系的孤立实体在 Graph 装配前删除。
6. 每个实体和关系都写入来源图片路径、图片 ID、Chunk ID、图片序号和视觉证据；不抽取事件。

批量入口：

```powershell
$env:PYTHONPATH=(Get-Location).Path
python util\run_two_dimensional_stratigraphic_log_live_batch.py
```

默认读取 `testImage/stratigraphic_profile_subtype_mock.json`，只处理其中 7 个二维目标 Chunk，并把主结果、节点侧车和关系侧车写入本目录的 `result` 文件夹。
