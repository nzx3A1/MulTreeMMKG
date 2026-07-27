# 表格嵌入混合图抽取结果

本目录保存真实模型运行 `table_embedded_hybrid` 抽取器生成的结果。JSON 中的
`subtype` 明确标记为 `table_embedded_hybrid`。

运行：

```powershell
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONUTF8='1'
python util\run_table_embedded_hybrid_live_single.py
python util\run_table_embedded_hybrid_live_batch.py
```

真实模型结果同时包含：

1. 坐标模型、轨道和专用解析图元组成的结构化中间结果；
2. 具备逐图来源、视觉证据和显式/推导标记的节点与关系；
3. 空事件列表，当前任务不扩展事件抽取。

- `table_embedded_hybrid_live_api_single_extraction_result.json`：单图真实模型抽取结果；
- `table_embedded_hybrid_live_batch_results.json`：批量真实模型逐图结果；
- `table_embedded_hybrid_live_batch_nodes.json`：批量结果中的全部节点；
- `table_embedded_hybrid_live_batch_relations.json`：批量结果中的全部关系。
