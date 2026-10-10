# Stage 06 多模态实体与关系对齐

本阶段读取 `output/stage_05_schema_aligned_graph.json`，在每个文档内部把文本、
图片和表格中的同一事物融合为全局实体。实体判同以 `schema_type_zh`、规范化名称、
别名、属性、上下文、邻接结构和可选 Embedding 为证据；灰区候选可交给 LLM 只返回
`SAME / DIFFERENT / UNCERTAIN`。

实体融合完成后，所有原始关系端点都会通过
`(document_id, chunk_id, original_entity_id) -> global_entity_id` 映射重写。相同端点、
相同语义和相同限定条件的关系合并为一条全局关系，并在 `evidence` 中保留每个
原始 Chunk 的关系；不同关系和不同限定条件继续并列连接到融合节点。

默认运行：

```powershell
C:\Users\nzx\.conda\envs\treeSchemeKG\python.exe -m src.alignment.stage_06_multimodal_alignment
```

离线确定性基线：

```powershell
C:\Users\nzx\.conda\envs\treeSchemeKG\python.exe -m src.alignment.stage_06_multimodal_alignment --disable-embedding --disable-llm
```

输出包括主融合图、逐实体映射报告、未决候选报告和冲突报告。类型作用域、中文类型
兼容关系以及关系证据字段配置位于 `config/multimodal_alignment_rules.json`。
如需让 LLM 处理全部灰区候选，可增加 `--max-llm-candidates -1`；大样本建议先使用
默认上限检查一轮输出质量，再扩大处理范围。
