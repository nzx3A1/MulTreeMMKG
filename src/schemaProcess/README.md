# Stage 05：开放抽取图的概念 Schema 对齐

本阶段读取 `output/stage_04_merged_extraction.json` 中每个 `graphs[]` 子图，先完成实体类型对齐，再按已对齐的有向端点类型查询 `SCHEMA_RELATION` 并完成关系对齐。所有原始字段均保留；新增字段不会覆盖 `type`、`relation_name`、属性或溯源。

## 模块边界

- `schema_repository.py`：从 Neo4j 动态加载 `EntityConcept` 与 `SCHEMA_RELATION`；也支持相同结构的离线 JSON 快照。
- `entity_rules.py`：加载 `config/schema_alignment_rules.json`，按配置顺序执行高精度名称规则和原始类型别名。
- `semantic_retriever.py`：分别构造概念与实体 embedding 文本，批量计算向量并召回 Top-K。
- `llm_selector.py`：让 LLM 在候选集合和 `NONE` 中保守选择，支持批处理。
- `entity_aligner.py`：规则优先，未命中时执行向量召回与 LLM 判断。
- `relation_aligner.py`：严格按有向端点查询候选；多候选必须结合原始关系语义消歧。
- `reports.py`：构造实体类型和关系模式的 Schema Gap 报告。
- `pipeline.py`：保持原图结构，编排全量实体/关系对齐并写出三个文件。

## 运行

先保证 Neo4j 数据库 `petrommkg-schema`、Ollama embedding 服务和项目 LLM 服务可用：

```powershell
python -m src.schemaProcess.stage_05_schema_alignment
```

默认输出：

- `output/schema_aligned_graph.json`
- `output/unmapped_entity_types.json`
- `output/unmapped_relations.json`

可用 `--schema-json snapshot.json` 改为读取离线 Schema 快照。快照顶层字段为 `concepts` 和 `relations`，属性命名与 Neo4j 查询结果一致。

## 保守策略

- LLM 只能从 Top-K 候选与 `NONE` 中选择，候选外输出会转为 `UNMAPPED`。
- LLM 置信度或向量相似度低于阈值时转为 `UNMAPPED`。
- 任一端点实体未映射时，不尝试关系 Schema 映射。
- 原始关系不是弱“相关”语义时，即使端点唯一候选是 `RELATED_TO` 也拒绝降级。
