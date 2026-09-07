# Stage 05：开放抽取图的概念 Schema 对齐

实体映射采用“标准化去重/缓存 → 规则 → Schema 一级分类 → 类别内向量召回 → 高置信向量直连 → 批量 LLM 消歧”的保守流程；关系映射仍在实体完成后，按有向端点 Schema 查询 `SCHEMA_RELATION`。


本阶段读取 `output/stage_04_merged_extraction.json` 中每个 `graphs[]` 子图，先完成实体类型对齐，再按已对齐的有向端点类型查询 `SCHEMA_RELATION` 并完成关系对齐。所有原始字段均保留；新增字段不会覆盖 `type`、`relation_name`、属性或溯源。

表格抽取器预定义的结构类型（如 `TableCell`、`TableRow`、`Parameter`、`Unit`）以及图像抽取器预定义的辅助结构类型（如 `chart_axis`、`chart_trend`、`data_series`、`diagram_track`、`MapTheme`）不会再次执行 Schema 映射。它们及相连关系保留在结果图中，并以 `schema_alignment.status = "SKIPPED"` 标识，也不会进入未映射缺口报告。图像中的井、岩性、地层等领域实体仍正常参与 Schema 映射。

## 模块边界

- `schema_repository.py`：从 Neo4j 动态加载 `EntityConcept` 与 `SCHEMA_RELATION`；也支持相同结构的离线 JSON 快照。
- `entity_rules.py`：加载 `config/schema_alignment_rules.json`，按配置顺序执行高精度名称规则和原始类型别名。
- `category_router.py`：根据 `raw_type/type_zh/name/attributes` 做轻量一级分类；无命中或最高分并列时回退全 Schema。
- `semantic_retriever.py`：分别构造概念与实体 embedding 文本，批量计算向量并召回 Top-K。
- `entity_cache.py`：按名称、原始类型和中文类型去重，提供实体映射 JSON 持久化缓存。
- `llm_selector.py`：让 LLM 在候选集合和 `NONE` 中保守选择，按 `llm_batch_size` 批处理。
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

- `output/stage_05_schema_aligned_graph.json`
- `output/stage_05_unmapped_entity_types.json`
- `output/stage_05_unmapped_relations.json`

默认运行时还会在 `output/.schema_process/` 保存实体映射缓存和 Schema embedding 缓存。缓存按 Schema/规则/关键阈值哈希隔离；Schema 或运行参数变化后会自动失效。

可用 `--schema-json snapshot.json` 改为读取离线 Schema 快照。快照顶层字段为 `concepts` 和 `relations`，属性命名与 Neo4j 查询结果一致。

## 保守策略

- LLM 只能从 Top-K 候选与 `NONE` 中选择，候选外输出会转为 `UNMAPPED`。
- 一级类别路由只缩小向量召回范围，不直接决定最终实体类型；路由证据不足或类别并列时自动回退全 Schema。
- 向量 Top-1 相似度默认不低于 `0.92` 且与第二候选的 margin 默认不低于 `0.12` 时，直接以 `method = "vector_auto_accept"` 映射并跳过 LLM。
- LLM 置信度或向量相似度低于阈值时转为 `UNMAPPED`。
- 任一端点实体未映射时，不尝试关系 Schema 映射。
- 原始关系不是弱“相关”语义时，即使端点唯一候选是 `RELATED_TO` 也拒绝降级。
