# Stage 05 postSchema 后处理

在执行完 Schema 映射后会得到未映射的实体和关系文件。本目录实现以下处理：

1. 读取 `output/stage_05_unmapped_entity_types.json` 与 `output/stage_05_schema_aligned_graph.json`。
2. 对每种未映射类型使用示例名称、属性和溯源做 LLM 一级 `ConceptCategory` 选择。
3. 在一级类别内做 LLM 二级概念选择。允许同义转换、实例/子类映射和合理上位抽象，例如储层参数平面分布图映射为 `GeologicalMap`。
4. 只有现有概念确实无法表达时才创建二级 `EntityConcept`。同义新增提议会在全局归并，避免重复扩充 Schema。
5. 新概念结构与原 `EntityConcept` 一致，生成 embedding，连接 `BELONGS_TO_CATEGORY`，并统一添加 `PostSchemaAdded` 标签和 `postSchema*` 审计属性。
6. 将主图实体的原 `type/type_zh` 替换为最终 Schema 类型，同时保留 `raw_type/raw_type_zh`。
7. 使用补映射后的有向端点查找现有 `SCHEMA_RELATION`：有唯一或精确候选时替换关系 `type/type_zh`；没有候选时保持原关系类型；原值保留在 `raw_relation/raw_type_zh`。

映射要求不是“只要相关就合并”。只有能自然表述为“该实体是一种/一个候选概念”时才复用；由某实验测得、能指示某对象、影响某对象或仅与其相关都不构成实体概念映射。

## 文件说明

- `models.py`：实体判定和新增概念模型。
- `llm_mapper.py`：两级 LLM 选择、非法批结果单条重试和新概念全局归并。
- `processor.py`：输入校验、计划生成、关系重算和主图类型替换。
- `schema_writer.py`：新概念向量化、Neo4j 单事务写入及写后验证。
- `mapping_overrides.json`：本次 93 个缺口实体的人工审查结论，只在输入 SHA-256 一致时生效。
- `main.py`：命令行入口。

## 安全运行

先生成计划和报告，不修改 Neo4j 与主图：

```powershell
C:\Users\nzx\.conda\envs\treeSchemeKG\python.exe -m src.schemaProcess.postSchema.main --dry-run
```

审查 `output/stage_05_post_schema_plan.json` 后，复用计划正式写入：

```powershell
C:\Users\nzx\.conda\envs\treeSchemeKG\python.exe -m src.schemaProcess.postSchema.main `
  --plan-input output/stage_05_post_schema_plan.json
```

正式运行会：

- 写入并验证新增 Neo4j 概念；
- 覆盖 `output/stage_05_schema_aligned_graph.json`；
- 自动保存 `stage_05_schema_aligned_graph.pre_post_schema_时间戳.json` 备份；
- 写出 `output/stage_05_post_schema_report.json`。

输入文件变化后旧计划和人工覆写表都会被指纹校验拒绝，必须重新生成计划。使用 `--no-overrides` 可只采用 LLM 结果；使用 `--schema-json` 和 `--skip-schema-write` 可做纯离线验证。

如果同一个 `raw_type` 内存在异质实例，覆写表还可通过 `entity_overrides` 按 `graph_index + entity_id` 精确修正单个实体；例如本次 `fluid` 组中的 `Mg2+` 单独映射为 `ChemicalIon`，不会随其余流体实例映射为 `GeologicalFluid`。
