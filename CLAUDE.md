# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目背景

石油地质领域多模态知识图谱（MulTreeMMKG）构建系统。输入为石油地质领域 PDF 论文，
经 MinerU 版面解析 → 章节结构恢复 → 多模态分块 → Schema 约束抽取（文本/表格/图像/公式）
→ 跨模态对齐与去重 → 关系发现 → 图谱融合与校验 → 写入 Neo4j，最终落地为可查询的多模态
知识图谱。详细架构设计见 `README_石油地质多模态知识图谱构建.md`。

## 常用命令

### 安装依赖

```bash
pip install -r requirements.txt
# MinerU 需单独按官方文档安装：https://github.com/opendatalab/MinerU
```

### 运行流水线

```bash
python run_pipeline.py --input data/raw_pdf/paper_001.pdf
python run_pipeline.py --input <pdf_or_dir> --from-stage 1 --to-stage 12
```

`run_pipeline.main()` 当前为占位（`NotImplementedError`），各阶段函数同样为骨架实现，
需要按阶段逐步落地。

### 运行测试

```bash
# 当前 tests/ 下均为占位测试（NotImplementedError）
pytest tests/                                  # 全部
pytest tests/test_parser.py                    # 单文件
pytest tests/test_parser.py::test_mineru_loader_returns_normalized_doc  # 单测试
```

## 流水线 12 阶段与对应文件

| 阶段 | 输出 JSON | 主要模块 |
| ---- | --------- | -------- |
| 01 MinerU 解析 | `stage_01_mineru_parse.json` | `src/parser/mineru_loader.py` |
| 02 章节树 | `stage_02_section_tree.json` | `src/parser/markdown_parser.py`、`src/parser/section_splitter.py` |
| 03 文档骨架图 | `stage_03_document_skeleton_graph.json` | `src/skeleton/document_skeleton_builder.py` |
| 04 章节摘要 | `stage_04_section_summary.json` | `src/summarizer/bottom_up_summarizer.py` |
| 05 多模态 chunk | `stage_05_modal_chunks.json` | `src/skeleton/chunk_builder.py` |
| 06 文本抽取 | `stage_06_text_extraction.json` | `src/extractors/text_extractor.py` |
| 07 表格抽取 | `stage_07_table_extraction.json` | `src/extractors/table_extractor.py` |
| 08 图像抽取 | `stage_08_image_extraction.json` | `src/extractors/image_extractor.py` |
| 09 公式抽取 | `stage_09_formula_extraction.json` | `src/extractors/formula_extractor.py` |
| 10 跨模态对齐去重 | `stage_10_modal_alignment_dedup.json` | `src/alignment/{entity_aligner,relation_aligner,deduplicator}.py` |
| 11 关系发现 | `stage_11_relation_discovery.json` | `src/discovery/relation_discovery.py` |
| 12 增强融合图 | `stage_12_enhanced_graph.json` | `src/graph/{graph_merger,graph_validator,graph_schema}.py` |
| — 最终图 | `final_graph.json` | `src/graph/neo4j_writer.py` 写入 Neo4j |

所有阶段产物统一写入 `output/`；运行日志写入 `logs/`。

## 架构与模块职责

### 分层结构

- **入口与编排**：`run_pipeline.py` 按阶段串行调用，每个阶段的产物 JSON 是断点续跑与回溯的最小单元。
- **`config/`**：全局配置（路径、并发、阶段开关、LLM/VLM/Embedding 服务地址、Neo4j 连接、Schema 版本与开关）。
- **`model/`**：Pydantic 数据模型层，统一对外 re-export。包括 `base`（`MMKGBaseModel`）、`document`（`Paper/Section/Block`）、`chunk`（`TextChunk/TableChunk/ImageChunk/FormulaChunk`）、`content`（`TextBlock/Table/Image/Formula`）、`entity/relation/event`、`extraction`、`alignment`、`skeleton`、`graph`、`pipeline`（`PipelineStage/StageStatus`）。MinerU 已直接产出 markdown 与内容清单，因此不再维护独立的 PDF 几何坐标模型。
- **`src/parser/`**：MinerU 输出归一化与 Markdown 解析。`mineru_loader.load_paper(paper_id)` 读取 `data/mineru_output/<paper_id>/` 下的 `content_list.json`、`layout.json`、`full.md` 及 `images/tables/formulas/` 子目录，归一化为内部 `Paper` 结构。
- **`src/skeleton/`**：构建文档骨架图（Paper → Section → Subsection）与多模态 chunk 窗口。
- **`src/summarizer/`**：自底向上摘要（chunk → 小节 → 节 → 章节 → 全文）。
- **`src/extractors/`**：四路多模态抽取器（`text/table/image/formula_extractor.py`），共享 `schema_constrained_extractor.constrained_extract()` 统一封装：注入 schema、解析 LLM JSON、用 `jsonschema` 校验。
- **`src/alignment/`**：跨模态实体对齐（名称归一化、Embedding 相似度、LLM 判定合并 cluster）、关系归一化与冲突消解、最终去重。
- **`src/discovery/`**：在已对齐的实体之上由 LLM 挖掘潜在新关系，是否允许并入 schema 由 `schema_config.ALLOW_OPEN_RELATIONS` 控制。
- **`src/graph/`**：图谱融合（合并实体/关系并附溯源边）、图谱校验（节点引用完整性、schema 一致性、必填项校验）、Neo4j 写入（先执行 `schema/schema_graph.cypher` 创建约束索引，再 `UNWIND + MERGE` 批量写入）。
- **`src/utils/`**：`llm_client.LLMClient`（OpenAI 兼容 chat/json）、`vlm_client.VLMClient`（图像 base64/URL/路径）、`embedding_client.EmbeddingClient`、`json_io`（自动创建父目录、UTF-8、`ensure_ascii=False`）、`logger.get_logger(name)`（基于 loguru）。
- **`schema/`**：领域 Schema 定义（`entity_schema.json` / `relation_schema.json` / `event_schema.json`），目前为 `_version: 0.1.0` 的空占位，抽取与校验都依赖其内容。`schema_graph.cypher` 是 Neo4j 约束与索引脚本（当前以注释形式列出，由 `neo4j_writer` 首次写入前启用）。
- **`prompts/`**：各阶段 LLM/VLM Prompt 模板，使用 `{chunk_text}` `{entity_schema}` 等占位符，由 `schema_constrained_extractor` 与各抽取器注入。
- **`tests/`**：单元测试骨架，覆盖 parser / extractor / alignment / neo4j_writer。

### 关键设计原则（README_*.md）

- **结构先行**：先恢复论文目录骨架再抽取，所有实体/关系都绑定 `Paper/Section/Chunk/图号/表号/公式号` 等证据字段。
- **Schema 约束**：抽取必须落在 `schema/*.json` 范围内，避免泛化无效词（如“研究/结果/方法/特征”）。
- **多模态分治**：四路独立抽取再统一融合。
- **阶段持久化**：每阶段独立 JSON，便于调试、回溯、评估与断点续跑。
- **统一图谱格式**：所有阶段图谱 JSON 采用统一的 `nodes` / `relations` / `metadata` 结构。

### 数据流向

```
data/raw_pdf/*.pdf
  → [MinerU] data/mineru_output/<paper_id>/
  → output/stage_01_mineru_parse.json
  → output/stage_02_section_tree.json
  → output/stage_03_document_skeleton_graph.json
  → output/stage_04_section_summary.json
  → output/stage_05_modal_chunks.json
  → output/stage_{06,07,08,09}_*_extraction.json
  → output/stage_10_modal_alignment_dedup.json
  → output/stage_11_relation_discovery.json
  → output/stage_12_enhanced_graph.json
  → output/final_graph.json
  → Neo4j (petrommkg-schema / petrommkg-document)
```

## 外部服务与配置

- **LLM / VLM**：`config/model_config.py` 中的 `APIConfig` 与 `VLMAPIConfig`，`API_BASE=https://api.minimaxi.com/v1`，`MODEL_NAME=MiniMax-M3`，`TIMEOUT_SECS=120`。
- **Embedding**：`VectorAPIConfig`，`BASE_URL=https://api.siliconflow.cn/v1/embeddings`，`MODEL_NAME=BAAI/bge-large-zh-v1.5`，`DIMENSIONS=1024`，`TIMEOUT_SECS=10`。
- **MinerU**：`url=https://mineru.net/api/v4/file-urls/batch`，需 `TOKEN`。
- **Neo4j**：`config/neo4j_config.py`，两个库：`petrommkg-schema`（存放抽取 schema）、`petrommkg-document`（存放抽取结果），`URI=bolt://localhost:7687`。`BATCH_SIZE / WRITE_TIMEOUT` 与是否启用约束自动创建待补。

**重要**：当前 `config/model_config.py` 与 `config/neo4j_config.py` 中包含明文 API Key / Token / 数据库密码，提交与共享前需迁移到环境变量或 `.env`（`python-dotenv` 已在 `requirements.txt` 中）。

## 现状与落地优先级

当前仓库为论文项目的**骨架阶段**，除 `run_pipeline.py` 外，所有模块函数均为 `raise NotImplementedError`，`schema/*.json` 为空列表，`model/*.py` 仅有一行注释占位，`tests/*.py` 同理。落地时应按以下顺序：

1. **`model/`**：先补齐 Pydantic 模型，作为后续所有模块的数据契约。
2. **`config/`**：把明文密钥迁出，定义 `settings` 单例（注释中已说明由 `from config.app_config import settings` 引用）。
3. **`src/utils/`**：实现 `LLMClient` / `VLMClient` / `EmbeddingClient` / `json_io` / `logger`，所有下游模块都依赖这几个客户端。
4. **`src/parser/`**：实现 MinerU 归一化与 Markdown 解析。
5. **`schema/*.json`**：定义石油地质领域实体/关系/事件类型，再展开 `src/extractors/` 与 `src/alignment/`。
6. **`src/graph/`** + Neo4j 写入与校验放最后（依赖前面所有阶段）。
