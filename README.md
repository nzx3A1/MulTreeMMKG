# 石油地质多模态知识图谱构建 (Petroleum Geology Multi-Modal Knowledge Graph)

## 项目简介

本项目从石油地质领域的 PDF 论文出发，借助 MinerU 工具完成版面解析，再经由
**章节切分 → 文档骨架构建 → 自底向上摘要 → 多模态抽取（文本 / 表格 / 图像 / 公式）→
跨模态对齐与去重 → 关系发现 → 图谱融合校验 → 写入 Neo4j** 的多阶段流水线，
构建一个面向石油地质领域的、多模态融合的知识图谱。

## 目录结构

```
petroleum-geology-mmkg/
├── README.md                 # 本文件
├── requirements.txt          # Python 依赖
├── run_pipeline.py           # 流水线主入口
├── config/                   # 全局配置（应用、schema、模型、Neo4j）
├── data/                     # 原始与解析数据（PDF / MinerU 输出）
├── schema/                   # 实体/关系/事件 schema 定义
├── src/                      # 核心源代码
│   ├── parser/               # 解析 MinerU 输出、Markdown、章节切分
│   ├── skeleton/             # 文档骨架与多模态 chunk 构建
│   ├── summarizer/           # 自底向上摘要
│   ├── extractors/           # 多模态实体关系抽取
│   ├── alignment/            # 跨模态实体关系对齐与去重
│   ├── discovery/            # 关系发现
│   ├── graph/                # 图谱融合、校验、写入 Neo4j
│   └── utils/                # LLM/VLM/Embedding 客户端、日志、IO
├── prompts/                  # 各阶段 LLM/VLM Prompt 模板
├── output/                   # 各阶段输出 JSON（流水线产物）
├── logs/                     # 运行日志
└── tests/                    # 单元测试
```

## 流水线阶段

| 阶段 | 输出文件 | 说明 |
| ---- | -------- | ---- |
| 01 | `stage_01_mineru_parse.json`       | MinerU 解析原始结果归一化 |
| 02 | `stage_02_section_tree.json`        | 章节树 |
| 03 | `stage_03_document_skeleton_graph.json` | 文档骨架图 |
| 04 | `stage_04_section_summary.json`     | 章节摘要 |
| 05 | `stage_05_modal_chunks.json`        | 多模态 chunk |
| 06 | `stage_06_text_extraction.json`     | 文本抽取 |
| 07 | `stage_07_table_extraction.json`    | 表格抽取 |
| 08 | `stage_08_image_extraction.json`    | 图像抽取 |
| 09 | `stage_09_formula_extraction.json`  | 公式抽取 |
| 10 | `stage_10_modal_alignment_dedup.json` | 跨模态对齐去重 |
| 11 | `stage_11_relation_discovery.json`  | 关系发现 |
| 12 | `stage_12_enhanced_graph.json`      | 增强融合图 |
| —  | `final_graph.json`                 | 最终图谱 |

## 快速开始

```bash
pip install -r requirements.txt
python run_pipeline.py --input data/raw_pdf/paper_001.pdf
```
