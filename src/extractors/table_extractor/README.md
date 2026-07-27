# 石油地质表格抽取模块

本模块将 Stage-02 中的表格或候选图片转换为规范 HTML、二维行列网格和统一 `Graph`，目标是优先保证石油地质表格的字段、记录和合并单元格语义正确。

## 处理流程

1. 输入适配：读取 `TableChunk`，也可通过 `--include-image-candidates` 纳入候选 `ImageChunk`。
2. 表格识别：HTML/Markdown 直接规范化；图片默认使用 RapidOCR + Unitable，失败时可由 `auto` 模式回退 MinerU。
3. HTML 修复：规范 `rowspan/colspan`，恢复多级表头，统一深度、TOC、渗透率、含气量等常见单位。
4. 长表恢复：当密集表尾部连续缺列时，以样品号、岩心编号、测点编号、井深或井号作为行锚点，执行原分辨率重叠分块 OCR，再按列中心重建完整数据行。
5. 二级表头归列：按石英、斜长石、方解石、白云石、黄铁矿、黏土矿物等领域词和 OCR 几何位置拆分被错误合并的子表头。
6. 质量门禁：检查空表、低非空率、OCR 内容覆盖、结构坐标一致性、稀疏尾部及含糊矿物子表头。未通过的表不会生成知识图谱。
7. 图谱装配：生成 `Table`、`TableRow`、`TableColumn`、`TableCell`、`TableHeader`、`Parameter`、`Unit` 等结构节点，并按领域主键建立语义实体。

石油地质横表的主键优先级为：样品/分析记录 > 井 > 岩性 > 储层或层位。样品实体会关联井、地层组、地层段、储层段和明确出现的岩性；只有表中存在岩性证据时才创建 `Lithology -> COMPOSED_OF -> Mineral`，避免凭空推断。

## 直接运行

在仓库根目录使用 `treeSchemeKG` 环境：

```powershell
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONNOUSERSITE='1'
& 'C:\Users\nzx\.conda\envs\treeSchemeKG\python.exe' `
  'src\extractors\table_extractor\table_extractor.py' `
  --input 'output\表格测试数据_stage_02.json' `
  --output 'output\表格测试数据_stage_04_table_extraction.json' `
  --recognition-output 'output\表格测试数据_stage_04_table_recognition.json' `
  --report-output 'output\表格测试数据_stage_04_table_tasks.json' `
  --work-dir 'output\表格测试数据_table_extraction' `
  --engine rapidtable --rapid-model unitable --device auto `
  --ocr-device cpu --ocr-backend onnxruntime --ocr-limit-side-len 1600 `
  --include-image-candidates
```

`table_extraction.json` 保存 Graph 列表，`table_recognition.json` 保存规范 HTML、单元格网格和质量证据，`table_tasks.json` 保存逐表状态。工作目录中的每个任务还会保存 `canonical.html`、`cells.json` 和 `quality.json`。

## 当前回归结果

- `表格测试数据_stage_02.json`：10/10 表格成功，全部 Graph Schema 校验通过。
- 两张密集长表：均恢复为 2 行表头 + 24 条数据，每条数据 15 列完整。
- 表格定向测试：19 项全部通过。
- 本机 Neo4j 当前未加载表格结构概念时，选择器使用 `config/schema_schema.cql` 中的结构 Schema 静态回退；数据库仅只读查询，不执行写入。

