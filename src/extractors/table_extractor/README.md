# 石油地质表格抽取模块

本模块将 Stage-02 中的表格或候选图片转换为规范 HTML、二维行列网格和统一 `Graph`，目标是优先保证石油地质表格的字段、记录和合并单元格语义正确。

## 处理流程

1. 输入适配：读取 `TableChunk`，也可通过 `--include-image-candidates` 纳入候选 `ImageChunk`。
2. 表格识别：HTML/Markdown 直接规范化；图片默认使用 RapidOCR + Unitable，失败时可由 `auto` 模式回退 MinerU。
3. HTML 修复：规范 `rowspan/colspan`，恢复多级表头，统一深度、TOC、渗透率、含气量等常见单位。
4. 视觉语义判断：图片表格在得到 HTML 后调用一次 VLM，识别横向/纵向记录方向、完整表头、行/列主题及第一条数据的位置；响应和降级原因保存到逐表 `visual_semantics.json`。
5. 合并单元格展开：严格按 HTML 的 `rowspan/colspan` 填满逻辑矩阵，不使用 VLM 猜测单元格坐标或数据值。
6. 长表恢复：当密集表尾部连续缺列时，以样品号、岩心编号、测点编号、井深或井号作为行锚点，执行原分辨率重叠分块 OCR，再按列中心重建完整数据行。
7. 二级表头归列：按石英、斜长石、方解石、白云石、黄铁矿、黏土矿物等领域词和 OCR 几何位置拆分被错误合并的子表头。
8. 质量门禁：检查空表、低非空率、OCR 内容覆盖、结构坐标一致性、稀疏尾部及含糊矿物子表头。未通过的表不会生成知识图谱。
9. 图谱装配：横向表逐行、纵向表逐列生成记录节点。优先使用原文主题字段值作为节点名；若没有可靠主键，则以表格任务 ID 与数据行号或列号生成稳定键，写入 `generated_key` 元数据。测量值仍按表头保存为属性，不会伪装成主键。当前记录模式不生成单独的表格结构层级或关系，Graph JSON 外层契约保持不变。

石油地质横表的原文主键优先级为：样品/分析记录 > 井 > 岩性 > 储层或层位。纯数值测量列不能作为主键，明确的样品编号、井号仍可使用数字编号。程序生成的记录键仅标识表内位置，不代表从论文中识别出了领域实体名称。

## 直接运行

在仓库根目录使用 `treeSchemeKG` 环境：

```powershell
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONNOUSERSITE='1'
& 'C:\Users\nzx\.conda\envs\treeSchemeKG\python.exe' `
  'src\extractors\table_extractor\table_extractor.py' `
  --input 'output\表格测试数据_stage_02.json' `
  --output 'output\表格测试数据_stage_04_table_extraction.json' `
  --work-dir 'output\表格测试数据_table_extraction' `
  --engine rapidtable --rapid-model unitable --device auto `
  --ocr-device cpu --ocr-backend onnxruntime --ocr-limit-side-len 1600 `
  --include-image-candidates
```

最终只保存一个 `table_extraction.json`，其顶层结构与文本抽取结果一致，包含 `_status`、`statistics` 和 `graphs`。工作目录仍保存逐表识别缓存和诊断产物，供断点复用及质量排查，不属于最终阶段 JSON。

## 当前回归结果

- `表格测试数据_stage_02.json`：10/10 表格成功，全部 Graph Schema 校验通过。
- 两张密集长表：均恢复为 2 行表头 + 24 条数据，每条数据 15 列完整。
- 表格定向测试：19 项全部通过。
- 本机 Neo4j 当前未加载表格结构概念时，选择器使用 `config/schema_schema.cql` 中的结构 Schema 静态回退；数据库仅只读查询，不执行写入。
