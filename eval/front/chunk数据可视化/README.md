# Chunk 数据集可视化

本地运行页面，浏览 `eval/dataset/dev/chunk` 下的文本、图像和表格 Chunk。

```powershell
python eval/front/chunk数据可视化/server.py
```

打开 <http://127.0.0.1:8766>。也可以使用 `--port` 指定端口。页面支持按模态切换、搜索样本、浏览前后样本，并显示章节信息、文本正文、图像资源和表格内容。

表格页面会按表格 Chunk 的 `sample_id` 和 Markdown 内容，在 MinerU `content_list_v2` 中匹配原表格图片并优先展示；找不到图片时显示 Chunk 中的表格内容。图片路径由本地接口补充，不改写 Chunk 数据集文件。

文本 Chunk 按 `sample_id` 关联 `text_chunk_result.json`。答案中能在正文匹配的实体标为红色，并显示实体和关系列表；匹配时忽略空白，未匹配到正文的实体显示为灰色。服务只监听本机地址，图片资源仅允许读取 MinerU 输出目录中的图片。

图像 Chunk 按 `sample_id` 关联 `image_chunk_result.json`，在对应图片组下方显示抽取实体。同一 Chunk 包含多张图片时，实体答案在图片组下方展示一次。

文本和图像 Chunk 的算法图谱分别读取 `eval/抽取评测/chunk抽取/text_extraction.json` 与 `eval/抽取评测/chunk抽取/image/image_extraction.json`；表格算法结果读取 `eval/抽取评测/chunk抽取/table/table_extraction.json`。纯 LLM 图谱默认读取 `eval/抽取评测/chunk抽取/LLM抽取` 下的文本、图像和表格 `*_extraction.json`。文本和图像页面对照原始内容、算法图谱、纯 LLM 图谱及标准答案；表格页面对照原表格、算法图谱和纯 LLM 图谱（当前没有表格标准答案）。表格算法结果按实体节点展示；点击节点可在图谱下方查看该记录的全部抽取字段。未抽取到实体的 Chunk 如实显示空图谱。顶部工具栏可切换上下或左右布局，调整原始内容宽度，并单独隐藏或恢复可用窗口；原始内容和图谱的高度可通过滑块或窗口底边拖拽调整。布局设置保存在本机浏览器中。

文本额外显示“Schema 约束抽取”卡片，默认读取 `eval/抽取评测/chunk抽取/LLM抽取/text_schema_extrction.json`（沿用指定文件名拼写），与同目录的 `text_extraction.json` 纯 LLM 结果对照。新卡片支持独立显示/隐藏、图谱高度、节点类型筛选、搜索及节点详情；标准答案增加 `Schema` 名称匹配标签。图谱按完整 `metadata.chunk_id = sample_id` 对齐，不能只用不同文档可能重名的局部 Chunk ID。未匹配到结果时明确提示；仍有验证问题的结果显示复核提示。

图谱上方可勾选节点类型，隐藏该类节点及相连关系，“全部显示”可恢复。点击节点会突出其直接关系和关系名称，并在图谱下方列出相邻节点、关系与原始证据；点击关系列表可跳转到相邻节点。点击原始图片可打开大图，支持按钮或滚轮缩放、拖动平移、“适应窗口”复位及 Esc 关闭。

需要换用其他 LLM 结果时，可用 `--baseline-text 路径.json`、`--baseline-image 路径.json` 或 `--baseline-table 路径.json` 覆盖对应默认文件。结果需包含 `graphs` 数组，并在各图谱的 `metadata.chunk_id` 中提供对应的 `sample_id`。

可用 `--schema-text 路径.json` 单独覆盖 Schema 文本结果。修改服务端路径配置后需要重启 `server.py`。补跑与页面相同的 30 个开发集文本样本：

```powershell
python -m src.schemaExtractors.textTextractor.main --input eval/dataset/dev/chunk/text_chunk.json --output eval/抽取评测/chunk抽取/LLM抽取/text_schema_extrction.json --workers 4
```

也可直接运行固定好默认输入与输出的脚本：

```powershell
python eval/抽取评测/chunk抽取/LLM抽取/text_schema_extract.py
```

旧的 8766 服务进程不认识新接口时，静态页面会提示并提供 8767 新版页面入口；新版服务可用 `python eval/front/chunk数据可视化/server.py --port 8767` 启动。

该命令会按完整样本 ID 保存结果，并恢复已成功完成的 Chunk。输出中实体和关系数量是各 Chunk 的记录数，浏览器中的同名匹配提示不代表完整评测指标。
