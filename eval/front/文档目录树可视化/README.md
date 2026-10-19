# Stage 03 文档目录树可视化

在当前目录运行：

```powershell
python server.py
```

然后打开 <http://127.0.0.1:8765>。可用 `python server.py --port 18765` 指定端口。

页面自动扫描 `eval/dataset/dev/mineru/_output` 和 `eval/dataset/dev/mineru_output` 下的直接子文件夹，并列出含有 `stage_03_document_summary.json` 的文件夹。当前仓库样例位于后一个目录。新增结果后刷新页面即可选择。

点击章节可查看摘要、Schema Keys、Concept Categories，以及文本、图像、表格 Chunk。图片从同一文档文件夹的 `images` 目录读取；原始 JSON 可在页面右上角查看。服务仅监听本机 `127.0.0.1`，无需安装第三方依赖。
