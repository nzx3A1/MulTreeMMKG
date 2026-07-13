# 从 output 目录启动本地静态服务，使页面可以安全读取 stage 03 与 stage 04 JSON 数据。
$outputDir = Split-Path -Parent $PSScriptRoot
$url = "http://127.0.0.1:8765/stage_04_text_extraction_front/index.html"
Start-Process $url
Set-Location $outputDir
Write-Host "知识图谱观测台已启动：$url" -ForegroundColor Cyan
Write-Host "关闭此窗口即可停止服务。" -ForegroundColor DarkGray
python -m http.server 8765 --bind 127.0.0.1
