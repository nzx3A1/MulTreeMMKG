# 定量图表与实验曲线抽取器

该模块处理图片分类 A14–A17，使用 VLM 抽取：

- 子图、线性/对数/分类/时间坐标轴；
- 散点、曲线、柱状和分布数据序列；
- 趋势、相关性、阈值、时间变化、序列比较与异常；
- 对应的统一 `Entity` / `Relation` / `Graph`，并保留图像、图注和正文证据。

## 调用

已有分类字段的 chunk 可继续使用统一 `extract_from_images` 入口。调用方已经确认输入均为定量图表时，可直接使用：

```python
from src.extractors.image_extractor.quantitative_chart import extract_quantitative_chart_chunks
from src.utils.vlm_client import VLMClient

graphs = extract_quantitative_chart_chunks(
    image_chunks,
    VLMClient(),
    output_path="output/quantitative_charts.json",
)
```

`quantitative_chart_max_points_per_series`（便捷入口中为 `max_points_per_series`）控制每个序列最多保留多少个代表点，默认 24。密集散点不会被伪精确数字化；无法确认的读数会记录在 `uncertainties`。

## 样例验证

```bash
python -m pytest tests/test_quantitative_chart_extractor.py -q
python util/run_quantitative_chart_live_examples.py
```

真实样例结果默认保存到 `result/quantitative_chart_live_results.json`。运行脚本需要项目 VLM 配置可用。
