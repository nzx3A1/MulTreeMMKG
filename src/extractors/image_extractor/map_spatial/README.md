# 地图与平面空间图片抽取器

本模块处理 A01、A02、A07、A18 四类图片。它通过一次 VLM 调用识别地图对象，随后在本地完成类型规范化、引用校验、关系去重和统一 `Graph` 装配。

一级输出包括地图主题、地层/层位、地理位置、井、油气田、岩相、沉积相、古地理单元、构造单元、地质属性及属性值/范围，并规范为 `LOCATED_IN`、`CONTAINS`、`OVERLAPS`、`ADJACENT_TO` 四类核心空间关系。

二级输出包括相带过渡、空间展布、高低值区、剖面线、边界、地层剥蚀线、相对方向和空间趋势。所有节点与边都保留 `evidence`、`evidence_scope` 和 `confidence`，无法确认的信息写入 `uncertainties`。

## 调用

已有 A01/A02/A07/A18 分类的 chunk 使用统一 `extract_from_images` 入口。若输入已确定全部是地图，可直接调用：

```python
from src.extractors.image_extractor.map_spatial import extract_map_spatial_chunks
from src.utils.vlm_client import VLMClient

graphs = extract_map_spatial_chunks(
    image_chunks,
    VLMClient(),
    output_path="output/map_spatial.json",
)
```

离线契约测试：

```bash
python -m pytest tests/test_map_spatial_extractor.py -q
```
