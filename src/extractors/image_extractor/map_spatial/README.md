# 地图与平面空间图片抽取器

本模块处理 A01、A02、A07、A18 四类图片。它通过一次 VLM 调用识别地图对象及点、线、面边界，随后在本地完成类型规范化、几何重建、空间关系计算、引用校验、关系去重和统一 `Graph` 装配。

一级输出包括地图主题、地层/层位、地理位置、井、油气田、岩性、岩相、沉积相、古地理单元、构造单元、地质属性及属性值/范围。每个实体同时保留 `type`、`type_zh`，并在属性和元数据中写入 `entity_type`、`entity_type_zh`，方便前端直接展示类型。

关系分为两层：

- 地质语义关系：`DISTRIBUTED_IN`（岩性/岩相分布于）、`DEVELOPED_IN`（沉积相发育于）、`PART_OF`（构造单元隶属于）、`HAS_LITHOLOGY`、`HAS_FACIES`。类型校验会把模型误写的“岩性 `LOCATED_IN` 地点”自动改为 `DISTRIBUTED_IN`。
- 可计算空间关系：`WITHIN`、`CONTAINS`、`COVERS`、`OVERLAPS`、`INTERSECTS`、`CROSSES`、`TOUCHES`、`ADJACENT_TO`、`NEAR`、八方位和页面方向。模型可提供显式关系，本地还会依据几何重新计算关系及距离。

二级输出包括相带过渡、空间展布、高低值区、剖面线、边界、地层剥蚀线、相对方向和空间趋势。所有节点与边都保留 `evidence`、`evidence_scope` 和 `confidence`，无法确认的信息写入 `uncertainties`。

## 几何重建

VLM 为真实地图对象返回 `point`、`line` 或 `polygon` 坐标。坐标统一到左上角为原点、范围为 `[0, 0, 1000, 1000]` 的 `normalized_1000` 坐标系，本地确定性生成：

- `bbox`、`centroid` 和多边形面积；
- 原图存在时的 `pixel_coordinates`、`pixel_bbox` 和 `pixel_centroid`；
- 有至少三个有效控制点时拟合 `pixel_to_world_affine`，并为对象生成 `world_coordinates`；
- 每对几何对象的质心距离、边界距离、方向，以及比例尺可用时的实际地图距离；
- 由几何验证的包含、重叠、穿越、接触、邻接、邻近和方向关系。

完整重建记录保存在 `graph.metadata.extra.geometry_reconstruction`。实体自身的规范几何保存在 `entity.attributes.geometry`。没有比例尺或控制点时仍保留归一化坐标和像素坐标，不推测实际距离或经纬度。

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
