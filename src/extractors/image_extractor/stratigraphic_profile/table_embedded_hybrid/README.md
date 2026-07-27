好的，我已经完整阅读了这个模块的所有核心代码。下面我用一个**具体的、完整的例子**来演示 `table_embedded_hybrid` 是如何解析一张复杂表格图片的。

---

## 具体例子：马五段综合测井—岩性—孔渗—沉积相图

下面以一张综合测井—岩性—孔渗—沉积相图为例说明真实模型抽取流程。

---

### 🖼️ 输入：一张复杂地质图

假设输入的是这样一张图：

```
┌──────────┬────────┬──────┬──────┬──────────┬──────┬──────────┬──────────┬──────────┬────────┐
│ 地层     │ GR/SP  │深度/m│ 岩性 │CNL/AC/DEN│RS/RD │测井/岩心 │测井/岩心 │ 岩性特征 │ 沉积相 │
│ (段/亚段)│ 曲线   │      │ 剖面 │  曲线    │电阻率│孔隙度%   │渗透率M3  │          │        │
├──────────┼────────┼──────┼──────┼──────────┼──────┼──────────┼──────────┼──────────┼────────┤
│ 马五₂    │ ~~~~   │ 3060 │▓▓▓▓▓│  ~~~~    │~~~~  │  ████    │  ████    │白云岩为主│ 云坪   │
├──────────┼────────┼──────┼──────┼──────────┼──────┼──────────┼──────────┼──────────┼────────┤
│ 马五₃    │ ~~~~   │ 3080 │░░░░░│  ~~~~    │~~~~  │  ████    │  ████    │泥质白云岩│ 泥云坪 │
│          │        │      │░░░░░│          │      │          │          │  为主    │        │
├──────────┼────────┼──────┼──────┼──────────┼──────┼──────────┼──────────┼──────────┼────────┤
│ 马五₄    │ ~~~~   │ 3100 │▓▓▓▓▓│  ~~~~    │~~~~  │  ████    │  ████    │白云岩    │ 云坪   │
│          │        │      │░░░░░│          │      │          │          │泥质白云岩│ 泥云坪 │
├──────────┼────────┼──────┼──────┼──────────┼──────┼──────────┼──────────┼──────────┼────────┤
│ ...      │ ...    │ ...  │ ...  │  ...     │ ...  │  ...     │  ...     │  ...     │  ...   │
└──────────┴────────┴──────┴──────┴──────────┴──────┴──────────┴──────────┴──────────┴────────┘
```

这张图有 **10 条轨道**（地层、GR/SP 曲线、深度、岩性柱、CNL/AC/DEN 曲线、电阻率、孔隙度、渗透率、岩性文字、沉积相），深度范围 3060—3150 m。

---

### 🔄 完整处理流程

#### **阶段 0：图片分类**

```
图片 → A01-A20 大类分类 → A06(测井综合图) → 子分类 → table_embedded_hybrid ✅
```

#### **阶段 1：三段视觉解析**（[segmented_vlm.py](file:///c:/Users/nzx/Desktop/Python/毕业论文项目/MulTreeMMKG/src/extractors/image_extractor/stratigraphic_profile/table_embedded_hybrid/segmented_vlm.py)）

同一张图片分 **3 次**发给 VLM，每次只读一部分内容：

| 分段 | 读什么 | 不读什么 |
|------|--------|----------|
| `layout` | 版式、坐标、轨道结构 | 不读地层内容 |
| `stratigraphy_lithology` | 地层、岩性、地质特征 | 不读曲线和储层 |
| `facies_reservoir_wells` | 曲线、相、储层、井号 | 不重复读地层 |

**第一段 `layout` 的输出**（简化）：

```json
{
  "layout_family": "multi_track_well_log",
  "image_size": {"width": 1234, "height": 972},
  "coordinate_system": {
    "content_bbox": [12, 143, 1232, 951],
    "vertical_axis": {
      "kind": "depth", "unit": "m", "increases": "downward",
      "calibration_points": [
        {"pixel_y": 176, "value": 3060},
        {"pixel_y": 339, "value": 3080},
        {"pixel_y": 501, "value": 3100},
        {"pixel_y": 665, "value": 3120},
        {"pixel_y": 827, "value": 3140},
        {"pixel_y": 909, "value": 3150}
      ]
    }
  },
  "tracks": [
    {"id": "track_stratigraphy", "role": "stratigraphy", "header": "地层（段/亚段）", "bbox": [12,143,153,951]},
    {"id": "track_gr_sp", "role": "curve", "header": "GR / SP", "bbox": [153,143,253,951]},
    {"id": "track_depth", "role": "depth", "header": "深度/m", "bbox": [253,143,298,951]},
    {"id": "track_lithology", "role": "lithology", "header": "剖面", "bbox": [298,143,378,951]},
    {"id": "track_facies", "role": "facies", "header": "沉积相", "bbox": [1128,143,1232,951]}
    // ... 共 10 条轨道
  ]
}
```

**第二段 `stratigraphy_lithology` 的输出**（简化）：

```json
{
  "primitives": {
    "stratigraphic_intervals": [
      {"id": "unit_ma5", "name": "马五段", "top_y": 143, "bottom_y": 951, "confidence": 0.99},
      {"id": "unit_ma5_2", "name": "马五₂", "parent_id": "unit_ma5", "top_y": 171, "bottom_y": 216},
      {"id": "unit_ma5_3", "name": "马五₃", "parent_id": "unit_ma5", "top_y": 216, "bottom_y": 380},
      {"id": "unit_ma5_4", "name": "马五₄", "parent_id": "unit_ma5", "top_y": 380, "bottom_y": 543}
      // ... 共 10 个亚段
    ],
    "lithology_intervals": [
      {"id": "lith_3060_3065", "name": "白云岩为主，夹云质泥岩", "top_y": 171, "bottom_y": 216},
      {"id": "lith_3065_3085", "name": "泥质白云岩为主", "top_y": 216, "bottom_y": 380}
      // ... 共 12 个岩性区间
    ]
  }
}
```

**第三段 `facies_reservoir_wells` 的输出**（简化）：

```json
{
  "primitives": {
    "facies_intervals": [
      {"id": "facies_cloud_flat_1", "name": "云坪", "top_y": 171, "bottom_y": 216},
      {"id": "facies_muddy_cloud_flat_1", "name": "泥云坪", "top_y": 216, "bottom_y": 380}
      // ... 共 12 个相区间
    ],
    "curve_tracks": [
      {"id": "curve_gr", "name": "GR", "scale_min": 0, "scale_max": 350},
      {"id": "curve_cnl", "name": "CNL", "scale_min": 45, "scale_max": -15}
      // ... 共 11 条曲线
    ],
    "reservoir_intervals": [
      {"id": "reservoir_3106_3117", "name": "3106—3117 m孔渗有利层段", "top_y": 550, "bottom_y": 640},
      {"id": "reservoir_3130_3150", "name": "3130—3150 m孔渗响应层段", "top_y": 746, "bottom_y": 909}
    ]
  }
}
```

三段合并后，得到完整的 `payload`。

---

#### **阶段 2：坐标检查与修复**（[segmented_vlm.py: validate_and_repair_pixel_geometry](file:///c:/Users/nzx/Desktop/Python/毕业论文项目/MulTreeMMKG/src/extractors/image_extractor/stratigraphic_profile/table_embedded_hybrid/segmented_vlm.py#L239)）

程序读取真实图片尺寸（1234×972），逐一校验：

- 越界的 `pixel_y` → 删除该刻度点
- 超出图片边界的区间 → 裁剪到画布范围
- 完全在图片外的图元 → 直接删除
- 所有修复写入 `uncertainties`，不静默修改

---

#### **阶段 3：坐标系重建**（[layout.py: fit_vertical_axis](file:///c:/Users/nzx/Desktop/Python/毕业论文项目/MulTreeMMKG/src/extractors/image_extractor/stratigraphic_profile/table_embedded_hybrid/layout.py#L57)）

用 6 个锚点做**最小二乘线性拟合**：

```
锚点: pixel_y=176 → 3060m, pixel_y=339 → 3080m, ..., pixel_y=909 → 3150m

拟合结果: depth ≈ 0.0814 × pixel_y + 3046.7
RMSE ≈ 0.3 m
```

之后，**任何像素 y 坐标都可以换算为真实深度**。例如：

| 图元 | top_y | bottom_y | 换算深度范围 |
|------|-------|----------|-------------|
| 马五₂ | 171 | 216 | ~3060—3064 m |
| 马五₃ | 216 | 380 | ~3064—3077 m |
| 储层 3106-3117 | 550 | 640 | ~3091—3099 m |

---

#### **阶段 4：表格线检测**（[layout.py: detect_rule_lines](file:///c:/Users/nzx/Desktop/Python/毕业论文项目/MulTreeMMKG/src/extractors/image_extractor/stratigraphic_profile/table_embedded_hybrid/layout.py#L97)）

用 Pillow 读取灰度图 → NumPy 做深色像素投影：

```
某 x 位置纵向深色像素占比 ≥ 45% → 可能是竖向表格线
某 y 位置横向深色像素占比 ≥ 55% → 可能是横向表格线
```

检测结果与 VLM 识别的轨道边界交叉验证，标记每条轨道的 `left_rule_supported` / `right_rule_supported`。

---

#### **阶段 5：图元规范化**（[pipeline.py: _normalize_interval](file:///c:/Users/nzx/Desktop/Python/毕业论文项目/MulTreeMMKG/src/extractors/image_extractor/stratigraphic_profile/table_embedded_hybrid/pipeline.py#L62)）

每个区间图元统一转换为：

```json
{
  "id": "unit_ma5_2",
  "name": "马五₂",
  "kind": "stratigraphic_interval",
  "top_y": 171,
  "bottom_y": 216,
  "top_value": 3060.6,
  "bottom_value": 3064.3,
  "vertical_unit": "m",
  "evidence": "左侧段列的马五段合并单元格",
  "confidence": 0.96,
  "attributes": {"parent_id": "unit_ma5", "rank": "亚段", "combination": "上组合"}
}
```

---

#### **阶段 6：深度对齐 — 核心步骤**（[pipeline.py: _depth_align](file:///c:/Users/nzx/Desktop/Python/毕业论文项目/MulTreeMMKG/src/extractors/image_extractor/stratigraphic_profile/table_embedded_hybrid/pipeline.py#L112)）

**关键思想**：不同轨道的图元之间不靠"看起来靠得近"建立关系，而是靠**公共纵轴上的区间交集**。

以 `reservoir_3106_3117`（储层，深度 3091—3099 m）为例：

```
与马五₃（3064—3077 m）交集 → 重叠比例 < 0.15 → ❌ 不建立关系
与马五₄（3077—3090 m）交集 → 重叠比例 < 0.15 → ❌ 不建立关系
与马五₅（3090—3095 m）交集 → 重叠比例 > 0.15 → ✅ 生成：马五₅ --contains_reservoir--> 储层
与马五₆（3095—3107 m）交集 → 重叠比例 > 0.15 → ✅ 生成：马五₆ --contains_reservoir--> 储层
```

如果马五₅ 和马五₆ 都是马五段的子层，优先连接更具体的子层，避免同一事实同时挂到父层和子层。

类似地，沉积相区间与地层区间对齐：

```
云坪（top_y=171, bottom_y=216）与马五₂（top_y=171, bottom_y=216）完全重叠
→ 生成：马五₂ --has_sedimentary_facies--> 云坪
```

---

#### **阶段 7：层级与层序关系**（[pipeline.py: _hierarchy_and_order](file:///c:/Users/nzx/Desktop/Python/毕业论文项目/MulTreeMMKG/src/extractors/image_extractor/stratigraphic_profile/table_embedded_hybrid/pipeline.py#L161)）

**层级关系**（来自合并单元格的 `parent_id`）：

```
马五₂ --part_of--> 马五段   (explicit=true, basis=merged_stratigraphic_table_cell)
马五₃ --part_of--> 马五段
马五₄ --part_of--> 马五段
...
```

**上覆关系**（同父层下按深度排序）：

```
马五₂ --directly_overlies--> 马五₃   (explicit=false, basis=same_parent_vertical_order)
马五₃ --directly_overlies--> 马五₄
马五₄ --directly_overlies--> 马五₅
...
```

---

#### **阶段 8：确定性知识图谱装配**（[graph.py](file:///c:/Users/nzx/Desktop/Python/毕业论文项目/MulTreeMMKG/src/extractors/image_extractor/stratigraphic_profile/table_embedded_hybrid/graph.py)）

**不再调用大模型**，完全由程序执行：

1. **生成稳定 ID**：`image_id + table-hybrid + 对象种类 + 局部ID → SHA1 哈希`
   ```
   "logging-panel:image:0:table-hybrid:entity:unit-ma5-2:abc123def4"
   ```

2. **注册节点**：

   | 局部 ID | 图谱节点类型 | 名称 |
   |---------|-------------|------|
   | `unit_ma5` | `stratigraphic_unit` | 马五段 |
   | `unit_ma5_2` | `stratigraphic_unit` | 马五₂ |
   | `lith_3060_3065` | `lithology_interval` | 白云岩为主 |
   | `facies_cloud_flat_1` | `sedimentary_facies_interval` | 云坪 |
   | `reservoir_3106_3117` | `reservoir_interval` | 3106—3117 m孔渗有利层段 |
   | `curve_gr` | `log_curve` | GR |
   | ... | ... | ... |

3. **装配关系**（每条都带 `explicit` 标记和 `inference_basis`）：

   ```
   马五段 --contains--> 马五₂           (explicit=true, 图内可见)
   马五₂ --part_of--> 马五段            (explicit=true, 合并单元格)
   马五₂ --directly_overlies--> 马五₃   (explicit=false, 程序按深度排序)
   马五₂ --has_lithology--> 白云岩为主   (explicit=false, 纵轴区间重叠)
   马五₂ --has_sedimentary_facies--> 云坪 (explicit=false, 纵轴区间重叠)
   马五₅ --contains_reservoir--> 储层    (explicit=false, 纵轴区间重叠)
   综合图 --contains--> 马五段           (explicit=true, 图内可见)
   综合图 --has_log_curve--> GR          (explicit=true, 图内可见)
   ```

4. **来源追踪**：每个节点和关系都强制写入：
   ```json
   {
     "source_image_path": "logging_panel.png",
     "source_image_id": "logging-panel:image:0",
     "source_chunk_id": "logging-panel",
     "image_index": 0,
     "visual_evidence": "左侧段列的马五段合并单元格"
   }
   ```

5. **质量校验**：引用完整性、来源一致性、证据非空、坐标范围、纵轴锚点数量等。

---

### 📊 最终输出：知识图谱

最终输出的 `Graph` 包含约 **30+ 个节点** 和 **50+ 条关系**，形成如下结构：

```
stratigraphic_profile (综合图)
  ├── contains → 马五段
  │     ├── contains → 马五₂ ── has_lithology → 白云岩为主
  │     │                    ── has_sedimentary_facies → 云坪
  │     │                    ── directly_overlies → 马五₃
  │     ├── contains → 马五₃ ── has_lithology → 泥质白云岩为主
  │     │                    ── has_sedimentary_facies → 泥云坪
  │     ├── contains → 马五₄ ...
  │     ├── contains → 马五₅ ── contains_reservoir → 3106-3117 m储层
  │     └── ...
  ├── has_log_curve → GR
  ├── has_log_curve → SP
  ├── has_log_curve → CNL
  └── ...
```

---

### 💡 核心设计思想总结

| 设计选择 | 原因 |
|---------|------|
| **三段 VLM 而非一次性** | 避免超时、截断、漏读下半图 |
| **像素坐标 + 最小二乘拟合** | 不同轨道统一到同一根深度轴 |
| **区间交集而非横向邻近** | 确定性对齐，不靠"看起来靠得近" |
| **程序生成关系而非模型** | 避免模型猜测深度错位和虚假关系 |
| **`explicit=true/false` 标记** | 区分"图上明确写了什么"和"程序计算出了什么" |
| **所有修复写入 uncertainties** | 可审计，不静默修改 |
| **稳定 ID（SHA1 哈希）** | 重复运行 ID 不变，跨图片不冲突 |
