# MulTreeMMKG 石油地质领域本体完整规范（Schema v1.0）

## 1. 总体设计原则

**核心准则：地质对象建实体；描述性类别建属性；具体测量数值建 `Measurement`；文献观点建 `Claim`；来源证据建 `Evidence`。**

- 只有可定位或可独立识别的对象、受控概念、地质过程和样品进入 20 类核心实体。属性词（如孔隙度、岩性、矿物组成、砂体结构）不因在文本中被提及就自动建成节点。
- `DepositionalSystem`（沉积体系）、`SedimentaryFacies`（沉积相）与 `DepositionalEnvironment`（沉积环境）分开建模；沉积相中的相、亚相、微相合并为一类并使用 `facies_level`。
- `StratigraphicUnit` 同时容纳岩石地层、油层组和层序地层等划分方案，但**不因名称相近就视为同一地层**。
- `Reservoir` 是储集岩层，`OilGasReservoir` 是油气聚集体；两者不得混用。
- 图像、表格、测井曲线、地质图属于证据资料；其中识别出的具体地质对象仍可成为实体。
- 文献断言必须尽可能保留空间范围、地层范围、时代、作者归属、肯定/否定、证据位置和审核状态。
- 类型及关系约束来源于现有 YAML；本 Markdown 不擅自新增会使机器 Schema 不兼容的实体或关系。

**实体分类 `entity_kind`：** `geographic`（地理地质区域）、`scope`（研究范围）、`physical`（实体地质对象）、`concept`（受控地质概念）、`process`（地质过程）、`concept_or_instance`（概念或具体系统实例）。

## 2. 核心概念实体：20 类

| 分组 | Schema 实体类型 | 中文名称 | 类别 | 定义与判定依据 | 典型示例 |
| --- | --- | --- | --- | --- | --- |
| 地理与勘探空间 | `Basin` | 盆地 | `geographic` | 具有相对独立沉积和构造演化背景的区域地质单元 | 鄂尔多斯盆地；四川盆地 |
| 地理与勘探空间 | `StructuralUnit` | 构造单元 | `geographic` | 盆地内部依据构造形态和演化划分的构造区域；凹陷、坳陷、洼陷、隆起、斜坡、构造带统一为子类型 | 伊陕斜坡；天环坳陷 |
| 地理与勘探空间 | `StudyArea` | 研究区/区块 | `scope` | 文献、项目或勘探开发定义的空间范围；区块、井区和露头地点作为范围子类型 | 鄂尔多斯盆地西南部；苏里格区块 |
| 地理与勘探空间 | `OilGasField` | 油气田 | `geographic` | 由一个或多个油气藏组成的油气聚集与开发区域；与储集岩层和单个油气藏区分 | 苏里格气田；大庆油田 |
| 地理与勘探空间 | `Well` | 井 | `physical` | 具有唯一井号或井位的钻井实体 | 苏6井；塔中1井 |
| 地层与地质构造 | `StratigraphicUnit` | 地层单元 | `physical` | 按岩石地层、含油层系或层序地层等划分方案确定的地层单位；组、段、亚段、小层、层序、体系域为层级属性 | 延长组；长8油层组；SQ1 |
| 地层与地质构造 | `GeologicalBoundary` | 地质界面 | `physical` | 用于限定或分隔地层单元的地质界面；不整合面、层序界面、地层界面为子类型 | 延长组底界面；SB1 |
| 地层与地质构造 | `GeologicalStructure` | 地质构造体 | `physical` | 可定位的断层、断裂带、褶皱、裂缝等构造对象；一般构造形态词仅作类型属性 | F1断层；郯庐断裂带 |
| 沉积体系、沉积相与地质作用 | `DepositionalSystem` | 沉积体系 | `concept` | 相互关联的沉积环境与沉积体组成的沉积系统；用于辫状河三角洲、浅水三角洲等体系级解释 | 辫状河三角洲；浅水三角洲 |
| 沉积体系、沉积相与地质作用 | `SedimentaryFacies` | 沉积相 | `concept` | 沉积相及其亚相、微相的受控分类概念；用 facies_level 表示层级 | 三角洲相；三角洲前缘亚相；水下分流河道微相 |
| 沉积体系、沉积相与地质作用 | `DepositionalEnvironment` | 沉积环境 | `concept` | 沉积形成时的环境类型概念；与沉积体系和沉积相不等同 | 浅湖环境；半深湖环境 |
| 沉积体系、沉积相与地质作用 | `SandBody` | 砂体 | `physical` | 具有可识别空间范围的砂质沉积体或研究区内特定砂体；砂体结构、厚度和展布为属性 | 长8分流河道砂体；河口坝砂体 |
| 沉积体系、沉积相与地质作用 | `GeologicalProcess` | 地质过程 | `process` | 影响地层、沉积体和成藏的地质作用或演化过程；成岩、构造、湖盆演化和油气运移为过程类型 | 湖盆演化；压实作用；溶蚀作用 |
| 油气成藏要素 | `SourceRock` | 烃源岩 | `physical` | 承担生烃作用的特定岩石体或层段；TOC、成熟度和干酪根类型为属性 | 长7烃源岩 |
| 油气成藏要素 | `Reservoir` | 储层 | `physical` | 具有储集和渗流空间的岩层或岩体；区别于已经聚集油气的油气藏 | 长8砂岩储层 |
| 油气成藏要素 | `Seal` | 盖层 | `physical` | 起封隔作用的低渗透岩层或岩体 | 泥岩盖层；膏盐岩盖层 |
| 油气成藏要素 | `Trap` | 圈闭 | `physical` | 具有阻挡油气运移并形成聚集条件的空间地质配置 | 长8岩性圈闭 |
| 油气成藏要素 | `OilGasReservoir` | 油气藏 | `physical` | 油气在圈闭或储层中形成的具体聚集体；不同于储层 Reservoir | 长8油藏；马五气藏 |
| 油气成藏要素 | `PetroleumSystem` | 含油气系统 | `concept_or_instance` | 烃源岩、储层、盖层、圈闭及成藏过程组成的地质系统 | 延长组成藏系统 |
| 样品 | `Sample` | 样品 | `physical` | 可定位来源的岩心、薄片、岩石或流体实物样品；实验方法不单独作为地质实体 | 苏6井岩心样品；铸体薄片样品 ；花岗岩|

### 2.1 实体类型的合并边界

- **构造单元**：`Depression`、`Sag`、`Uplift`、`Slope`、`StructuralBelt` → `StructuralUnit`，用 `structural_unit_type` 区分。
- **地层单位**：`Formation`、`StratigraphicMember`、`SubMember`、`Bed`、`SequenceStratigraphicUnit`、`SystemTract` → `StratigraphicUnit`，用 `stratigraphic_scheme` + `unit_level` 区分。
- **地质界面**：`StratigraphicBoundary`、`Unconformity`、`SequenceBoundary` → `GeologicalBoundary`，用 `boundary_type` 区分。
- **地质构造体**：具名断层、断裂带、可定位裂缝等 → `GeologicalStructure`，用 `structure_type` 区分；一般裂缝发育程度是属性。
- **沉积相层级**：`SedimentaryFacies`、`Subfacies`、`Microfacies` → `SedimentaryFacies`，用 `facies_level` 区分。
- **样品**：岩心、薄片、岩石、流体等实物 → `Sample`，用 `sample_type` 区分；实验方法作为测量元数据。
- **不再独立建实体**：岩性、矿物、岩石结构、沉积构造、地质时代、干酪根类型、孔隙结构、流体类型、一般的砂体结构与储集性能。

## 3. 实体属性：全部字段

属性名严格沿用 `properties.yaml`。表中“规范要求”表示字段定义中未带 `?`；**现有 `validate_schema.py` 对已提供字段校验类型，但尚未逐项强制要求所有此类属性必须出现**，不要把脚本的通过结果理解为属性完整。

### 3.1 所有实体的公共字段

| 字段 | 中文含义 | 数据类型 | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| `id` | 唯一标识 | 字符串 | 必填 | 稳定的实体 ID |
| `type` | 实体类型 | 字符串 | 必填 | 限 20 类 Schema |
| `canonical_name` | 标准名称 | 字符串 | 必填 | 需结合研究区和层位消歧 |
| `entity_kind` | 实体种类 | 枚举 | 必填 | 地理/范围/物理/概念/过程等 |
| `aliases` | 别名 | 字符串列表 | 可选 | 原文别称及归一化别名 |
| `attributes` | 专属属性 | 对象 | 可选 | 只允许本体中声明的字段 |
| `spatial_scope` | 空间范围 | 字符串 | 可选 | 保留来源范围 |
| `stratigraphic_scope` | 地层范围 | 字符串 | 可选 | 保留来源层位 |
| `temporal_scope` | 时间范围 | 字符串 | 可选 | 地质时代或过程期次 |
| `normalization_status` | 归一化状态 | 枚举 | 可选 | confirmed / pending / ambiguous |

### 3.2 按实体类型列出的专属属性
#### 地理与勘探空间

##### `Basin` — 盆地

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `basin_type` | 盆地类型 | 字符串 | 可选 | — |
| `geologic_age` | 地质时代 | 字符串 | 可选 | — |
| `area_km2` | 面积（平方千米） | 浮点数 | 可选 | — |

##### `StructuralUnit` — 构造单元

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `structural_unit_type` | 构造单元类型 | 枚举 | 规范要求 | `depression`、`sag`、`uplift`、`slope`、`structural_belt`、`other` |
| `structural_position` | 构造部位 | 字符串 | 可选 | — |
| `orientation` | 构造走向或方位 | 字符串 | 可选 | — |

##### `StudyArea` — 研究区/区块

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `area_type` | 研究区类别 | 枚举 | 规范要求 | `study_area`、`exploration_block`、`well_area`、`outcrop_site`、`sweet_spot_area`、`other` |
| `boundary_description` | 边界描述 | 字符串 | 可选 | — |
| `geometry_wkt` | WKT空间几何 | 字符串 | 可选 | — |
| `crs` | 坐标参考系统 | 字符串 | 可选 | — |

##### `OilGasField` — 油气田

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `field_type` | 油气田类型 | 枚举 | 规范要求 | `oil`、`gas`、`mixed`、`unknown` |
| `development_status` | 开发状态 | 字符串 | 可选 | — |

##### `Well` — 井

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `well_number` | 井号 | 字符串 | 规范要求 | — |
| `longitude` | 经度 | 浮点数 | 可选 | — |
| `latitude` | 纬度 | 浮点数 | 可选 | — |
| `total_depth_m` | 井深（米） | 浮点数 | 可选 | — |
| `well_type` | 井类型 | 字符串 | 可选 | — |

#### 地层与地质构造

##### `StratigraphicUnit` — 地层单元

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `stratigraphic_scheme` | 地层划分体系 | 枚举 | 规范要求 | `lithostratigraphic`、`oil_bearing_interval`、`sequence_stratigraphic`、`chronostratigraphic`、`other` |
| `unit_level` | 地层单位级别 | 枚举 | 规范要求 | `formation`、`member`、`submember`、`bed`、`oil_layer_group`、`sequence`、`system_tract`、`other` |
| `geologic_age` | 地质时代 | 字符串 | 可选 | — |
| `lithology` | 岩性 | 字符串列表 | 可选 | — |
| `thickness_m` | 厚度（米） | 浮点数 | 可选 | — |
| `is_target_interval` | 是否目的层 | 布尔值 | 可选 | — |
| `top_depth_m` | 顶界深度（米） | 浮点数 | 可选 | — |
| `bottom_depth_m` | 底界深度（米） | 浮点数 | 可选 | — |
| `mineral_composition` | 矿物组成 | 字符串列表 | 可选 | — |
| `rock_texture` | 岩石结构/组构 | 字符串 | 可选 | — |

##### `GeologicalBoundary` — 地质界面

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `boundary_type` | 界面类型 | 枚举 | 规范要求 | `stratigraphic_boundary`、`unconformity`、`sequence_boundary`、`erosional_surface`、`other` |
| `boundary_level` | 界面级别 | 字符串 | 可选 | — |
| `surface_code` | 界面编号 | 字符串 | 可选 | — |

##### `GeologicalStructure` — 地质构造体

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `structure_type` | 地质构造类型 | 枚举 | 规范要求 | `fault`、`fault_zone`、`fracture`、`fold`、`anticline`、`syncline`、`other` |
| `fault_type` | 断层类型 | 字符串 | 可选 | — |
| `strike_deg` | 走向（度） | 浮点数 | 可选 | — |
| `dip_deg` | 倾角（度） | 浮点数 | 可选 | — |
| `throw_m` | 断距（米） | 浮点数 | 可选 | — |
| `sealing_behavior` | 断层封堵行为 | 字符串 | 可选 | — |
| `pathway_role` | 输导作用角色 | 字符串 | 可选 | — |

#### 沉积体系、沉积相与地质作用

##### `DepositionalSystem` — 沉积体系

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `system_family` | 沉积体系大类 | 枚举 | 规范要求 | `delta`、`fluvial`、`lacustrine`、`turbidite_fan`、`carbonate_platform`、`other` |
| `delta_feeding_style` | 三角洲供源河流样式 | 枚举 | 规范要求 | `braided_river`、`meandering_river`、`fan`、`unknown` |
| `water_depth_setting` | 沉积体系水深背景 | 枚举 | 规范要求 | `shallow`、`deep`、`unknown` |
| `basin_setting` | 盆地环境类型 | 枚举 | 规范要求 | `lacustrine`、`marine`、`other`、`unknown` |
| `system_description` | 沉积体系描述 | 字符串 | 可选 | — |

##### `SedimentaryFacies` — 沉积相

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `facies_level` | 沉积相层级 | 枚举 | 规范要求 | `facies`、`subfacies`、`microfacies` |
| `facies_family` | 沉积相类别 | 字符串 | 可选 | — |
| `sedimentary_structures` | 沉积构造特征 | 字符串列表 | 可选 | — |

##### `DepositionalEnvironment` — 沉积环境

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `environment_type` | 沉积环境类型 | 枚举 | 规范要求 | `fluvial`、`lacustrine`、`deltaic`、`marine`、`transitional`、`other` |
| `water_depth_class` | 水深等级 | 字符串 | 可选 | — |
| `energy_level` | 水动力能级 | 字符串 | 可选 | — |
| `salinity` | 盐度或咸化程度 | 字符串 | 可选 | — |

##### `SandBody` — 砂体

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `sandbody_type` | 砂体类型 | 枚举 | 规范要求 | `distributary_channel`、`mouth_bar`、`sheet_sand`、`channel_fill`、`turbidite_sand`、`other` |
| `sandbody_architecture` | 砂体构型 | 字符串 | 可选 | — |
| `geometry_type` | 几何形态类型 | 字符串 | 可选 | — |
| `distribution` | 空间展布 | 字符串 | 可选 | — |
| `connectivity` | 连通性 | 字符串 | 可选 | — |
| `thickness_m` | 厚度（米） | 浮点数 | 可选 | — |
| `lithology` | 岩性 | 字符串列表 | 可选 | — |
| `mineral_composition` | 矿物组成 | 字符串列表 | 可选 | — |
| `rock_texture` | 岩石结构/组构 | 字符串 | 可选 | — |
| `pathway_role` | 输导作用角色 | 字符串 | 可选 | — |

##### `GeologicalProcess` — 地质过程

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `process_type` | 地质过程类型 | 枚举 | 规范要求 | `lake_basin_evolution`、`compaction`、`cementation`、`dissolution`、`tectonism`、`erosion`、`hydrocarbon_generation`、`expulsion`、`migration`、`accumulation`、`other` |
| `diagenetic_stage` | 成岩阶段 | 字符串 | 可选 | — |
| `process_interval` | 过程发生期段 | 字符串 | 可选 | — |
| `mechanism` | 作用机制 | 字符串 | 可选 | — |

#### 油气成藏要素

##### `SourceRock` — 烃源岩

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `lithology` | 岩性 | 字符串列表 | 可选 | — |
| `organic_matter_type` | 有机质类型 | 字符串 | 可选 | — |
| `kerogen_type` | 干酪根类型 | 字符串 | 可选 | — |
| `toc_wt_pct` | 总有机碳含量（wt%） | 浮点数 | 可选 | — |
| `ro_pct` | 镜质体反射率（%） | 浮点数 | 可选 | — |
| `hydrocarbon_potential` | 生烃潜力 | 字符串 | 可选 | — |
| `mineral_composition` | 矿物组成 | 字符串列表 | 可选 | — |
| `rock_texture` | 岩石结构/组构 | 字符串 | 可选 | — |

##### `Reservoir` — 储层

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `lithology` | 岩性 | 字符串列表 | 可选 | — |
| `mineral_composition` | 矿物组成 | 字符串列表 | 可选 | — |
| `rock_texture` | 岩石结构/组构 | 字符串 | 可选 | — |
| `porosity_pct` | 孔隙度（%） | 浮点数 | 可选 | — |
| `permeability_md` | 渗透率（mD） | 浮点数 | 可选 | — |
| `pore_types` | 孔隙类型 | 字符串列表 | 可选 | — |
| `pore_structure` | 孔隙结构 | 字符串 | 可选 | — |
| `diagenetic_facies` | 成岩相 | 字符串 | 可选 | — |
| `reservoir_quality` | 储层品质 | 字符串 | 可选 | — |
| `storage_capacity` | 储集能力/储集性能 | 字符串 | 可选 | — |
| `fracture_development` | 裂缝发育特征 | 字符串 | 可选 | — |

##### `Seal` — 盖层

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `lithology` | 岩性 | 字符串列表 | 可选 | — |
| `thickness_m` | 厚度（米） | 浮点数 | 可选 | — |
| `permeability_md` | 渗透率（mD） | 浮点数 | 可选 | — |
| `sealing_capacity` | 封盖能力 | 字符串 | 可选 | — |
| `mineral_composition` | 矿物组成 | 字符串列表 | 可选 | — |
| `rock_texture` | 岩石结构/组构 | 字符串 | 可选 | — |

##### `Trap` — 圈闭

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `trap_type` | 圈闭类型 | 枚举 | 规范要求 | `structural`、`lithologic`、`stratigraphic`、`composite`、`other` |
| `closure_height_m` | 圈闭闭合高度（米） | 浮点数 | 可选 | — |
| `closure_area_km2` | 圈闭闭合面积（平方千米） | 浮点数 | 可选 | — |

##### `OilGasReservoir` — 油气藏

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `fluid_type` | 流体类型 | 枚举 | 规范要求 | `oil`、`gas`、`condensate`、`mixed`、`unknown` |
| `accumulation_type` | 油气聚集类型 | 字符串 | 可选 | — |
| `fluid_contact_type` | 流体界面类型 | 字符串 | 可选 | — |
| `fluid_contact_depth_m` | 流体界面深度（米） | 浮点数 | 可选 | — |
| `pressure_mpa` | 压力（MPa） | 浮点数 | 可选 | — |

##### `PetroleumSystem` — 含油气系统

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `system_type` | 含油气系统类型 | 字符串 | 可选 | — |
| `generation_period` | 生烃时期 | 字符串 | 可选 | — |
| `accumulation_period` | 成藏时期 | 字符串 | 可选 | — |

#### 样品

##### `Sample` — 样品

| 属性字段 | 中文含义 | 类型 | 可选性 | 枚举范围/备注 |
| --- | --- | --- | --- | --- |
| `sample_type` | 样品类型 | 枚举 | 规范要求 | `core`、`thin_section`、`rock`、`fluid`、`organic_matter`、`other` |
| `sample_number` | 样品编号 | 字符串 | 可选 | — |
| `depth_top_m` | 样品顶深（米） | 浮点数 | 可选 | — |
| `depth_bottom_m` | 样品底深（米） | 浮点数 | 可选 | — |
| `preparation_method` | 样品制备方法 | 字符串 | 可选 | — |
| `mineral_composition` | 矿物组成 | 字符串列表 | 可选 | — |
| `rock_texture` | 岩石结构/组构 | 字符串 | 可选 | — |

### 3.3 属性建模说明

- 单值的 `porosity_pct`、`permeability_md`、`thickness_m` 等可以表示有明确统计口径的汇总属性；不同井、样品、实验条件下的原始值应另建 `Measurement`。
- `lithology`、`mineral_composition`、`pore_types` 等允许列表值，不应强行合并不同组成项。
- `geologic_age` 是地质时代标签，当前版本不单独建立地质时代节点。
- `SandBody.sandbody_architecture` 表示砂体结构；可作为 `controls` / `influences` 的 `source_property`，无需额外建立“砂体结构”节点。
- 枚举字段使用 YAML 中的机器值，中文显示名不应替换枚举值。

## 4. 实体关系：32 类及约束

统一采用 **主语 source → 谓词 predicate → 宾语 target** 的有向表达。默认多对多；除显式 `symmetric: true` 外，不自动添加反向边。

### 4.1 关系总览

| 关系 ID | 中文名称 | 允许主语类型 | 允许宾语类型 | 必需限定词 |
| --- | --- | --- | --- | --- |
| `located_in` | 位于 | `StructuralUnit`、`StudyArea`、`OilGasField`、`Well`、`StratigraphicUnit`、`GeologicalStructure`、`SandBody`、`SourceRock`、`Reservoir`、`Seal`、`Trap`、`OilGasReservoir`、`Sample` | `Basin`、`StructuralUnit`、`StudyArea`、`OilGasField` | — |
| `structural_part_of` | 构造组成 | `StructuralUnit` | `Basin`、`StructuralUnit` | — |
| `stratigraphic_part_of` | 地层隶属 | `StratigraphicUnit` | `StratigraphicUnit` | `relation_basis` |
| `stratigraphically_overlies` | 地层上覆于 | `StratigraphicUnit` | `StratigraphicUnit` | `spatial_scope_id` |
| `bounded_by` | 以界面为界 | `StratigraphicUnit` | `GeologicalBoundary` | `boundary_role` |
| `cuts_through` | 切穿 | `GeologicalStructure` | `StratigraphicUnit`、`Reservoir`、`Seal`、`SourceRock` | — |
| `has_depositional_system` | 发育沉积体系 | `StratigraphicUnit`、`StudyArea` | `DepositionalSystem` | `spatial_scope_id` |
| `system_contains_facies` | 体系包含沉积相 | `DepositionalSystem` | `SedimentaryFacies` | — |
| `facies_part_of` | 相级隶属 | `SedimentaryFacies` | `SedimentaryFacies` | `facies_hierarchy_scheme` |
| `has_facies` | 发育沉积相 | `StratigraphicUnit`、`SandBody`、`Reservoir`、`StudyArea` | `SedimentaryFacies` | `spatial_scope_id` |
| `deposited_in` | 形成于沉积环境 | `StratigraphicUnit`、`SandBody`、`SedimentaryFacies` | `DepositionalEnvironment` | — |
| `contains_sandbody` | 发育砂体 | `StratigraphicUnit`、`Reservoir`、`StudyArea` | `SandBody` | `spatial_scope_id` |
| `occurs_in_unit` | 赋存于地层 | `SourceRock`、`Reservoir`、`Seal`、`OilGasReservoir` | `StratigraphicUnit` | — |
| `process_affects` | 过程作用于 | `GeologicalProcess` | `StratigraphicUnit`、`SandBody`、`Reservoir`、`SourceRock`、`Seal`、`Trap`、`OilGasReservoir` | `target_property` |
| `controls` | 控制 | `StructuralUnit`、`GeologicalStructure`、`DepositionalSystem`、`SedimentaryFacies`、`SandBody`、`GeologicalProcess` | `StratigraphicUnit`、`SandBody`、`Reservoir`、`Trap`、`OilGasReservoir` | `target_property`、`spatial_scope_id` |
| `influences` | 影响 | `StructuralUnit`、`GeologicalStructure`、`DepositionalSystem`、`SedimentaryFacies`、`SandBody`、`GeologicalProcess` | `StratigraphicUnit`、`SandBody`、`Reservoir`、`Trap`、`OilGasReservoir` | `target_property` |
| `petroleum_system_has_source` | 系统包含烃源岩 | `PetroleumSystem` | `SourceRock` | — |
| `petroleum_system_has_reservoir` | 系统包含储层 | `PetroleumSystem` | `Reservoir` | — |
| `petroleum_system_has_seal` | 系统包含盖层 | `PetroleumSystem` | `Seal` | — |
| `petroleum_system_has_trap` | 系统包含圈闭 | `PetroleumSystem` | `Trap` | — |
| `petroleum_system_has_accumulation` | 系统包含油气藏 | `PetroleumSystem` | `OilGasReservoir` | — |
| `field_contains_accumulation` | 油气田包含油气藏 | `OilGasField` | `OilGasReservoir` | — |
| `accumulation_hosted_in` | 油气藏赋存于储层 | `OilGasReservoir` | `Reservoir` | — |
| `accumulation_trapped_by` | 油气藏受圈闭约束 | `OilGasReservoir` | `Trap` | — |
| `accumulation_sealed_by` | 油气藏由盖层封隔 | `OilGasReservoir` | `Seal` | — |
| `accumulation_charged_from` | 油气藏来源于烃源岩 | `OilGasReservoir` | `SourceRock` | — |
| `well_intersects_unit` | 井钻遇地层 | `Well` | `StratigraphicUnit` | — |
| `sample_from_well` | 样品取自井 | `Sample` | `Well` | — |
| `sample_from_unit` | 样品取自地层 | `Sample` | `StratigraphicUnit` | — |
| `sample_represents` | 样品代表地质对象 | `Sample` | `SourceRock`、`Reservoir`、`Seal`、`SandBody` | — |
| `correlates_with` | 地层对比对应 | `StratigraphicUnit` | `StratigraphicUnit` | `spatial_scope_id` |
| `facies_transitions_to` | 沉积相过渡为 | `SedimentaryFacies` | `SedimentaryFacies` | `spatial_scope_id`、`transition_axis` |

### 4.2 每类关系的语义与约束
#### 空间与层级关系

##### `located_in` — 位于

- **定义**：表示明确的地理空间归属或分布范围
- **主语类型**：`StructuralUnit`、`StudyArea`、`OilGasField`、`Well`、`StratigraphicUnit`、`GeologicalStructure`、`SandBody`、`SourceRock`、`Reservoir`、`Seal`、`Trap`、`OilGasReservoir`、`Sample`
- **宾语类型**：`Basin`、`StructuralUnit`、`StudyArea`、`OilGasField`
- **必需限定词**：无额外必需限定词
- **业务约束**：不得把论文研究范围自动当作实体的唯一地理位置；需有空间证据。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `structural_part_of` — 构造组成

- **定义**：构造单元的层级包含关系
- **主语类型**：`StructuralUnit`
- **宾语类型**：`Basin`、`StructuralUnit`
- **必需限定词**：无额外必需限定词
- **业务约束**：不得形成环；不等同于地理相邻。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `stratigraphic_part_of` — 地层隶属

- **定义**：地层单位之间有证据支持的形式层级或物理包含关系
- **主语类型**：`StratigraphicUnit`
- **宾语类型**：`StratigraphicUnit`
- **必需限定词**：`relation_basis`
- **业务约束**：relation_basis=formal_hierarchy/physical_inclusion；同一体系检查层级；跨体系仅在有明确物理包含证据时允许（如油层组位于某组内）；禁止成环，禁止同名即合并。 跨体系物理包含须填写spatial_scope_id。
- **对称性**：否；**基数**：`contextual; at_most_one_formal_parent_per_scheme_and_scope; cross_scheme_physical_inclusion_can_coexist`。

##### `stratigraphically_overlies` — 地层上覆于

- **定义**：在特定剖面或区域中地层单元A位于B之上
- **主语类型**：`StratigraphicUnit`
- **宾语类型**：`StratigraphicUnit`
- **必需限定词**：`spatial_scope_id`
- **业务约束**：不可仅凭编号推断上下关系；需地层剖面证据；反向为下伏于，不额外存边。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `bounded_by` — 以界面为界

- **定义**：地层单元被某个地质界面限定
- **主语类型**：`StratigraphicUnit`
- **宾语类型**：`GeologicalBoundary`
- **必需限定词**：`boundary_role`
- **业务约束**：boundary_role=top/bottom/other；不得把所有界面自动判断为不整合面。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `cuts_through` — 切穿

- **定义**：构造体切穿特定地层或岩体
- **主语类型**：`GeologicalStructure`
- **宾语类型**：`StratigraphicUnit`、`Reservoir`、`Seal`、`SourceRock`
- **必需限定词**：无额外必需限定词
- **业务约束**：结构类型通常为断层、裂缝；要求明确剖面或图件证据。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

#### 沉积与储层关系

##### `has_depositional_system` — 发育沉积体系

- **定义**：地层或研究区内识别到特定沉积体系
- **主语类型**：`StratigraphicUnit`、`StudyArea`
- **宾语类型**：`DepositionalSystem`
- **必需限定词**：`spatial_scope_id`
- **业务约束**：“辫状河三角洲沉积”优先标注为沉积体系，除非原文明确为沉积相层级。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `system_contains_facies` — 体系包含沉积相

- **定义**：沉积体系中包括某类沉积相或亚相、微相
- **主语类型**：`DepositionalSystem`
- **宾语类型**：`SedimentaryFacies`
- **必需限定词**：无额外必需限定词
- **业务约束**：属于领域分类/组合关系，不能自动推断该相在每个研究区均发育。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `facies_part_of` — 相级隶属

- **定义**：微相→亚相→相的层级关系
- **主语类型**：`SedimentaryFacies`
- **宾语类型**：`SedimentaryFacies`
- **必需限定词**：`facies_hierarchy_scheme`
- **业务约束**：必须满足facies_level层级递增且无环；相级分类体系不同时需显式说明。
- **对称性**：否；**基数**：`at_most_one_parent_per_facies_hierarchy_scheme`。

##### `has_facies` — 发育沉积相

- **定义**：特定地层/砂体/储层中识别到沉积相、亚相或微相
- **主语类型**：`StratigraphicUnit`、`SandBody`、`Reservoir`、`StudyArea`
- **宾语类型**：`SedimentaryFacies`
- **必需限定词**：`spatial_scope_id`
- **业务约束**：结论应保留原文措辞、层位、空间范围与证据；不要由共现推断。 若主语为StudyArea，只表示该范围内存在该相/砂体，不可推断全部层位均发育。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `deposited_in` — 形成于沉积环境

- **定义**：特定地质对象在所述环境下沉积或形成
- **主语类型**：`StratigraphicUnit`、`SandBody`、`SedimentaryFacies`
- **宾语类型**：`DepositionalEnvironment`
- **必需限定词**：无额外必需限定词
- **业务约束**：沉积相与沉积环境并非同义词；需有解释性证据。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `contains_sandbody` — 发育砂体

- **定义**：地层或储层中发育可识别砂体
- **主语类型**：`StratigraphicUnit`、`Reservoir`、`StudyArea`
- **宾语类型**：`SandBody`
- **必需限定词**：`spatial_scope_id`
- **业务约束**：砂体需有具体研究区/层位定位；“分流河道型砂体”泛称通常仅为sandbody_type。 若主语为StudyArea，只表示该范围内存在该相/砂体，不可推断全部层位均发育。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `occurs_in_unit` — 赋存于地层

- **定义**：成藏要素或油气藏赋存于某地层单元
- **主语类型**：`SourceRock`、`Reservoir`、`Seal`、`OilGasReservoir`
- **宾语类型**：`StratigraphicUnit`
- **必需限定词**：无额外必需限定词
- **业务约束**：允许跨多个层位；储层和油气藏不是同一类型。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `process_affects` — 过程作用于

- **定义**：地质过程对特定对象发生作用
- **主语类型**：`GeologicalProcess`
- **宾语类型**：`StratigraphicUnit`、`SandBody`、`Reservoir`、`SourceRock`、`Seal`、`Trap`、`OilGasReservoir`
- **必需限定词**：`target_property`
- **业务约束**：target_property指定受作用属性；若只有共现而无作用证据则不抽取。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `controls` — 控制

- **定义**：文献明确提出某因素控制另一对象的某项特征
- **主语类型**：`StructuralUnit`、`GeologicalStructure`、`DepositionalSystem`、`SedimentaryFacies`、`SandBody`、`GeologicalProcess`
- **宾语类型**：`StratigraphicUnit`、`SandBody`、`Reservoir`、`Trap`、`OilGasReservoir`
- **必需限定词**：`target_property`、`spatial_scope_id`
- **业务约束**：严格因果/控制语义；不得将相关性直接转为控制；target_property描述被控制的属性。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `influences` — 影响

- **定义**：文献明确描述因素对对象的影响，但未必达到严格控制关系
- **主语类型**：`StructuralUnit`、`GeologicalStructure`、`DepositionalSystem`、`SedimentaryFacies`、`SandBody`、`GeologicalProcess`
- **宾语类型**：`StratigraphicUnit`、`SandBody`、`Reservoir`、`Trap`、`OilGasReservoir`
- **必需限定词**：`target_property`
- **业务约束**：不同于controls；影响方向、增减趋势可作为effect_direction限定词。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

#### 成藏关系

##### `petroleum_system_has_source` — 系统包含烃源岩

- **定义**：含油气系统具有特定烃源岩
- **主语类型**：`PetroleumSystem`
- **宾语类型**：`SourceRock`
- **必需限定词**：无额外必需限定词
- **业务约束**：无单独追加的业务约束；仍须遵守全局证据、类型和范围规则。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `petroleum_system_has_reservoir` — 系统包含储层

- **定义**：含油气系统具有特定储层
- **主语类型**：`PetroleumSystem`
- **宾语类型**：`Reservoir`
- **必需限定词**：无额外必需限定词
- **业务约束**：无单独追加的业务约束；仍须遵守全局证据、类型和范围规则。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `petroleum_system_has_seal` — 系统包含盖层

- **定义**：含油气系统具有特定盖层
- **主语类型**：`PetroleumSystem`
- **宾语类型**：`Seal`
- **必需限定词**：无额外必需限定词
- **业务约束**：无单独追加的业务约束；仍须遵守全局证据、类型和范围规则。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `petroleum_system_has_trap` — 系统包含圈闭

- **定义**：含油气系统具有特定圈闭
- **主语类型**：`PetroleumSystem`
- **宾语类型**：`Trap`
- **必需限定词**：无额外必需限定词
- **业务约束**：无单独追加的业务约束；仍须遵守全局证据、类型和范围规则。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `petroleum_system_has_accumulation` — 系统包含油气藏

- **定义**：含油气系统包含特定油气聚集体
- **主语类型**：`PetroleumSystem`
- **宾语类型**：`OilGasReservoir`
- **必需限定词**：无额外必需限定词
- **业务约束**：无单独追加的业务约束；仍须遵守全局证据、类型和范围规则。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `field_contains_accumulation` — 油气田包含油气藏

- **定义**：一个油气田中存在具体油气藏
- **主语类型**：`OilGasField`
- **宾语类型**：`OilGasReservoir`
- **必需限定词**：无额外必需限定词
- **业务约束**：无单独追加的业务约束；仍须遵守全局证据、类型和范围规则。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `accumulation_hosted_in` — 油气藏赋存于储层

- **定义**：油气藏赋存于可储集和渗流的储层中
- **主语类型**：`OilGasReservoir`
- **宾语类型**：`Reservoir`
- **必需限定词**：无额外必需限定词
- **业务约束**：无单独追加的业务约束；仍须遵守全局证据、类型和范围规则。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `accumulation_trapped_by` — 油气藏受圈闭约束

- **定义**：油气聚集受特定圈闭控制
- **主语类型**：`OilGasReservoir`
- **宾语类型**：`Trap`
- **必需限定词**：无额外必需限定词
- **业务约束**：无单独追加的业务约束；仍须遵守全局证据、类型和范围规则。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `accumulation_sealed_by` — 油气藏由盖层封隔

- **定义**：特定盖层为油气藏提供封隔条件
- **主语类型**：`OilGasReservoir`
- **宾语类型**：`Seal`
- **必需限定词**：无额外必需限定词
- **业务约束**：无单独追加的业务约束；仍须遵守全局证据、类型和范围规则。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `accumulation_charged_from` — 油气藏来源于烃源岩

- **定义**：证据支持该油气藏的油气由特定烃源岩供给
- **主语类型**：`OilGasReservoir`
- **宾语类型**：`SourceRock`
- **必需限定词**：无额外必需限定词
- **业务约束**：必须有油源对比、成藏解释或直接文献断言；不能因同一盆地就推断。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

#### 井与样品、地层对比

##### `well_intersects_unit` — 井钻遇地层

- **定义**：井钻遇指定地层或油层组
- **主语类型**：`Well`
- **宾语类型**：`StratigraphicUnit`
- **必需限定词**：无额外必需限定词
- **业务约束**：有深度时在测量/观测记录中保留，不能仅凭井位推断。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `sample_from_well` — 样品取自井

- **定义**：样品由某井获取
- **主语类型**：`Sample`
- **宾语类型**：`Well`
- **必需限定词**：无额外必需限定词
- **业务约束**：同一实物样品原则上来源井唯一；复合样品另行标记。
- **对称性**：否；**基数**：`at_most_one_target_per_sample`。

##### `sample_from_unit` — 样品取自地层

- **定义**：样品所属的地层层位
- **主语类型**：`Sample`
- **宾语类型**：`StratigraphicUnit`
- **必需限定词**：无额外必需限定词
- **业务约束**：需有井深或地层归属证据；跨层混合样品应保留不确定性。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `sample_represents` — 样品代表地质对象

- **定义**：样品在论文中用于表征某个地质对象
- **主语类型**：`Sample`
- **宾语类型**：`SourceRock`、`Reservoir`、`Seal`、`SandBody`
- **必需限定词**：无额外必需限定词
- **业务约束**：必须有明确采样对象对应关系，不能只凭相同岩性匹配。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

##### `correlates_with` — 地层对比对应

- **定义**：不同地区或划分体系的地层单元在证据支持下对比对应
- **主语类型**：`StratigraphicUnit`
- **宾语类型**：`StratigraphicUnit`
- **必需限定词**：`spatial_scope_id`
- **业务约束**：对比对应不等于sameAs或完全同一实体；需保留依据。 对比关系应具有同一对比目的与可定位的地层证据；禁止自指。
- **对称性**：是（同一适用范围内）；**基数**：`many_to_many_unless_context_specific`。

##### `facies_transitions_to` — 沉积相过渡为

- **定义**：在特定空间方向或层序中沉积相发生过渡
- **主语类型**：`SedimentaryFacies`
- **宾语类型**：`SedimentaryFacies`
- **必需限定词**：`spatial_scope_id`、`transition_axis`
- **业务约束**：transition_axis=lateral/vertical；应注明方向和地层/研究区，不视为无条件分类关系。
- **对称性**：否；**基数**：`many_to_many_unless_context_specific`。

## 6. 测量属性与单位规范

### 6.1 支持的测量类型

| property_name | 中文名称 | 统一单位 | 适用实体 | 值域约束 |
| --- | --- | --- | --- | --- |
| `porosity` | 孔隙度 | `%` | `Reservoir`、`Sample` | 0 ～ 100 |
| `permeability` | 渗透率 | `mD` | `Reservoir`、`Seal`、`Sample` | ≥ 0 |
| `thickness` | 厚度 | `m` | `StratigraphicUnit`、`SandBody`、`Reservoir`、`Seal` | ≥ 0 |
| `depth` | 深度 | `m` | `Well`、`Sample`、`StratigraphicUnit`、`OilGasReservoir` | ≥ 0 |
| `toc` | 总有机碳 | `wt%` | `SourceRock`、`Sample` | ≥ 0 |
| `ro` | 镜质体反射率 | `%` | `SourceRock`、`Sample` | ≥ 0 |
| `pressure` | 压力 | `MPa` | `OilGasReservoir`、`Sample` | ≥ 0 |
| `delta13c` | 碳同位素值 | `‰` | `SourceRock`、`Sample`、`OilGasReservoir` | 按原文与方法审核 |
| `biomarker_ratio` | 生物标志化合物比值 | `dimensionless` | `SourceRock`、`Sample` | 按原文与方法审核 |

### 6.2 Measurement 记录结构

| 类别 | 字段 | 说明 |
| --- | --- | --- |
| 必填 | `measurement_id` | 记录标识、目标、测量项目、数值、单位或证据引用 |
| 必填 | `target_entity_id` | 记录标识、目标、测量项目、数值、单位或证据引用 |
| 必填 | `property_name` | 记录标识、目标、测量项目、数值、单位或证据引用 |
| 必填 | `value` | 记录标识、目标、测量项目、数值、单位或证据引用 |
| 必填 | `unit` | 记录标识、目标、测量项目、数值、单位或证据引用 |
| 必填 | `evidence_ids` | 记录标识、目标、测量项目、数值、单位或证据引用 |
| 可选 | `sample_id` | 视样品、井、层位、方法和统计口径填写 |
| 可选 | `well_id` | 视样品、井、层位、方法和统计口径填写 |
| 可选 | `stratigraphic_unit_id` | 视样品、井、层位、方法和统计口径填写 |
| 可选 | `method` | 视样品、井、层位、方法和统计口径填写 |
| 可选 | `aggregation` | 视样品、井、层位、方法和统计口径填写 |
| 可选 | `uncertainty` | 视样品、井、层位、方法和统计口径填写 |
| 可选 | `qualifier` | 视样品、井、层位、方法和统计口径填写 |
| 可选 | `original_value` | 视样品、井、层位、方法和统计口径填写 |
| 可选 | `original_unit` | 视样品、井、层位、方法和统计口径填写 |
| 可选 | `depth_interval_m` | 视样品、井、层位、方法和统计口径填写 |
| 可选 | `verification_status` | 视样品、井、层位、方法和统计口径填写 |

> **实现差异**：当前 `validate_schema.py` 仅接受数值型 `value`。虽然属性规范备注提及上下界结构，但区间测量需要先扩展校验器，不能直接用现有脚本验证区间对象。


