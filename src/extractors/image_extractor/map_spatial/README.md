你的流程方向是对的，但现在还需要把“**VLM负责语义判断**”和“**程序负责确定性建图**”再切得更干净一些。尤其是地点方位关系，如果让 VLM 直接生成大量两两关系，很容易关系爆炸、重复和矛盾。

我建议把整个地图类图片抽取流程固定成下面这套。

## 一、总体流程

```text
图片
 ↓
① 图例解析 Legend Parsing
 ↓
② 图中对象/实体抽取 Entity Extraction
 ↓
③ 图例实体 ↔ 图中对象绑定 Legend Grounding
 ↓
④ 根据图例类型生成候选关系类型 Relation Schema Generation
 ↓
⑤ VLM 判断语义关系候选 Relation Candidate Extraction
 ↓
⑥ 地点方位候选筛选 Spatial Candidate Selection
 ↓
⑦ 程序确定具体映射 Deterministic Relation Mapping
 ↓
⑧ 关系约束/去重/冲突消解
 ↓
最终 entities + relations
```

核心原则是：

> **VLM 决定“什么值得连”和“关系语义是什么”；程序决定“具体连谁、是否合法、是否冲突”。**

---

# 二、第一阶段：图例解析

不要一开始直接抽整张图实体，而是首先理解图例。

比如：

```json
{
  "legend_items": [
    {
      "legend_id": "legend_01",
      "label": "颗粒滩",
      "semantic_type": "sedimentary_microfacies",
      "visual_encoding": {
        "fill_color": "橄榄绿色"
      }
    },
    {
      "legend_id": "legend_02",
      "label": "井位",
      "semantic_type": "well",
      "visual_encoding": {
        "symbol": "circle"
      }
    },
    {
      "legend_id": "legend_03",
      "label": "地名",
      "semantic_type": "place",
      "visual_encoding": {
        "symbol": "square"
      }
    }
  ]
}
```

这里实际上是在回答两个问题：

1. 图上有哪些**类别**
2. 每个类别用什么视觉符号表示

这一步不要生成真实实体。

---

# 三、第二阶段：实体抽取

然后根据图例类型去找具体实例。

例如：

```text
井位
 ├─ 陕164
 ├─ 陕157
 ├─ 陕354
 └─ 陕228

地名
 ├─ 横山
 ├─ 靖边
 ├─ 安塞
 └─ 华池

沉积微相
 ├─ 颗粒滩
 ├─ 泥云坪
 └─ 硬石膏结核云坪
```

这里建议每个实体保存一个**图像坐标锚点**。

例如：

```json
{
  "id": "place_jingbian",
  "name": "靖边",
  "type": "place",
  "geometry": {
    "bbox": [451, 490, 511, 530],
    "center": [481, 510]
  }
}
```

地点方位关系以后最好**不要让 VLM直接凭语言判断**，而是由这个 `center` 或 polygon/bbox 计算。

---

# 四、第三阶段：图例实体和图中实体绑定

这个阶段非常关键。

实际上应该建立：

```text
Legend Class
     ↓ instance_of
Image Entity
```

例如：

```text
地名图例
   ├── 横山
   ├── 靖边
   ├── 安塞
   └── 华池

井位图例
   ├── 陕164
   ├── 陕157
   └── 陕354
```

但是这个关系属于**抽取系统内部关系**，不一定最后进知识图谱。

建议保存在中间结果：

```json
{
  "legend_bindings": [
    {
      "legend_id": "legend_place",
      "entity_ids": [
        "place_hengshan",
        "place_jingbian",
        "place_ansai",
        "place_huachi"
      ]
    }
  ]
}
```

---

# 五、第四阶段：根据实体类型自动生成“允许的关系类型”

这一块你现在的思路很好。

不是让 VLM随意创造关系，而是：

> 根据 `source_type + target_type` 自动生成允许的 Relation Schema。

例如：

### 沉积相 × 沉积相

```text
adjacent_to
surrounds
contained_in
transition_to
```

### 井 × 沉积相

```text
located_in
located_near
```

### 气田 × 沉积相

```text
located_in
overlaps
along_boundary_of
```

### 剖面线 × 地质单元

```text
crosses
```

### 储层参数区 × 参数

```text
high_value_of
low_value_of
```

### 地点 × 地点

只允许：

```text
north_of
south_of
east_of
west_of
northeast_of
northwest_of
southeast_of
southwest_of
```

因此可以维护一个：

```python
RELATION_SCHEMA = {
    ("well", "sedimentary_microfacies"): [
        "located_in",
        "located_near"
    ],

    ("place", "place"): [
        "north_of",
        "south_of",
        "east_of",
        "west_of",
        "northeast_of",
        "northwest_of",
        "southeast_of",
        "southwest_of"
    ],

    ("profile_line", "depositional_zone"): [
        "crosses"
    ]
}
```

这样比让大模型自己创造：

```text
位于东北侧
东北方向上
处于东北部
东北邻近
```

规范得多。

---

# 六、VLM不负责直接生成所有地点关系

这是你这次最需要修改的地方。

你现在应该让 VLM 输出的不是：

```text
横山 东北 靖边
横山 北 安塞
横山 东北 华池
……
```

而是：

> **哪些地点适合参与空间方位关系。**

例如：

```json
{
  "spatial_relation_candidates": [
    "place_hengshan",
    "place_jingbian",
    "place_ansai",
    "place_huachi"
  ]
}
```

甚至进一步让 VLM 给一个重要性：

```json
{
  "spatial_relation_candidates": [
    {
      "entity_id": "place_jingbian",
      "priority": 1.0,
      "reason": "研究区中心地名"
    },
    {
      "entity_id": "place_hengshan",
      "priority": 0.9
    },
    {
      "entity_id": "place_ansai",
      "priority": 0.9
    },
    {
      "entity_id": "place_huachi",
      "priority": 0.7
    }
  ]
}
```

然后程序处理。

---

# 七、地点方位关系不要全排列

例如四个地点：

```text
横山
靖边
安塞
华池
```

如果两两全部建关系，理论上：

```text
4 × 3 = 12
```

已经很多了。

如果有 20 个地点：

```text
20 × 19 = 380
```

知识图谱会完全被空间关系淹没。

所以你提出：

> 每个实体最多一条出度 + 一条入度

这个约束非常合理。

---

# 八、地点方位建议构建“空间骨架”，而不是完全图

最终更适合这样的：

```text
横山
 ↓ 西南
靖边
 ↓ 东南
安塞
 ↓ 西
华池
```

或者：

```text
横山 --西南--> 靖边
靖边 --东南--> 安塞
华池 --东北--> 靖边
```

但你规定：

```text
out_degree <= 1
in_degree <= 1
```

那么最好进一步规定：

> **禁止环路。**

否则可能出现：

```text
A → B
B → C
C → A
```

虽然每个节点都只有一入一出，但语义上没有意义。

因此空间关系子图最好满足：

```text
out_degree ≤ 1
in_degree ≤ 1
acyclic = true
```

数学上最终就会形成：

> **若干条有向路径。**

这非常适合你的场景。

---

# 九、具体地点关系怎么选

不要随机选。

我推荐程序采用：

## “最近邻 + 信息覆盖”策略

假设候选地点：

```text
横山
靖边
安塞
华池
```

都有：

```text
center=(x,y)
```

计算所有距离：

```python
d(A, B)
```

优先连接相邻地点。

例如可能形成：

```text
横山 → 靖边
靖边 → 安塞
安塞 → 华池
```

然后根据坐标差判断八方位。

这样只产生：

```text
N - 1
```

条关系。

4个地点：

```text
3条关系
```

10个地点：

```text
9条左右
```

而不是：

```text
N(N-1)
```

---

# 十、八方位由程序计算，不让VLM判断

假设：

```text
A(x1,y1)
B(x2,y2)
```

注意图像坐标：

```text
x向右增加
y向下增加
```

计算：

```python
dx = x1 - x2
dy = y1 - y2
```

如果我们表达：

```text
A relative_to B
```

可以用角度：

```python
angle = atan2(-dy, dx)
```

然后：

```text
-22.5° ~ 22.5°       东
22.5° ~ 67.5°        东北
67.5° ~ 112.5°       北
112.5° ~ 157.5°      西北
157.5° ~ 180°
-180° ~ -157.5°      西
-157.5° ~ -112.5°    西南
-112.5° ~ -67.5°     南
-67.5° ~ -22.5°      东南
```

于是完全确定。

例如：

```text
横山 → 靖边
```

坐标计算结果：

```text
横山在靖边东北
```

最终生成：

```json
{
  "source": "横山",
  "relation": "northeast_of",
  "target": "靖边"
}
```

而不再反向生成：

```text
靖边 southwest_of 横山
```

因为你现在明确要求**单向**。

---

# 十一、建议增加“方向显著性阈值”

仅靠八等分有一个问题。

例如：

```text
dx = 100
dy = 5
```

明显应该是：

```text
东
```

而不是东北。

所以可以设置比例阈值。

例如：

```python
ratio = abs(dx) / abs(dy)
```

如果：

```text
abs(dx) > 2 * abs(dy)
```

直接：

```text
东 / 西
```

如果：

```text
abs(dy) > 2 * abs(dx)
```

直接：

```text
北 / 南
```

否则：

```text
东北 / 东南 / 西北 / 西南
```

这个比固定45°更符合地图阅读习惯。

---

# 十二、VLM 输出格式建议改成这样

VLM 的地图抽取结果可以增加：

```json
{
  "entities": [],
  "relation_candidates": [],
  "spatial_mapping": {
    "enabled": true,
    "candidate_entity_types": [
      "place"
    ],
    "candidate_entities": [
      {
        "entity_id": "place_hengshan",
        "priority": 0.92
      },
      {
        "entity_id": "place_jingbian",
        "priority": 1.0
      },
      {
        "entity_id": "place_ansai",
        "priority": 0.91
      },
      {
        "entity_id": "place_huachi",
        "priority": 0.72
      }
    ]
  }
}
```

注意：

**VLM这里不输出东北、西南。**

只输出：

```text
“这些实体应该进入地点空间映射。”
```

---

# 十三、程序的地点映射阶段

建议：

```python
def build_place_spatial_relations(entities):
    candidates = select_spatial_candidates(entities)

    # 1. 获取中心坐标
    positions = get_centers(candidates)

    # 2. 构造距离候选边
    edges = build_distance_edges(positions)

    # 3. 优先最近邻
    edges.sort(key=lambda x: x.distance)

    # 4. 受约束选边
    selected = []

    for edge in edges:
        if out_degree[edge.source] >= 1:
            continue

        if in_degree[edge.target] >= 1:
            continue

        if creates_cycle(selected, edge):
            continue

        selected.append(edge)

    # 5. 几何确定方位
    for edge in selected:
        edge.relation = calculate_direction(
            edge.source.center,
            edge.target.center
        )

    return selected
```

---

# 十四、但“最近邻”还要加一个规则

纯最近邻可能变成：

```text
A → B
C → B
```

第二条因为：

```text
B indegree 已经1
```

就不能连。

所以实际上更建议使用：

> **最小生成树 MST → 定向 → 度约束修剪**

或者更简单：

> 从核心地点出发做“空间链”。

例如由VLM给出：

```json
{
  "anchor_entity": "靖边"
}
```

程序：

```text
找靖边最近的地点 A
找 A 最近的未访问地点 B
找 B 最近的未访问地点 C
……
```

最终形成：

```text
A → 靖边 → B → C
```

天然满足：

```text
indegree <= 1
outdegree <= 1
```

---

# 十五、我更推荐“主地点锚点 + 空间链”

因为地质图一般都有一个核心区域：

比如：

```text
靖边
```

可以由VLM判断：

```json
{
  "spatial_anchor": "place_jingbian"
}
```

然后程序寻找：

```text
横山 → 靖边 → 安塞 → 华池
```

方向再计算：

```text
横山 -[西南]-> 靖边
靖边 -[东南]-> 安塞
安塞 -[西]-> 华池
```

总共只需要3条关系。

这比刚才我们生成12条明显更合理。

---

# 十六、你整个 MapSpatialExtractor 可以最终拆成6层

我建议你的代码结构直接拆成：

```text
MapSpatialExtractor
│
├── 1. LegendParser
│      └── 图例类别、颜色、符号
│
├── 2. MapEntityExtractor
│      └── 实体 + bbox/polygon/center
│
├── 3. LegendGrounder
│      └── 图例 ↔ 实体
│
├── 4. RelationSchemaGenerator
│      └── 根据实体类型生成允许关系
│
├── 5. SemanticRelationExtractor
│      └── VLM判断哪些关系值得抽
│
└── 6. DeterministicRelationAssembler
       │
       ├── containment mapper
       ├── overlap mapper
       ├── adjacency mapper
       ├── crossing mapper
       └── place direction mapper
```

这比一个 Prompt 直接让 VLM：

```text
图片 → entities + relations
```

稳定得多。

---

# 十七、不同关系分别应该由谁负责

这一点你最好在系统设计里明确下来：

| 关系         | VLM |    程序 |
| ---------- | --: | ----: |
| 图例是什么      |   ✅ |       |
| 哪些实体属于某图例  |   ✅ |   ✅验证 |
| 井位于哪个微相    |     |  ✅ 几何 |
| 某区域包围哪个区域  | ✅候选 | ✅几何确认 |
| 两区域相邻      | ✅候选 | ✅边界计算 |
| 剖面线穿过区域    |     |     ✅ |
| 地点是否参与方位关系 |   ✅ |       |
| 地点之间选哪些边   |     |     ✅ |
| 东/西/南/北    |     |     ✅ |
| 是否超过度约束    |     |     ✅ |
| 去重         |     |     ✅ |
| 防止环路       |     |     ✅ |

也就是：

> **语义由模型判断，空间拓扑尽可能由程序计算。**

---

# 十八、最终地点映射规则

你这个项目我建议最后定死成下面 7 条：

```text
1. 只有 VLM 标记为 spatial_candidate 的 place 实体参与映射。

2. place-place 方位关系只允许：
   north_of
   south_of
   east_of
   west_of
   northeast_of
   northwest_of
   southeast_of
   southwest_of

3. 方位由实体 bbox/polygon 的 center 通过程序计算，
   不由 VLM 直接生成。

4. 地点关系为单向关系，
   不自动创建逆关系。

5. 每个 place：
   out_degree <= 1
   in_degree <= 1

6. 禁止 cycle。

7. 优先连接空间距离较近、VLM priority 较高、
   且能提高整体地点覆盖率的实体。
```

这样最终地点子图就从原来的：

```text
4个地点
12条关系
```

压缩成类似：

```text
4个地点
3条关系
```

而且表达的信息基本没有损失。

---

你这套流程的关键思想可以最后总结成一句：

> **图例负责定义“有哪些语义类别”，VLM负责识别“图中有哪些实体以及哪些实体值得建立关系”，关系 Schema 根据实体类型自动生成；具体的包含、相交、相邻及地点八方位关系由几何信息和确定性程序完成，其中地点方位仅构建稀疏单向空间骨架，每个地点最多一条入边和一条出边，避免空间关系全连接。**

这个版本很适合作为你 `map_spatial` 抽取器的正式设计原则。
