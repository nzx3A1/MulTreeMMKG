//# 油气标准标准化对象知识图谱结构说明
//## 一、节点（实体）分层定义
//图谱共2类节点标签，总节点66个：
//1. **ConceptCategory（概念分类节点）**
//    数量：1个
//    作用：作为顶层分类汇总节点，承载某一大类油气专业概念集合，用于归集同领域下所有实体概念。
//    - 内置唯一ID：节点主键标识
//    - name: 概念分类名字
//2. **EntityConcept（实体概念节点）**
//    数量：65个
//    作用：存储油气勘探开发专业标准内的具体专业术语、地质对象、工程指标（示例：层序界面），每个节点附带完整业务属性：
//    - `zhName`：中文标准术语名（如“层序界面”）
//    - `category`：所属大类文本标识
//    - `schema`：对应标准英文术语标识（SequenceBoundary）
//    - `description`：专业释义
//    - `examples`：行业常用代号、实例数组
//    - `updatedAt`：数据更新时间戳
//    - 内置唯一ID：节点主键标识
//
//## 二、关系（关联边）定义，总关系298条
//图谱包含两类业务关系，区分**分类归属**与**实体间业务关联**：
//### 1. BELONGS_TO_CATEGORY（归属分类关系，共1条）
//- 流向：`EntityConcept` → `ConceptCategory`
//- 业务含义：单个专业实体概念归属于某一顶层领域大类
//- 示例：「层序界面」-[BELONGS_TO_CATEGORY]->「地层与地质时代」，代表层序界面属于地层与地质时代大类下的专业概念。
//
//### 2. SCHEMA_RELATION（实体业务关联关系，共297条）
//- 流向：`EntityConcept` ↔ `EntityConcept`（双向业务关联）
//- 业务含义：描述两个专业实体之间的油气地质/工程逻辑关联，如地层互层、上下层级、包含、控制、界面接触等专业语义（如前文`interbedded_with`互层关系可封装在此类边内）。
//- 作用：支撑后续标准比对、语义检索、GraphRAG查询，实现多标准间对象关联溯源。



// ================================
// 0. 创建唯一约束
// ================================

CREATE CONSTRAINT entity_concept_schema_unique IF NOT EXISTS
FOR (n:EntityConcept)
REQUIRE n.schema IS UNIQUE;

CREATE CONSTRAINT concept_category_name_unique IF NOT EXISTS
FOR (n:ConceptCategory)
REQUIRE n.name IS UNIQUE;


// ================================
// 1. 创建实体概念节点
// ================================

UNWIND [

  // ----------------
  // 1. 地理与构造单元
  // ----------------
  {
    schema: "Basin",
    zhName: "盆地",
    category: "地理与构造单元",
    description: "具有相对独立沉积、构造演化和油气成藏背景的大型地质空间单元，是石油地质研究中的核心区域实体。",
    examples: ["鄂尔多斯盆地", "四川盆地", "塔里木盆地"]
  },
  {
    schema: "StructuralUnit",
    zhName: "构造单元",
    category: "地理与构造单元",
    description: "盆地内部按照构造形态、构造位置或构造演化特征划分出的次级地质单元。",
    examples: ["伊陕斜坡", "天环坳陷", "中央隆起带"]
  },
  {
    schema: "Depression",
    zhName: "凹陷/坳陷",
    category: "地理与构造单元",
    description: "相对于周缘构造位置较低、沉积厚度较大、常为烃源岩发育有利区的负向构造单元。",
    examples: ["济阳坳陷", "黄骅坳陷", "天环坳陷"]
  },
  {
    schema: "Sag",
    zhName: "洼陷",
    category: "地理与构造单元",
    description: "通常指坳陷内部更局部、更低洼的沉积中心或构造低部位，是细粒沉积和烃源岩发育的重要区域。",
    examples: ["渤南洼陷", "苏桥洼陷", "辽中洼陷"]
  },
  {
    schema: "Uplift",
    zhName: "隆起",
    category: "地理与构造单元",
    description: "相对于周边地区构造位置较高的正向构造单元，常控制地层剥蚀、圈闭形成和油气运移方向。",
    examples: ["伊盟隆起", "渤中隆起", "塔中隆起"]
  },
  {
    schema: "Slope",
    zhName: "斜坡",
    category: "地理与构造单元",
    description: "位于隆起与坳陷之间的倾斜过渡带，常是砂体展布、油气运移和岩性圈闭发育的重要区域。",
    examples: ["伊陕斜坡", "北部斜坡带", "南部斜坡区"]
  },
  {
    schema: "StructuralBelt",
    zhName: "构造带",
    category: "地理与构造单元",
    description: "由一组具有相似构造变形特征的构造要素组成的带状区域，如逆冲带、褶皱带、断裂构造带等。",
    examples: ["西缘逆冲带", "贺兰－六盘山构造带", "断裂构造带"]
  },
  {
    schema: "Fault",
    zhName: "断层",
    category: "地理与构造单元",
    description: "岩层或地质体沿破裂面发生明显位移形成的构造实体，可影响地层展布、油气运移和圈闭形成。",
    examples: ["F1断层", "郯庐断裂", "正断层"]
  },
  {
    schema: "FaultZone",
    zhName: "断裂带",
    category: "地理与构造单元",
    description: "由多条断层及其相关破碎带、裂缝带共同组成的带状构造区域，常作为油气运移通道或封堵边界。",
    examples: ["郯庐断裂带", "贺兰－六盘山断裂带", "边界断裂带"]
  },
  {
    schema: "Block",
    zhName: "区块",
    category: "地理与构造单元",
    description: "油气勘探开发中按照地理位置、构造位置或开发管理需要划分的研究或开发单元。",
    examples: ["苏里格区块", "塔中区块", "长庆油田某开发区块"]
  },
  {
    schema: "StudyArea",
    zhName: "研究区",
    category: "地理与构造单元",
    description: "论文、报告或项目中限定的具体研究范围，可以是盆地、凹陷、区块、剖面或井区。",
    examples: ["鄂尔多斯盆地东南部研究区", "川东北研究区", "塔里木盆地塔中地区"]
  },


  // ----------------
  // 2. 地层与地质时代
  // ----------------
  {
    schema: "GeologicAge",
    zhName: "地质时代",
    category: "地层与地质时代",
    description: "表示地层形成或地质事件发生的时间单位，可包括宙、代、纪、世、期等不同层级。",
    examples: ["三叠纪", "侏罗纪", "寒武纪"]
  },
  {
    schema: "Formation",
    zhName: "组",
    category: "地层与地质时代",
    description: "岩石地层单位中的基本单位，通常具有相对稳定的岩性组合和区域可识别性。",
    examples: ["延长组", "马家沟组", "龙马溪组"]
  },
  {
    schema: "StratigraphicMember",
    zhName: "段",
    category: "地层与地质时代",
    description: "组以下的次一级地层单位，常用于进一步划分储层、烃源岩或沉积旋回。",
    examples: ["长7段", "长8段", "马五段"]
  },
  {
    schema: "SubMember",
    zhName: "亚段/小层",
    category: "地层与地质时代",
    description: "段以下更精细的地层划分单元，常用于储层精细对比和小层划分。",
    examples: ["长7三亚段", "马五5小层", "长6一小层"]
  },
  {
    schema: "Bed",
    zhName: "层/单层",
    category: "地层与地质时代",
    description: "最基本的层状地质单元，通常对应一个单独的岩层、砂层、泥岩层或煤层。",
    examples: ["单砂层", "泥岩层", "7号煤层"]
  },
  {
    schema: "ReservoirInterval",
    zhName: "目的层/储层段",
    category: "地层与地质时代",
    description: "勘探开发或储层评价中重点研究的含油气层段或目标储层层位。",
    examples: ["长7储层段", "马五目的层", "盒8储层段"]
  },
  {
    schema: "StratigraphicBoundary",
    zhName: "地层界面",
    category: "地层与地质时代",
    description: "不同地层单元之间的分界面，可反映沉积间断、岩性变化或地质时代转换。",
    examples: ["奥陶系顶界面", "延长组底界面", "组段分界面"]
  },
  {
    schema: "Unconformity",
    zhName: "不整合面",
    category: "地层与地质时代",
    description: "上下两套地层之间存在沉积间断、剥蚀或构造变形差异的地层接触界面。",
    examples: ["加里东不整合面", "印支期不整合面", "区域不整合面"]
  },
  {
    schema: "SequenceStratigraphicUnit",
    zhName: "层序单元",
    category: "地层与地质时代",
    description: "层序地层学中依据基准面变化、沉积旋回和界面特征划分出的地层单元。",
    examples: ["三级层序", "SQ1", "SQ2"]
  },
  {
    schema: "SequenceBoundary",
    zhName: "层序界面",
    category: "地层与地质时代",
    description: "层序单元之间的边界，通常与相对湖平面或海平面变化、沉积间断或侵蚀面有关。",
    examples: ["SB1", "SB2", "三级层序界面"]
  },
  {
    schema: "SystemTract",
    zhName: "体系域",
    category: "地层与地质时代",
    description: "层序内部根据沉积充填样式和基准面变化阶段划分的沉积组合单元。",
    examples: ["低位体系域", "湖侵体系域", "高位体系域"]
  },


  // ----------------
  // 3. 岩石、矿物与沉积体系
  // ----------------
  {
    schema: "Lithology",
    zhName: "岩性",
    category: "岩石、矿物与沉积体系",
    description: "岩石类型及其基本物质组成特征，是描述地层、储层和沉积环境的重要实体。",
    examples: ["砂岩", "泥岩", "灰岩", "白云岩"]
  },
  {
    schema: "Mineral",
    zhName: "矿物",
    category: "岩石、矿物与沉积体系",
    description: "组成岩石的基本矿物成分，可影响储层物性、成岩作用和脆性特征。",
    examples: ["石英", "长石", "方解石", "黏土矿物"]
  },
  {
    schema: "RockTexture",
    zhName: "岩石结构/组构",
    category: "岩石、矿物与沉积体系",
    description: "岩石中颗粒、晶体或组分的大小、形态、排列和接触关系等结构特征。",
    examples: ["粒状结构", "泥晶结构", "交代结构", "碎屑结构"]
  },
  {
    schema: "SedimentaryStructure",
    zhName: "沉积构造",
    category: "岩石、矿物与沉积体系",
    description: "沉积过程中形成的层理、波痕、冲刷面等构造现象，可用于判断沉积环境和水动力条件。",
    examples: ["交错层理", "水平层理", "波痕", "冲刷面"]
  },
  {
    schema: "SedimentaryFacies",
    zhName: "沉积相",
    category: "岩石、矿物与沉积体系",
    description: "在一定沉积环境下形成的岩性、结构、构造和古生物特征组合。",
    examples: ["三角洲相", "湖泊相", "河流相", "碳酸盐台地相"]
  },
  {
    schema: "Subfacies",
    zhName: "亚相",
    category: "岩石、矿物与沉积体系",
    description: "沉积相内部的次一级划分，用于更精细描述沉积环境和砂体分布。",
    examples: ["三角洲前缘亚相", "滨浅湖亚相", "半深湖亚相"]
  },
  {
    schema: "Microfacies",
    zhName: "微相",
    category: "岩石、矿物与沉积体系",
    description: "亚相内部更精细的沉积单元，常用于储层非均质性分析和砂体精细刻画。",
    examples: ["水下分流河道微相", "河口坝微相", "席状砂微相"]
  },
  {
    schema: "DepositionalEnvironment",
    zhName: "沉积环境",
    category: "岩石、矿物与沉积体系",
    description: "沉积物形成时的自然环境条件，包括水体类型、能量条件、气候背景和构造背景等。",
    examples: ["浅湖环境", "半深湖环境", "滨浅海环境", "三角洲前缘环境"]
  },
  {
    schema: "DepositionalSystem",
    zhName: "沉积体系",
    category: "岩石、矿物与沉积体系",
    description: "由若干相关沉积相、亚相和微相组成的沉积组合系统，反映区域沉积过程和沉积格局。",
    examples: ["三角洲沉积体系", "扇三角洲沉积体系", "浊积扇沉积体系"]
  },
  {
    schema: "Diagenesis",
    zhName: "成岩作用",
    category: "岩石、矿物与沉积体系",
    description: "沉积物埋藏后到变质作用前经历的物理、化学和生物改造过程。",
    examples: ["压实作用", "胶结作用", "溶蚀作用", "交代作用"]
  },
  {
    schema: "DiageneticStage",
    zhName: "成岩阶段",
    category: "岩石、矿物与沉积体系",
    description: "根据埋藏深度、温度、成岩矿物和孔隙演化特征划分的成岩演化时期。",
    examples: ["早成岩A期", "早成岩B期", "中成岩A期"]
  },
  {
    schema: "DiageneticFacies",
    zhName: "成岩相",
    category: "岩石、矿物与沉积体系",
    description: "由特定成岩作用、成岩矿物组合和孔隙演化特征共同表征的成岩类型单元。",
    examples: ["溶蚀相", "碳酸盐胶结相", "压实致密相"]
  },


  // ----------------
  // 4. 油气成藏要素
  // ----------------
  {
    schema: "PetroleumSystem",
    zhName: "含油气系统/成藏系统",
    category: "油气成藏要素",
    description: "由烃源岩、储层、盖层、圈闭、运移和保存条件等共同构成的油气生成、运移与聚集系统。",
    examples: ["鄂尔多斯盆地延长组成藏系统", "四川盆地下古生界含气系统"]
  },
  {
    schema: "SourceRock",
    zhName: "烃源岩",
    category: "油气成藏要素",
    description: "富含有机质并能够生成油气的岩石，是油气成藏的物质基础。",
    examples: ["长7烃源岩", "龙马溪组页岩", "湖相泥页岩"]
  },
  {
    schema: "Reservoir",
    zhName: "储层/储集层",
    category: "油气成藏要素",
    description: "具有一定孔隙、裂缝或洞穴空间，能够储集和渗流油气的岩层。",
    examples: ["长6砂岩储层", "马五碳酸盐岩储层", "页岩储层"]
  },
  {
    schema: "Seal",
    zhName: "盖层",
    category: "油气成藏要素",
    description: "覆盖在储层之上或周围、能够阻止油气逸散的低渗透性岩层。",
    examples: ["泥岩盖层", "膏盐岩盖层", "致密灰岩盖层"]
  },
  {
    schema: "Trap",
    zhName: "圈闭",
    category: "油气成藏要素",
    description: "能够阻止油气继续运移并使油气聚集保存的地质空间结构。",
    examples: ["构造圈闭", "岩性圈闭", "地层圈闭", "复合圈闭"]
  },
  {
    schema: "MigrationPathway",
    zhName: "运移通道",
    category: "油气成藏要素",
    description: "油气从烃源岩向储层或圈闭运移过程中经过的通道或介质。",
    examples: ["断层运移通道", "砂体输导层", "裂缝通道"]
  },
  {
    schema: "CarrierSystem",
    zhName: "输导体系",
    category: "油气成藏要素",
    description: "由断层、裂缝、砂体、不整合面等共同组成的油气输导网络。",
    examples: ["断砂组合输导体系", "裂缝输导体系", "不整合面输导体系"]
  },
  {
    schema: "AccumulationProcess",
    zhName: "成藏过程",
    category: "油气成藏要素",
    description: "油气从生成、排出、运移、聚集到保存的动态地质过程。",
    examples: ["生烃", "排烃", "油气运移", "油气聚集"]
  },


  // ----------------
  // 5. 油气藏与流体
  // ----------------
  {
    schema: "OilGasField",
    zhName: "油气田",
    category: "油气藏与流体",
    description: "在一定区域内由一个或多个油气藏组成、具有工业开发价值的油气聚集区。",
    examples: ["苏里格气田", "大庆油田", "长庆油田"]
  },
  {
    schema: "OilGasReservoir",
    zhName: "油气藏",
    category: "油气藏与流体",
    description: "油气在单一圈闭或储层单元中聚集形成的地质体，是油气田的基本组成单元。",
    examples: ["长8油藏", "马五气藏", "岩性油藏"]
  },
  {
    schema: "Hydrocarbon",
    zhName: "烃类/油气",
    category: "油气藏与流体",
    description: "由碳氢化合物组成的油气物质，包括原油、天然气、凝析油等。",
    examples: ["原油", "天然气", "凝析油", "页岩气"]
  },
  {
    schema: "FluidContact",
    zhName: "流体界面",
    category: "油气藏与流体",
    description: "油、气、水等不同流体在储层中的接触界面，常用于判断油气藏边界和流体分布。",
    examples: ["油水界面", "气水界面", "油气界面"]
  },


  // ----------------
  // 6. 储层结构与评价实体
  // ----------------
  {
    schema: "PoreStructure",
    zhName: "孔隙结构",
    category: "储层结构与评价实体",
    description: "储层中孔隙、喉道及其连通关系的几何结构特征，直接影响储集能力和渗流能力。",
    examples: ["粒间孔", "溶蚀孔", "微孔", "孔喉网络"]
  },
  {
    schema: "Fracture",
    zhName: "裂缝",
    category: "储层结构与评价实体",
    description: "岩石中因构造应力、成岩收缩或溶蚀作用形成的破裂面，可改善储层渗流能力。",
    examples: ["构造裂缝", "成岩裂缝", "高角度裂缝"]
  },
  {
    schema: "Cavity",
    zhName: "溶蚀孔洞/洞穴",
    category: "储层结构与评价实体",
    description: "碳酸盐岩或其他可溶岩石中因溶蚀作用形成的较大孔洞空间，是重要的储集空间类型。",
    examples: ["溶洞", "溶蚀孔", "洞穴型储集空间"]
  },
  {
    schema: "SweetSpot",
    zhName: "甜点",
    category: "储层结构与评价实体",
    description: "在非常规或复杂油气藏中，储层品质、含油气性、可压裂性或产能较优的有利区域或层段。",
    examples: ["页岩气甜点段", "致密油甜点区", "优质储层甜点区"]
  },


  // ----------------
  // 7. 地球化学与有机质实体
  // ----------------
  {
    schema: "OrganicMatter",
    zhName: "有机质",
    category: "地球化学与有机质实体",
    description: "沉积岩中来源于生物残体并可转化为油气的有机组分，是评价烃源岩的重要对象。",
    examples: ["腐泥型有机质", "腐殖型有机质", "藻类有机质"]
  },
  {
    schema: "KerogenType",
    zhName: "干酪根类型",
    category: "地球化学与有机质实体",
    description: "根据有机质来源、显微组分和生烃潜力划分的干酪根类别。",
    examples: ["I型干酪根", "II1型干酪根", "III型干酪根"]
  },
  {
    schema: "Biomarker",
    zhName: "生物标志化合物",
    category: "地球化学与有机质实体",
    description: "能够反映有机质来源、沉积环境、成熟度和油源对比信息的特征有机化合物。",
    examples: ["甾烷", "萜烷", "藿烷", "伽马蜡烷"]
  },
  {
    schema: "Isotope",
    zhName: "同位素",
    category: "地球化学与有机质实体",
    description: "用于地球化学示踪、成因判别、古环境恢复和流体来源分析的同位素指标。",
    examples: ["δ13C", "δ18O", "87Sr/86Sr", "碳同位素"]
  },


  // ----------------
  // 8. 勘探开发与资料类型
  // ----------------
  {
    schema: "SeismicProfile",
    zhName: "地震剖面",
    category: "勘探开发与资料类型",
    description: "通过地震勘探获得的地下反射结构剖面，用于识别断层、构造、地层界面和沉积体。",
    examples: ["过井地震剖面", "二维地震剖面", "三维地震剖面"]
  },
  {
    schema: "Well",
    zhName: "井",
    category: "勘探开发与资料类型",
    description: "用于钻探、取心、测井、试油气或开发生产的钻井实体。",
    examples: ["苏6井", "鄂123井", "塔中1井"]
  },
  {
    schema: "WellLog",
    zhName: "测井资料",
    category: "勘探开发与资料类型",
    description: "通过测井方法获得的井筒地球物理响应数据，用于解释岩性、物性、含油气性和地层界面。",
    examples: ["自然伽马曲线", "电阻率测井", "声波时差测井", "密度测井"]
  },
  {
    schema: "Core",
    zhName: "岩心",
    category: "勘探开发与资料类型",
    description: "从井下取出的柱状岩石样品，可直接观察岩性、沉积构造、裂缝和储层特征。",
    examples: ["取心段岩心", "砂岩岩心", "泥页岩岩心"]
  },
  {
    schema: "ThinSection",
    zhName: "薄片",
    category: "勘探开发与资料类型",
    description: "将岩石样品磨制成薄片后在显微镜下观察，用于研究矿物组成、孔隙类型和成岩特征。",
    examples: ["铸体薄片", "普通薄片", "阴极发光薄片"]
  },
  {
    schema: "Outcrop",
    zhName: "露头",
    category: "勘探开发与资料类型",
    description: "地表出露的岩层或地质体，可用于观察沉积构造、地层接触关系和构造变形。",
    examples: ["野外露头", "剖面露头", "河谷露头"]
  },
  {
    schema: "GeologicalMap",
    zhName: "地质图",
    category: "勘探开发与资料类型",
    description: "表达地层、构造、岩性、沉积相或油气分布等空间信息的图件资料。",
    examples: ["构造图", "沉积相图", "地层厚度图", "油藏分布图"]
  },
  {
    schema: "ProductionTest",
    zhName: "测试/试采",
    category: "勘探开发与资料类型",
    description: "通过试油、试气、试采或产能测试获取储层流体性质、产量和开发潜力的资料类型。",
    examples: ["试油", "试气", "产能测试", "试采资料"]
  },


  // ----------------
  // 9. 实验样品与测试
  // ----------------
  {
    schema: "Sample",
    zhName: "样品",
    category: "实验样品与测试",
    description: "用于实验分析、测试或观察的岩石、流体、有机质或矿物样品。",
    examples: ["岩心样品", "烃源岩样品", "原油样品", "薄片样品"]
  },
  {
    schema: "Experiment",
    zhName: "实验",
    category: "实验样品与测试",
    description: "针对样品开展的测试或分析过程，用于获取物性、地球化学、矿物学或储层结构参数。",
    examples: ["压汞实验", "热解实验", "X射线衍射实验", "扫描电镜实验"]
  },
  {
    schema: "AnalyticalMethod",
    zhName: "分析方法",
    category: "实验样品与测试",
    description: "用于样品测试、数据分析和地质解释的具体技术方法或实验手段。",
    examples: ["TOC分析", "Rock-Eval热解", "XRD分析", "气相色谱分析"]
  }

] AS row

MERGE (c:EntityConcept {schema: row.schema})
SET
  c.zhName = row.zhName,
  c.category = row.category,
  c.description = row.description,
  c.examples = row.examples,
  c.updatedAt = datetime()

MERGE (cat:ConceptCategory {name: row.category})
SET cat.updatedAt = datetime()

MERGE (c)-[:BELONGS_TO_CATEGORY]->(cat);







// =====================================================
// 实体概念间可能关系 Schema
// 说明：
// 使用统一关系类型 :SCHEMA_RELATION
// 关系属性只保留英文 relationEn 和中文 relationZh
// =====================================================

UNWIND [

  // =====================================================
  // 1. 地理与构造单元内部关系
  // =====================================================

  {source:"Basin", relationEn:"CONTAINS", relationZh:"包含", target:"StructuralUnit"},
  {source:"Basin", relationEn:"CONTAINS", relationZh:"包含", target:"Depression"},
  {source:"Basin", relationEn:"CONTAINS", relationZh:"包含", target:"Sag"},
  {source:"Basin", relationEn:"CONTAINS", relationZh:"包含", target:"Uplift"},
  {source:"Basin", relationEn:"CONTAINS", relationZh:"包含", target:"Slope"},
  {source:"Basin", relationEn:"CONTAINS", relationZh:"包含", target:"StructuralBelt"},
  {source:"Basin", relationEn:"CONTAINS", relationZh:"包含", target:"FaultZone"},
  {source:"Basin", relationEn:"CONTAINS", relationZh:"包含", target:"Block"},
  {source:"Basin", relationEn:"CONTAINS", relationZh:"包含", target:"StudyArea"},

  {source:"StructuralUnit", relationEn:"PART_OF", relationZh:"属于", target:"Basin"},
  {source:"StructuralUnit", relationEn:"CONTAINS", relationZh:"包含", target:"Depression"},
  {source:"StructuralUnit", relationEn:"CONTAINS", relationZh:"包含", target:"Sag"},
  {source:"StructuralUnit", relationEn:"CONTAINS", relationZh:"包含", target:"Uplift"},
  {source:"StructuralUnit", relationEn:"CONTAINS", relationZh:"包含", target:"Slope"},
  {source:"StructuralUnit", relationEn:"CONTAINS", relationZh:"包含", target:"StructuralBelt"},
  {source:"StructuralUnit", relationEn:"CONTAINS", relationZh:"包含", target:"FaultZone"},
  {source:"StructuralUnit", relationEn:"CONTAINS", relationZh:"包含", target:"Block"},

  {source:"Depression", relationEn:"PART_OF", relationZh:"属于", target:"Basin"},
  {source:"Depression", relationEn:"CONTAINS", relationZh:"包含", target:"Sag"},
  {source:"Sag", relationEn:"PART_OF", relationZh:"属于", target:"Depression"},

  {source:"Uplift", relationEn:"ADJACENT_TO", relationZh:"邻接", target:"Slope"},
  {source:"Slope", relationEn:"ADJACENT_TO", relationZh:"邻接", target:"Uplift"},
  {source:"Slope", relationEn:"ADJACENT_TO", relationZh:"邻接", target:"Depression"},
  {source:"Slope", relationEn:"TRANSITION_BETWEEN", relationZh:"过渡于", target:"Uplift"},
  {source:"Slope", relationEn:"TRANSITION_BETWEEN", relationZh:"过渡于", target:"Depression"},

  {source:"StructuralBelt", relationEn:"CONTAINS", relationZh:"包含", target:"FaultZone"},
  {source:"StructuralBelt", relationEn:"CONTAINS", relationZh:"包含", target:"Fault"},
  {source:"FaultZone", relationEn:"CONTAINS", relationZh:"包含", target:"Fault"},
  {source:"Fault", relationEn:"PART_OF", relationZh:"属于", target:"FaultZone"},

  {source:"Fault", relationEn:"CONTROLS", relationZh:"控制", target:"StructuralUnit"},
  {source:"Fault", relationEn:"CONTROLS", relationZh:"控制", target:"Block"},
  {source:"FaultZone", relationEn:"CONTROLS", relationZh:"控制", target:"StructuralBelt"},
  {source:"FaultZone", relationEn:"CONTROLS", relationZh:"控制", target:"StructuralUnit"},

  {source:"StudyArea", relationEn:"COVERS", relationZh:"覆盖", target:"Basin"},
  {source:"StudyArea", relationEn:"COVERS", relationZh:"覆盖", target:"StructuralUnit"},
  {source:"StudyArea", relationEn:"COVERS", relationZh:"覆盖", target:"Depression"},
  {source:"StudyArea", relationEn:"COVERS", relationZh:"覆盖", target:"Sag"},
  {source:"StudyArea", relationEn:"COVERS", relationZh:"覆盖", target:"Uplift"},
  {source:"StudyArea", relationEn:"COVERS", relationZh:"覆盖", target:"Slope"},
  {source:"StudyArea", relationEn:"COVERS", relationZh:"覆盖", target:"StructuralBelt"},
  {source:"StudyArea", relationEn:"COVERS", relationZh:"覆盖", target:"Block"},


  // =====================================================
  // 2. 地层与地质时代内部关系
  // =====================================================

  {source:"GeologicAge", relationEn:"DEFINES_AGE_OF", relationZh:"限定时代", target:"Formation"},
  {source:"GeologicAge", relationEn:"DEFINES_AGE_OF", relationZh:"限定时代", target:"StratigraphicMember"},
  {source:"GeologicAge", relationEn:"DEFINES_AGE_OF", relationZh:"限定时代", target:"SubMember"},
  {source:"GeologicAge", relationEn:"DEFINES_AGE_OF", relationZh:"限定时代", target:"Bed"},
  {source:"GeologicAge", relationEn:"DEFINES_AGE_OF", relationZh:"限定时代", target:"SequenceStratigraphicUnit"},

  {source:"Formation", relationEn:"HAS_MEMBER", relationZh:"具有段", target:"StratigraphicMember"},
  {source:"StratigraphicMember", relationEn:"PART_OF", relationZh:"属于", target:"Formation"},

  {source:"StratigraphicMember", relationEn:"HAS_SUBMEMBER", relationZh:"具有亚段/小层", target:"SubMember"},
  {source:"SubMember", relationEn:"PART_OF", relationZh:"属于", target:"StratigraphicMember"},

  {source:"SubMember", relationEn:"HAS_BED", relationZh:"具有层", target:"Bed"},
  {source:"Bed", relationEn:"PART_OF", relationZh:"属于", target:"SubMember"},

  {source:"Formation", relationEn:"CONTAINS", relationZh:"包含", target:"ReservoirInterval"},
  {source:"StratigraphicMember", relationEn:"CONTAINS", relationZh:"包含", target:"ReservoirInterval"},
  {source:"SubMember", relationEn:"CONTAINS", relationZh:"包含", target:"ReservoirInterval"},
  {source:"Bed", relationEn:"CONTAINS", relationZh:"包含", target:"ReservoirInterval"},

  {source:"ReservoirInterval", relationEn:"PART_OF", relationZh:"属于", target:"Formation"},
  {source:"ReservoirInterval", relationEn:"PART_OF", relationZh:"属于", target:"StratigraphicMember"},
  {source:"ReservoirInterval", relationEn:"PART_OF", relationZh:"属于", target:"SubMember"},
  {source:"ReservoirInterval", relationEn:"PART_OF", relationZh:"属于", target:"Bed"},

  {source:"StratigraphicBoundary", relationEn:"BOUNDS", relationZh:"限定边界", target:"Formation"},
  {source:"StratigraphicBoundary", relationEn:"BOUNDS", relationZh:"限定边界", target:"StratigraphicMember"},
  {source:"StratigraphicBoundary", relationEn:"BOUNDS", relationZh:"限定边界", target:"SubMember"},
  {source:"StratigraphicBoundary", relationEn:"BOUNDS", relationZh:"限定边界", target:"Bed"},

  {source:"Unconformity", relationEn:"IS_TYPE_OF", relationZh:"属于类型", target:"StratigraphicBoundary"},

  {source:"SequenceStratigraphicUnit", relationEn:"CONTAINS", relationZh:"包含", target:"SystemTract"},
  {source:"SequenceStratigraphicUnit", relationEn:"BOUNDED_BY", relationZh:"受界面限定", target:"SequenceBoundary"},
  {source:"SequenceBoundary", relationEn:"IS_TYPE_OF", relationZh:"属于类型", target:"StratigraphicBoundary"},
  {source:"SystemTract", relationEn:"PART_OF", relationZh:"属于", target:"SequenceStratigraphicUnit"},


  // =====================================================
  // 3. 地理构造单元 与 地层地质时代关系
  // =====================================================

  {source:"Basin", relationEn:"DEVELOPS", relationZh:"发育", target:"Formation"},
  {source:"Basin", relationEn:"DEVELOPS", relationZh:"发育", target:"StratigraphicMember"},
  {source:"Basin", relationEn:"DEVELOPS", relationZh:"发育", target:"ReservoirInterval"},
  {source:"StructuralUnit", relationEn:"DEVELOPS", relationZh:"发育", target:"Formation"},
  {source:"StructuralUnit", relationEn:"DEVELOPS", relationZh:"发育", target:"StratigraphicMember"},
  {source:"StructuralUnit", relationEn:"DEVELOPS", relationZh:"发育", target:"ReservoirInterval"},
  {source:"Depression", relationEn:"DEVELOPS", relationZh:"发育", target:"SourceRock"},
  {source:"Sag", relationEn:"DEVELOPS", relationZh:"发育", target:"SourceRock"},
  {source:"Uplift", relationEn:"EXPOSES_OR_ERODES", relationZh:"出露或剥蚀", target:"Formation"},
  {source:"Fault", relationEn:"CUTS", relationZh:"切割", target:"Formation"},
  {source:"Fault", relationEn:"CUTS", relationZh:"切割", target:"StratigraphicMember"},
  {source:"Fault", relationEn:"CUTS", relationZh:"切割", target:"ReservoirInterval"},
  {source:"Fault", relationEn:"CONTROLS", relationZh:"控制", target:"StratigraphicBoundary"},
  {source:"FaultZone", relationEn:"CONTROLS", relationZh:"控制", target:"Unconformity"},


  // =====================================================
  // 4. 岩石、矿物与沉积体系内部关系
  // =====================================================

  {source:"Lithology", relationEn:"COMPOSED_OF", relationZh:"由……组成", target:"Mineral"},
  {source:"Lithology", relationEn:"HAS_TEXTURE", relationZh:"具有结构/组构", target:"RockTexture"},
  {source:"Lithology", relationEn:"HAS_STRUCTURE", relationZh:"具有沉积构造", target:"SedimentaryStructure"},

  {source:"SedimentaryFacies", relationEn:"CONTAINS", relationZh:"包含", target:"Subfacies"},
  {source:"Subfacies", relationEn:"PART_OF", relationZh:"属于", target:"SedimentaryFacies"},
  {source:"Subfacies", relationEn:"CONTAINS", relationZh:"包含", target:"Microfacies"},
  {source:"Microfacies", relationEn:"PART_OF", relationZh:"属于", target:"Subfacies"},

  {source:"DepositionalSystem", relationEn:"CONTAINS", relationZh:"包含", target:"SedimentaryFacies"},
  {source:"DepositionalSystem", relationEn:"CONTAINS", relationZh:"包含", target:"Subfacies"},
  {source:"DepositionalSystem", relationEn:"CONTAINS", relationZh:"包含", target:"Microfacies"},
  {source:"SedimentaryFacies", relationEn:"PART_OF", relationZh:"属于", target:"DepositionalSystem"},

  {source:"DepositionalEnvironment", relationEn:"CONTROLS", relationZh:"控制", target:"SedimentaryFacies"},
  {source:"DepositionalEnvironment", relationEn:"CONTROLS", relationZh:"控制", target:"Lithology"},
  {source:"SedimentaryStructure", relationEn:"INDICATES", relationZh:"指示", target:"DepositionalEnvironment"},
  {source:"Lithology", relationEn:"INDICATES", relationZh:"指示", target:"SedimentaryFacies"},
  {source:"Lithology", relationEn:"INDICATES", relationZh:"指示", target:"DepositionalEnvironment"},

  {source:"Diagenesis", relationEn:"AFFECTS", relationZh:"影响", target:"Lithology"},
  {source:"Diagenesis", relationEn:"AFFECTS", relationZh:"影响", target:"Mineral"},
  {source:"Diagenesis", relationEn:"AFFECTS", relationZh:"影响", target:"RockTexture"},
  {source:"Diagenesis", relationEn:"HAS_STAGE", relationZh:"具有阶段", target:"DiageneticStage"},
  {source:"Diagenesis", relationEn:"FORMS", relationZh:"形成", target:"DiageneticFacies"},
  {source:"DiageneticFacies", relationEn:"CONTROLLED_BY", relationZh:"受……控制", target:"Diagenesis"},


  // =====================================================
  // 5. 地层 与 岩石沉积体系关系
  // =====================================================

  {source:"Formation", relationEn:"HAS_LITHOLOGY", relationZh:"具有岩性", target:"Lithology"},
  {source:"StratigraphicMember", relationEn:"HAS_LITHOLOGY", relationZh:"具有岩性", target:"Lithology"},
  {source:"SubMember", relationEn:"HAS_LITHOLOGY", relationZh:"具有岩性", target:"Lithology"},
  {source:"Bed", relationEn:"HAS_LITHOLOGY", relationZh:"具有岩性", target:"Lithology"},
  {source:"ReservoirInterval", relationEn:"HAS_LITHOLOGY", relationZh:"具有岩性", target:"Lithology"},

  {source:"Formation", relationEn:"HAS_FACIES", relationZh:"具有沉积相", target:"SedimentaryFacies"},
  {source:"StratigraphicMember", relationEn:"HAS_FACIES", relationZh:"具有沉积相", target:"SedimentaryFacies"},
  {source:"ReservoirInterval", relationEn:"HAS_FACIES", relationZh:"具有沉积相", target:"SedimentaryFacies"},

  {source:"Formation", relationEn:"DEPOSITED_IN", relationZh:"沉积于", target:"DepositionalEnvironment"},
  {source:"StratigraphicMember", relationEn:"DEPOSITED_IN", relationZh:"沉积于", target:"DepositionalEnvironment"},
  {source:"ReservoirInterval", relationEn:"DEPOSITED_IN", relationZh:"沉积于", target:"DepositionalEnvironment"},

  {source:"Formation", relationEn:"BELONGS_TO", relationZh:"属于", target:"DepositionalSystem"},
  {source:"StratigraphicMember", relationEn:"BELONGS_TO", relationZh:"属于", target:"DepositionalSystem"},
  {source:"ReservoirInterval", relationEn:"BELONGS_TO", relationZh:"属于", target:"DepositionalSystem"},

  {source:"SequenceStratigraphicUnit", relationEn:"CONTROLS", relationZh:"控制", target:"DepositionalSystem"},
  {source:"SystemTract", relationEn:"CONTROLS", relationZh:"控制", target:"SedimentaryFacies"},


  // =====================================================
  // 6. 油气成藏要素内部关系
  // =====================================================

  {source:"PetroleumSystem", relationEn:"HAS_SOURCE_ROCK", relationZh:"具有烃源岩", target:"SourceRock"},
  {source:"PetroleumSystem", relationEn:"HAS_RESERVOIR", relationZh:"具有储层", target:"Reservoir"},
  {source:"PetroleumSystem", relationEn:"HAS_SEAL", relationZh:"具有盖层", target:"Seal"},
  {source:"PetroleumSystem", relationEn:"HAS_TRAP", relationZh:"具有圈闭", target:"Trap"},
  {source:"PetroleumSystem", relationEn:"HAS_MIGRATION_PATHWAY", relationZh:"具有运移通道", target:"MigrationPathway"},
  {source:"PetroleumSystem", relationEn:"HAS_CARRIER_SYSTEM", relationZh:"具有输导体系", target:"CarrierSystem"},
  {source:"PetroleumSystem", relationEn:"HAS_PROCESS", relationZh:"具有成藏过程", target:"AccumulationProcess"},

  {source:"SourceRock", relationEn:"GENERATES", relationZh:"生成", target:"Hydrocarbon"},
  {source:"SourceRock", relationEn:"EXPELS", relationZh:"排出", target:"Hydrocarbon"},
  {source:"Reservoir", relationEn:"STORES", relationZh:"储集", target:"Hydrocarbon"},
  {source:"Seal", relationEn:"SEALS", relationZh:"封盖", target:"Reservoir"},
  {source:"Seal", relationEn:"SEALS", relationZh:"封盖", target:"OilGasReservoir"},
  {source:"Trap", relationEn:"TRAPS", relationZh:"圈闭", target:"Hydrocarbon"},

  {source:"MigrationPathway", relationEn:"CONNECTS", relationZh:"连接", target:"SourceRock"},
  {source:"MigrationPathway", relationEn:"CONNECTS", relationZh:"连接", target:"Reservoir"},
  {source:"MigrationPathway", relationEn:"CONNECTS", relationZh:"连接", target:"Trap"},
  {source:"CarrierSystem", relationEn:"CONTAINS", relationZh:"包含", target:"MigrationPathway"},

  {source:"AccumulationProcess", relationEn:"INVOLVES", relationZh:"涉及", target:"SourceRock"},
  {source:"AccumulationProcess", relationEn:"INVOLVES", relationZh:"涉及", target:"Reservoir"},
  {source:"AccumulationProcess", relationEn:"INVOLVES", relationZh:"涉及", target:"Seal"},
  {source:"AccumulationProcess", relationEn:"INVOLVES", relationZh:"涉及", target:"Trap"},
  {source:"AccumulationProcess", relationEn:"INVOLVES", relationZh:"涉及", target:"MigrationPathway"},
  {source:"AccumulationProcess", relationEn:"OCCURS_IN", relationZh:"发生于", target:"PetroleumSystem"},


  // =====================================================
  // 7. 成藏要素 与 地理构造关系
  // =====================================================

  {source:"PetroleumSystem", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"Basin"},
  {source:"PetroleumSystem", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"StructuralUnit"},
  {source:"PetroleumSystem", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"Block"},

  {source:"SourceRock", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"Depression"},
  {source:"SourceRock", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"Sag"},
  {source:"Reservoir", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"Slope"},
  {source:"Reservoir", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"Uplift"},
  {source:"Trap", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"StructuralBelt"},
  {source:"Trap", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"Uplift"},
  {source:"Trap", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"Slope"},

  {source:"Fault", relationEn:"CONTROLS", relationZh:"控制", target:"Trap"},
  {source:"Fault", relationEn:"CONTROLS", relationZh:"控制", target:"MigrationPathway"},
  {source:"Fault", relationEn:"CONTROLS", relationZh:"控制", target:"Reservoir"},
  {source:"FaultZone", relationEn:"CONTROLS", relationZh:"控制", target:"CarrierSystem"},
  {source:"Unconformity", relationEn:"ACTS_AS", relationZh:"作为", target:"MigrationPathway"},
  {source:"Unconformity", relationEn:"CONTROLS", relationZh:"控制", target:"Trap"},


  // =====================================================
  // 8. 成藏要素 与 地层岩性关系
  // =====================================================

  {source:"SourceRock", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"Formation"},
  {source:"SourceRock", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"StratigraphicMember"},
  {source:"SourceRock", relationEn:"HAS_LITHOLOGY", relationZh:"具有岩性", target:"Lithology"},

  {source:"Reservoir", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"Formation"},
  {source:"Reservoir", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"StratigraphicMember"},
  {source:"Reservoir", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"ReservoirInterval"},
  {source:"Reservoir", relationEn:"HAS_LITHOLOGY", relationZh:"具有岩性", target:"Lithology"},

  {source:"Seal", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"Formation"},
  {source:"Seal", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"StratigraphicMember"},
  {source:"Seal", relationEn:"HAS_LITHOLOGY", relationZh:"具有岩性", target:"Lithology"},

  {source:"Trap", relationEn:"RELATED_TO", relationZh:"相关于", target:"Lithology"},
  {source:"Trap", relationEn:"RELATED_TO", relationZh:"相关于", target:"StratigraphicBoundary"},
  {source:"Trap", relationEn:"RELATED_TO", relationZh:"相关于", target:"Unconformity"},

  {source:"ReservoirInterval", relationEn:"CORRESPONDS_TO", relationZh:"对应", target:"Reservoir"},
  {source:"ReservoirInterval", relationEn:"CORRESPONDS_TO", relationZh:"对应", target:"SourceRock"},
  {source:"ReservoirInterval", relationEn:"CORRESPONDS_TO", relationZh:"对应", target:"Seal"},


  // =====================================================
  // 9. 油气藏与流体关系
  // =====================================================

  {source:"OilGasField", relationEn:"CONTAINS", relationZh:"包含", target:"OilGasReservoir"},
  {source:"OilGasField", relationEn:"LOCATED_IN", relationZh:"位于", target:"Basin"},
  {source:"OilGasField", relationEn:"LOCATED_IN", relationZh:"位于", target:"StructuralUnit"},
  {source:"OilGasField", relationEn:"LOCATED_IN", relationZh:"位于", target:"Block"},

  {source:"OilGasReservoir", relationEn:"PART_OF", relationZh:"属于", target:"OilGasField"},
  {source:"OilGasReservoir", relationEn:"CONTAINS", relationZh:"包含", target:"Hydrocarbon"},
  {source:"OilGasReservoir", relationEn:"HAS_FLUID_CONTACT", relationZh:"具有流体界面", target:"FluidContact"},
  {source:"OilGasReservoir", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"Reservoir"},
  {source:"OilGasReservoir", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"Trap"},
  {source:"OilGasReservoir", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"ReservoirInterval"},

  {source:"Hydrocarbon", relationEn:"ACCUMULATES_IN", relationZh:"聚集于", target:"Reservoir"},
  {source:"Hydrocarbon", relationEn:"ACCUMULATES_IN", relationZh:"聚集于", target:"Trap"},
  {source:"Hydrocarbon", relationEn:"ACCUMULATES_IN", relationZh:"聚集于", target:"OilGasReservoir"},

  {source:"FluidContact", relationEn:"BOUNDS", relationZh:"限定边界", target:"OilGasReservoir"},
  {source:"FluidContact", relationEn:"INDICATES", relationZh:"指示", target:"Hydrocarbon"},


  // =====================================================
  // 10. 储层结构与评价实体关系
  // =====================================================

  {source:"Reservoir", relationEn:"HAS_PORE_STRUCTURE", relationZh:"具有孔隙结构", target:"PoreStructure"},
  {source:"Reservoir", relationEn:"HAS_FRACTURE", relationZh:"具有裂缝", target:"Fracture"},
  {source:"Reservoir", relationEn:"HAS_CAVITY", relationZh:"具有孔洞", target:"Cavity"},
  {source:"Reservoir", relationEn:"HAS_SWEET_SPOT", relationZh:"具有甜点", target:"SweetSpot"},

  {source:"PoreStructure", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"Lithology"},
  {source:"Fracture", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"Reservoir"},
  {source:"Cavity", relationEn:"DEVELOPED_IN", relationZh:"发育于", target:"Reservoir"},
  {source:"SweetSpot", relationEn:"LOCATED_IN", relationZh:"位于", target:"Reservoir"},
  {source:"SweetSpot", relationEn:"LOCATED_IN", relationZh:"位于", target:"ReservoirInterval"},
  {source:"SweetSpot", relationEn:"LOCATED_IN", relationZh:"位于", target:"Block"},

  {source:"Fault", relationEn:"CONTROLS", relationZh:"控制", target:"Fracture"},
  {source:"Diagenesis", relationEn:"AFFECTS", relationZh:"影响", target:"PoreStructure"},
  {source:"Diagenesis", relationEn:"FORMS", relationZh:"形成", target:"Cavity"},
  {source:"DiageneticFacies", relationEn:"AFFECTS", relationZh:"影响", target:"Reservoir"},
  {source:"Mineral", relationEn:"AFFECTS", relationZh:"影响", target:"PoreStructure"},
  {source:"Lithology", relationEn:"CONTROLS", relationZh:"控制", target:"PoreStructure"},


  // =====================================================
  // 11. 地球化学与有机质关系
  // =====================================================

  {source:"SourceRock", relationEn:"CONTAINS", relationZh:"包含", target:"OrganicMatter"},
  {source:"OrganicMatter", relationEn:"HAS_TYPE", relationZh:"具有类型", target:"KerogenType"},
  {source:"OrganicMatter", relationEn:"GENERATES", relationZh:"生成", target:"Hydrocarbon"},

  {source:"SourceRock", relationEn:"HAS_BIOMARKER", relationZh:"具有生物标志化合物", target:"Biomarker"},
  {source:"SourceRock", relationEn:"HAS_ISOTOPE", relationZh:"具有同位素", target:"Isotope"},
  {source:"Hydrocarbon", relationEn:"HAS_BIOMARKER", relationZh:"具有生物标志化合物", target:"Biomarker"},
  {source:"Hydrocarbon", relationEn:"HAS_ISOTOPE", relationZh:"具有同位素", target:"Isotope"},

  {source:"Biomarker", relationEn:"INDICATES", relationZh:"指示", target:"OrganicMatter"},
  {source:"Biomarker", relationEn:"INDICATES", relationZh:"指示", target:"SourceRock"},
  {source:"Biomarker", relationEn:"INDICATES", relationZh:"指示", target:"DepositionalEnvironment"},
  {source:"Biomarker", relationEn:"INDICATES", relationZh:"指示", target:"Hydrocarbon"},

  {source:"Isotope", relationEn:"INDICATES", relationZh:"指示", target:"SourceRock"},
  {source:"Isotope", relationEn:"INDICATES", relationZh:"指示", target:"Hydrocarbon"},
  {source:"Isotope", relationEn:"INDICATES", relationZh:"指示", target:"DepositionalEnvironment"},

  {source:"Hydrocarbon", relationEn:"CORRELATED_WITH", relationZh:"油源对比于", target:"SourceRock"},


  // =====================================================
  // 12. 勘探开发与资料类型关系
  // =====================================================

  {source:"Well", relationEn:"LOCATED_IN", relationZh:"位于", target:"Basin"},
  {source:"Well", relationEn:"LOCATED_IN", relationZh:"位于", target:"StructuralUnit"},
  {source:"Well", relationEn:"LOCATED_IN", relationZh:"位于", target:"Block"},
  {source:"Well", relationEn:"LOCATED_IN", relationZh:"位于", target:"StudyArea"},

  {source:"Well", relationEn:"ENCOUNTERS", relationZh:"钻遇", target:"Formation"},
  {source:"Well", relationEn:"ENCOUNTERS", relationZh:"钻遇", target:"StratigraphicMember"},
  {source:"Well", relationEn:"ENCOUNTERS", relationZh:"钻遇", target:"ReservoirInterval"},
  {source:"Well", relationEn:"ENCOUNTERS", relationZh:"钻遇", target:"Reservoir"},
  {source:"Well", relationEn:"ENCOUNTERS", relationZh:"钻遇", target:"FluidContact"},

  {source:"Well", relationEn:"HAS_WELL_LOG", relationZh:"具有测井资料", target:"WellLog"},
  {source:"Well", relationEn:"HAS_CORE", relationZh:"具有岩心", target:"Core"},

  {source:"WellLog", relationEn:"INTERPRETS", relationZh:"解释", target:"Lithology"},
  {source:"WellLog", relationEn:"INTERPRETS", relationZh:"解释", target:"Reservoir"},
  {source:"WellLog", relationEn:"INTERPRETS", relationZh:"解释", target:"ReservoirInterval"},
  {source:"WellLog", relationEn:"INTERPRETS", relationZh:"解释", target:"FluidContact"},
  {source:"WellLog", relationEn:"INTERPRETS", relationZh:"解释", target:"StratigraphicBoundary"},

  {source:"Core", relationEn:"SAMPLED_FROM", relationZh:"取自", target:"Well"},
  {source:"Core", relationEn:"SAMPLED_FROM", relationZh:"取自", target:"ReservoirInterval"},
  {source:"Core", relationEn:"REVEALS", relationZh:"揭示", target:"Lithology"},
  {source:"Core", relationEn:"REVEALS", relationZh:"揭示", target:"SedimentaryStructure"},
  {source:"Core", relationEn:"REVEALS", relationZh:"揭示", target:"PoreStructure"},
  {source:"Core", relationEn:"REVEALS", relationZh:"揭示", target:"Fracture"},

  {source:"ThinSection", relationEn:"PREPARED_FROM", relationZh:"制备自", target:"Core"},
  {source:"ThinSection", relationEn:"OBSERVES", relationZh:"观察", target:"Mineral"},
  {source:"ThinSection", relationEn:"OBSERVES", relationZh:"观察", target:"RockTexture"},
  {source:"ThinSection", relationEn:"OBSERVES", relationZh:"观察", target:"PoreStructure"},
  {source:"ThinSection", relationEn:"OBSERVES", relationZh:"观察", target:"Diagenesis"},
  {source:"ThinSection", relationEn:"OBSERVES", relationZh:"观察", target:"DiageneticFacies"},

  {source:"Outcrop", relationEn:"EXPOSES", relationZh:"出露", target:"Formation"},
  {source:"Outcrop", relationEn:"EXPOSES", relationZh:"出露", target:"StratigraphicMember"},
  {source:"Outcrop", relationEn:"EXPOSES", relationZh:"出露", target:"Lithology"},
  {source:"Outcrop", relationEn:"EXPOSES", relationZh:"出露", target:"Fault"},
  {source:"Outcrop", relationEn:"EXPOSES", relationZh:"出露", target:"StratigraphicBoundary"},

  {source:"SeismicProfile", relationEn:"INTERPRETS", relationZh:"解释", target:"Fault"},
  {source:"SeismicProfile", relationEn:"INTERPRETS", relationZh:"解释", target:"FaultZone"},
  {source:"SeismicProfile", relationEn:"INTERPRETS", relationZh:"解释", target:"StructuralUnit"},
  {source:"SeismicProfile", relationEn:"INTERPRETS", relationZh:"解释", target:"Trap"},
  {source:"SeismicProfile", relationEn:"INTERPRETS", relationZh:"解释", target:"StratigraphicBoundary"},
  {source:"SeismicProfile", relationEn:"INTERPRETS", relationZh:"解释", target:"Unconformity"},

  {source:"GeologicalMap", relationEn:"SHOWS", relationZh:"展示", target:"Basin"},
  {source:"GeologicalMap", relationEn:"SHOWS", relationZh:"展示", target:"StructuralUnit"},
  {source:"GeologicalMap", relationEn:"SHOWS", relationZh:"展示", target:"Fault"},
  {source:"GeologicalMap", relationEn:"SHOWS", relationZh:"展示", target:"FaultZone"},
  {source:"GeologicalMap", relationEn:"SHOWS", relationZh:"展示", target:"Formation"},
  {source:"GeologicalMap", relationEn:"SHOWS", relationZh:"展示", target:"SedimentaryFacies"},
  {source:"GeologicalMap", relationEn:"SHOWS", relationZh:"展示", target:"OilGasReservoir"},

  {source:"ProductionTest", relationEn:"TESTS", relationZh:"测试", target:"Well"},
  {source:"ProductionTest", relationEn:"TESTS", relationZh:"测试", target:"Reservoir"},
  {source:"ProductionTest", relationEn:"TESTS", relationZh:"测试", target:"ReservoirInterval"},
  {source:"ProductionTest", relationEn:"TESTS", relationZh:"测试", target:"OilGasReservoir"},
  {source:"ProductionTest", relationEn:"CONFIRMS", relationZh:"证实", target:"Hydrocarbon"},


  // =====================================================
  // 13. 实验样品与测试关系
  // =====================================================

  {source:"Sample", relationEn:"COLLECTED_FROM", relationZh:"采自", target:"Core"},
  {source:"Sample", relationEn:"COLLECTED_FROM", relationZh:"采自", target:"Outcrop"},
  {source:"Sample", relationEn:"COLLECTED_FROM", relationZh:"采自", target:"Well"},
  {source:"Sample", relationEn:"COLLECTED_FROM", relationZh:"采自", target:"Formation"},
  {source:"Sample", relationEn:"COLLECTED_FROM", relationZh:"采自", target:"StratigraphicMember"},
  {source:"Sample", relationEn:"COLLECTED_FROM", relationZh:"采自", target:"ReservoirInterval"},
  {source:"Sample", relationEn:"COLLECTED_FROM", relationZh:"采自", target:"Lithology"},
  {source:"Sample", relationEn:"COLLECTED_FROM", relationZh:"采自", target:"Hydrocarbon"},

  {source:"Experiment", relationEn:"USES", relationZh:"使用", target:"Sample"},
  {source:"Experiment", relationEn:"USES_METHOD", relationZh:"使用方法", target:"AnalyticalMethod"},

  {source:"AnalyticalMethod", relationEn:"ANALYZES", relationZh:"分析", target:"Sample"},
  {source:"AnalyticalMethod", relationEn:"ANALYZES", relationZh:"分析", target:"Lithology"},
  {source:"AnalyticalMethod", relationEn:"ANALYZES", relationZh:"分析", target:"Mineral"},
  {source:"AnalyticalMethod", relationEn:"ANALYZES", relationZh:"分析", target:"OrganicMatter"},
  {source:"AnalyticalMethod", relationEn:"ANALYZES", relationZh:"分析", target:"KerogenType"},
  {source:"AnalyticalMethod", relationEn:"ANALYZES", relationZh:"分析", target:"Biomarker"},
  {source:"AnalyticalMethod", relationEn:"ANALYZES", relationZh:"分析", target:"Isotope"},
  {source:"AnalyticalMethod", relationEn:"ANALYZES", relationZh:"分析", target:"PoreStructure"},
  {source:"AnalyticalMethod", relationEn:"ANALYZES", relationZh:"分析", target:"Fracture"},
  {source:"AnalyticalMethod", relationEn:"ANALYZES", relationZh:"分析", target:"Cavity"},

  {source:"Experiment", relationEn:"EVALUATES", relationZh:"评价", target:"SourceRock"},
  {source:"Experiment", relationEn:"EVALUATES", relationZh:"评价", target:"Reservoir"},
  {source:"Experiment", relationEn:"EVALUATES", relationZh:"评价", target:"Seal"},
  {source:"Experiment", relationEn:"EVALUATES", relationZh:"评价", target:"Hydrocarbon"}

] AS row

MATCH (s:EntityConcept {schema: row.source})
MATCH (t:EntityConcept {schema: row.target})

MERGE (s)-[r:SCHEMA_RELATION {
  key: row.source + "_" + row.relationEn + "_" + row.target
}]->(t)

SET
  r.relationEn = row.relationEn,
  r.relationZh = row.relationZh,
  r.updatedAt = datetime();






  // =====================================================
// 14. 通用表格、图片与公式概念实体
// 可直接追加到现有 Schema CQL 后执行
// =====================================================

UNWIND [

  // ----------------
  // 表格结构实体
  // ----------------
  {
    schema: "Table",
    zhName: "表格",
    category: "通用表格图片与公式",
    description: "由行、列、表头和单元格组成的结构化数据载体，用于表达样品、参数、实验结果、统计值和对比信息。",
    examples: ["样品参数表", "储层物性统计表", "地球化学参数表"]
  },
  {
    schema: "TableHeader",
    zhName: "表头",
    category: "通用表格图片与公式",
    description: "用于说明表格字段、数据维度或分组含义的标题单元，可对应单级表头或多级表头。",
    examples: ["样品编号", "孔隙度/%", "地层层位"]
  },
  {
    schema: "TableRow",
    zhName: "表格行",
    category: "通用表格图片与公式",
    description: "表格中的横向记录单元，通常表示一个样品、对象、实验条件或统计记录。",
    examples: ["样品S1所在行", "长7段数据行", "实验组记录行"]
  },
  {
    schema: "TableColumn",
    zhName: "表格列",
    category: "通用表格图片与公式",
    description: "表格中的纵向字段单元，通常表示一种属性、参数、条件、单位或分类维度。",
    examples: ["TOC列", "孔隙度列", "温度列"]
  },
  {
    schema: "TableCell",
    zhName: "表格单元格",
    category: "通用表格图片与公式",
    description: "表格行列交叉位置的最小数据单元，可存储文本、数值、范围、单位或对象名称。",
    examples: ["TOC=2.1%", "孔隙度=8.3%", "长7段"]
  },

  // ----------------
  // 图片与图表结构实体
  // ----------------
  {
    schema: "Figure",
    zhName: "图片/图件",
    category: "通用表格图片与公式",
    description: "论文或报告中的通用视觉内容载体，包括照片、示意图、地质图、剖面图、统计图和组合图。",
    examples: ["图1研究区位置图", "图5成藏模式图", "图8岩心照片"]
  },
  {
    schema: "FigurePanel",
    zhName: "子图",
    category: "通用表格图片与公式",
    description: "组合图片中具有相对独立内容的子图区域，通常使用a、b、c等编号进行区分。",
    examples: ["图3a", "图3b", "左侧子图"]
  },
  {
    schema: "FigureElement",
    zhName: "图片元素",
    category: "通用表格图片与公式",
    description: "图片内部可识别的视觉对象，如断层线、井位、曲线、岩心、矿物颗粒、箭头或区域边界。",
    examples: ["F1断层线", "苏6井井位", "红色趋势曲线"]
  },
  {
    schema: "Chart",
    zhName: "统计图/曲线图",
    category: "通用表格图片与公式",
    description: "通过坐标轴、数据系列和图例表达数值变化、相关性、分布或对比关系的图件。",
    examples: ["孔隙度—渗透率交会图", "等温吸附曲线", "柱状统计图"]
  },
  {
    schema: "Axis",
    zhName: "坐标轴",
    category: "通用表格图片与公式",
    description: "统计图中表达变量、取值范围和单位的坐标维度，包括横轴、纵轴和辅助坐标轴。",
    examples: ["横轴温度/℃", "纵轴吸附量/mmol·g⁻¹", "深度轴"]
  },
  {
    schema: "Legend",
    zhName: "图例",
    category: "通用表格图片与公式",
    description: "用于解释图片中颜色、符号、线型、填充或数据系列含义的视觉说明单元。",
    examples: ["红线代表长7段", "三角形代表泥岩样品", "蓝色区域代表湖相"]
  },
  {
    schema: "VisualAnnotation",
    zhName: "图中标注",
    category: "通用表格图片与公式",
    description: "图片中的文字、箭头、编号、框选区域或其他用于解释视觉对象的标记。",
    examples: ["异常高值区", "油气运移方向箭头", "井号标注"]
  },
  {
    schema: "DataSeries",
    zhName: "数据系列",
    category: "通用表格图片与公式",
    description: "表格或图表中由一组有序数据点组成的数据集合，可表示曲线、样品组、时间序列或参数序列。",
    examples: ["温度—吸附量序列", "不同样品TOC序列", "深度—孔隙度曲线"]
  },

  // ----------------
  // 公式结构实体
  // ----------------
  {
    schema: "Formula",
    zhName: "公式",
    category: "通用表格图片与公式",
    description: "以数学符号表达参数定义、计算方法、变量关系、模型约束或评价指标的结构化表达。",
    examples: ["孔隙度计算公式", "Langmuir方程", "等量吸附热公式"]
  },
  {
    schema: "Symbol",
    zhName: "公式符号",
    category: "通用表格图片与公式",
    description: "公式中用于表示参数、变量、常量、系数或运算对象的字符或符号。",
    examples: ["φ", "q_st", "T", "R"]
  },
  {
    schema: "Variable",
    zhName: "变量",
    category: "通用表格图片与公式",
    description: "公式中取值可以变化的符号，通常对应实验条件、测量参数、计算结果或自变量。",
    examples: ["温度T", "压力P", "吸附量V"]
  },
  {
    schema: "Constant",
    zhName: "常量",
    category: "通用表格图片与公式",
    description: "公式中取值固定或在指定条件下保持不变的符号或系数。",
    examples: ["气体常数R", "圆周率π", "经验常数C"]
  },
  {
    schema: "Parameter",
    zhName: "参数",
    category: "通用表格图片与公式",
    description: "能够被测量、计算、统计或用于描述对象性质的量，可具有数值、单位、范围和实验条件。",
    examples: ["孔隙度", "渗透率", "TOC", "温度", "压力"]
  },
  {
    schema: "Unit",
    zhName: "单位",
    category: "通用表格图片与公式",
    description: "用于表示参数或数值计量尺度的标准单位。",
    examples: ["%", "mD", "MPa", "℃", "mmol/g"]
  },
  {
    schema: "Caption",
    zhName: "标题/图题/表题",
    category: "通用表格图片与公式",
    description: "用于说明表格、图片、子图或公式编号、主题和内容范围的文本描述。",
    examples: ["表1 样品基本参数", "图5 成藏模式", "公式（3）"]
  }

] AS row

MERGE (c:EntityConcept {schema: row.schema})
SET
  c.zhName = row.zhName,
  c.category = row.category,
  c.description = row.description,
  c.examples = row.examples,
  c.updatedAt = datetime()

MERGE (cat:ConceptCategory {name: row.category})
SET
  cat.description = "描述论文中通用表格结构、图片视觉结构、统计图数据、公式符号、参数及跨模态证据关系。",
  cat.updatedAt = datetime()

MERGE (c)-[:BELONGS_TO_CATEGORY]->(cat);


// =====================================================
// 15. 通用表格、图片与公式内部关系
// =====================================================

UNWIND [

  // ----------------
  // 表格内部关系
  // ----------------
  {source:"Table", relationEn:"HAS_CAPTION", relationZh:"具有标题", target:"Caption"},
  {source:"Table", relationEn:"HAS_HEADER", relationZh:"具有表头", target:"TableHeader"},
  {source:"Table", relationEn:"HAS_ROW", relationZh:"具有表格行", target:"TableRow"},
  {source:"Table", relationEn:"HAS_COLUMN", relationZh:"具有表格列", target:"TableColumn"},
  {source:"Table", relationEn:"HAS_CELL", relationZh:"具有单元格", target:"TableCell"},
  {source:"Table", relationEn:"HAS_DATA_SERIES", relationZh:"具有数据系列", target:"DataSeries"},

  {source:"TableHeader", relationEn:"PART_OF", relationZh:"属于", target:"Table"},
  {source:"TableHeader", relationEn:"DEFINES_COLUMN", relationZh:"定义表格列", target:"TableColumn"},
  {source:"TableHeader", relationEn:"REPRESENTS", relationZh:"表示", target:"Parameter"},
  {source:"TableHeader", relationEn:"HAS_UNIT", relationZh:"具有单位", target:"Unit"},

  {source:"TableRow", relationEn:"PART_OF", relationZh:"属于", target:"Table"},
  {source:"TableRow", relationEn:"CONTAINS_CELL", relationZh:"包含单元格", target:"TableCell"},

  {source:"TableColumn", relationEn:"PART_OF", relationZh:"属于", target:"Table"},
  {source:"TableColumn", relationEn:"CONTAINS_CELL", relationZh:"包含单元格", target:"TableCell"},
  {source:"TableColumn", relationEn:"REPRESENTS", relationZh:"表示", target:"Parameter"},
  {source:"TableColumn", relationEn:"HAS_UNIT", relationZh:"具有单位", target:"Unit"},

  {source:"TableCell", relationEn:"PART_OF", relationZh:"属于", target:"Table"},
  {source:"TableCell", relationEn:"LOCATED_IN_ROW", relationZh:"位于表格行", target:"TableRow"},
  {source:"TableCell", relationEn:"LOCATED_IN_COLUMN", relationZh:"位于表格列", target:"TableColumn"},
  {source:"TableCell", relationEn:"REPRESENTS", relationZh:"表示", target:"Parameter"},
  {source:"TableCell", relationEn:"HAS_UNIT", relationZh:"具有单位", target:"Unit"},
  {source:"TableCell", relationEn:"MEMBER_OF_SERIES", relationZh:"属于数据系列", target:"DataSeries"},

  // ----------------
  // 图片内部关系
  // ----------------
  {source:"Figure", relationEn:"HAS_CAPTION", relationZh:"具有图题", target:"Caption"},
  {source:"Figure", relationEn:"HAS_PANEL", relationZh:"具有子图", target:"FigurePanel"},
  {source:"Figure", relationEn:"HAS_ELEMENT", relationZh:"具有图片元素", target:"FigureElement"},
  {source:"Figure", relationEn:"HAS_ANNOTATION", relationZh:"具有图中标注", target:"VisualAnnotation"},

  {source:"FigurePanel", relationEn:"PART_OF", relationZh:"属于", target:"Figure"},
  {source:"FigurePanel", relationEn:"HAS_CAPTION", relationZh:"具有子图标题", target:"Caption"},
  {source:"FigurePanel", relationEn:"CONTAINS", relationZh:"包含", target:"FigureElement"},
  {source:"FigurePanel", relationEn:"HAS_ANNOTATION", relationZh:"具有图中标注", target:"VisualAnnotation"},

  {source:"FigureElement", relationEn:"PART_OF", relationZh:"属于", target:"Figure"},
  {source:"FigureElement", relationEn:"LOCATED_IN", relationZh:"位于", target:"FigurePanel"},

  {source:"Chart", relationEn:"IS_TYPE_OF", relationZh:"属于类型", target:"Figure"},
  {source:"Chart", relationEn:"HAS_AXIS", relationZh:"具有坐标轴", target:"Axis"},
  {source:"Chart", relationEn:"HAS_LEGEND", relationZh:"具有图例", target:"Legend"},
  {source:"Chart", relationEn:"HAS_DATA_SERIES", relationZh:"具有数据系列", target:"DataSeries"},
  {source:"Chart", relationEn:"HAS_ANNOTATION", relationZh:"具有图中标注", target:"VisualAnnotation"},

  {source:"Axis", relationEn:"PART_OF", relationZh:"属于", target:"Chart"},
  {source:"Axis", relationEn:"REPRESENTS", relationZh:"表示", target:"Parameter"},
  {source:"Axis", relationEn:"HAS_UNIT", relationZh:"具有单位", target:"Unit"},

  {source:"Legend", relationEn:"PART_OF", relationZh:"属于", target:"Chart"},
  {source:"Legend", relationEn:"DESCRIBES", relationZh:"描述", target:"DataSeries"},
  {source:"Legend", relationEn:"DESCRIBES", relationZh:"描述", target:"FigureElement"},

  {source:"VisualAnnotation", relationEn:"ANNOTATES", relationZh:"标注", target:"Figure"},
  {source:"VisualAnnotation", relationEn:"ANNOTATES", relationZh:"标注", target:"FigurePanel"},
  {source:"VisualAnnotation", relationEn:"ANNOTATES", relationZh:"标注", target:"FigureElement"},
  {source:"VisualAnnotation", relationEn:"ANNOTATES", relationZh:"标注", target:"DataSeries"},

  {source:"DataSeries", relationEn:"REPRESENTS", relationZh:"表示", target:"Parameter"},
  {source:"DataSeries", relationEn:"SHOWN_IN", relationZh:"显示于", target:"Chart"},
  {source:"DataSeries", relationEn:"RECORDED_IN", relationZh:"记录于", target:"Table"},

  // ----------------
  // 公式内部关系
  // ----------------
  {source:"Formula", relationEn:"HAS_CAPTION", relationZh:"具有公式标题或编号", target:"Caption"},
  {source:"Formula", relationEn:"HAS_SYMBOL", relationZh:"具有符号", target:"Symbol"},
  {source:"Formula", relationEn:"HAS_VARIABLE", relationZh:"具有变量", target:"Variable"},
  {source:"Formula", relationEn:"HAS_CONSTANT", relationZh:"具有常量", target:"Constant"},
  {source:"Formula", relationEn:"DEFINES", relationZh:"定义", target:"Parameter"},
  {source:"Formula", relationEn:"COMPUTES", relationZh:"计算", target:"Parameter"},
  {source:"Formula", relationEn:"DEPENDS_ON", relationZh:"依赖", target:"Parameter"},
  {source:"Formula", relationEn:"FITS", relationZh:"拟合", target:"DataSeries"},

  {source:"Variable", relationEn:"IS_TYPE_OF", relationZh:"属于类型", target:"Symbol"},
  {source:"Constant", relationEn:"IS_TYPE_OF", relationZh:"属于类型", target:"Symbol"},

  {source:"Symbol", relationEn:"REPRESENTS", relationZh:"表示", target:"Parameter"},
  {source:"Symbol", relationEn:"HAS_UNIT", relationZh:"具有单位", target:"Unit"},
  {source:"Symbol", relationEn:"PART_OF", relationZh:"属于", target:"Formula"},

  {source:"Parameter", relationEn:"REPRESENTED_BY", relationZh:"由符号表示", target:"Symbol"},
  {source:"Parameter", relationEn:"HAS_UNIT", relationZh:"具有单位", target:"Unit"},
  {source:"Parameter", relationEn:"COMPUTED_BY", relationZh:"由公式计算", target:"Formula"},
  {source:"Parameter", relationEn:"RECORDED_IN", relationZh:"记录于", target:"Table"},
  {source:"Parameter", relationEn:"VISUALIZED_IN", relationZh:"可视化于", target:"Chart"}

] AS row

MATCH (s:EntityConcept {schema: row.source})
MATCH (t:EntityConcept {schema: row.target})

MERGE (s)-[r:SCHEMA_RELATION {
  key: row.source + "_" + row.relationEn + "_" + row.target
}]->(t)

SET
  r.relationEn = row.relationEn,
  r.relationZh = row.relationZh,
  r.updatedAt = datetime();


// =====================================================
// 16. 表格、图片、公式与原有领域实体的关系
// =====================================================

UNWIND [

  // ----------------
  // 表格与领域对象关系
  // ----------------
  {source:"Table", relationEn:"REPORTS", relationZh:"报告", target:"Sample"},
  {source:"Table", relationEn:"REPORTS", relationZh:"报告", target:"Experiment"},
  {source:"Table", relationEn:"REPORTS", relationZh:"报告", target:"AnalyticalMethod"},
  {source:"Table", relationEn:"DESCRIBES", relationZh:"描述", target:"Formation"},
  {source:"Table", relationEn:"DESCRIBES", relationZh:"描述", target:"StratigraphicMember"},
  {source:"Table", relationEn:"DESCRIBES", relationZh:"描述", target:"ReservoirInterval"},
  {source:"Table", relationEn:"DESCRIBES", relationZh:"描述", target:"Lithology"},
  {source:"Table", relationEn:"DESCRIBES", relationZh:"描述", target:"SourceRock"},
  {source:"Table", relationEn:"DESCRIBES", relationZh:"描述", target:"Reservoir"},
  {source:"Table", relationEn:"DESCRIBES", relationZh:"描述", target:"Hydrocarbon"},
  {source:"Table", relationEn:"QUANTIFIES", relationZh:"量化", target:"OrganicMatter"},
  {source:"Table", relationEn:"QUANTIFIES", relationZh:"量化", target:"PoreStructure"},
  {source:"Table", relationEn:"QUANTIFIES", relationZh:"量化", target:"OilGasReservoir"},

  {source:"Experiment", relationEn:"PRODUCES", relationZh:"产生", target:"Table"},
  {source:"AnalyticalMethod", relationEn:"PRODUCES", relationZh:"产生", target:"Table"},
  {source:"Sample", relationEn:"RECORDED_IN", relationZh:"记录于", target:"Table"},

  {source:"Parameter", relationEn:"DESCRIBES", relationZh:"描述", target:"Sample"},
  {source:"Parameter", relationEn:"DESCRIBES", relationZh:"描述", target:"Lithology"},
  {source:"Parameter", relationEn:"DESCRIBES", relationZh:"描述", target:"SourceRock"},
  {source:"Parameter", relationEn:"DESCRIBES", relationZh:"描述", target:"Reservoir"},
  {source:"Parameter", relationEn:"DESCRIBES", relationZh:"描述", target:"Seal"},
  {source:"Parameter", relationEn:"DESCRIBES", relationZh:"描述", target:"Hydrocarbon"},
  {source:"Parameter", relationEn:"DESCRIBES", relationZh:"描述", target:"PoreStructure"},

  {source:"Experiment", relationEn:"MEASURES", relationZh:"测量", target:"Parameter"},
  {source:"AnalyticalMethod", relationEn:"MEASURES", relationZh:"测量", target:"Parameter"},

  // ----------------
  // 图片与领域对象关系
  // ----------------
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"Basin"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"StructuralUnit"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"Depression"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"Sag"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"Uplift"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"Slope"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"Fault"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"FaultZone"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"Block"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"Formation"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"StratigraphicMember"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"ReservoirInterval"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"Lithology"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"Mineral"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"SedimentaryFacies"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"DepositionalSystem"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"SourceRock"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"Reservoir"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"Seal"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"Trap"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"MigrationPathway"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"OilGasReservoir"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"PoreStructure"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"Fracture"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"Cavity"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"Well"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"Core"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"ThinSection"},
  {source:"Figure", relationEn:"SHOWS", relationZh:"展示", target:"Sample"},

  {source:"SeismicProfile", relationEn:"IS_TYPE_OF", relationZh:"属于类型", target:"Figure"},
  {source:"GeologicalMap", relationEn:"IS_TYPE_OF", relationZh:"属于类型", target:"Figure"},
  {source:"Core", relationEn:"SHOWN_IN", relationZh:"显示于", target:"Figure"},
  {source:"ThinSection", relationEn:"SHOWN_IN", relationZh:"显示于", target:"Figure"},
  {source:"WellLog", relationEn:"SHOWN_IN", relationZh:"显示于", target:"Chart"},
  {source:"ProductionTest", relationEn:"VISUALIZED_IN", relationZh:"可视化于", target:"Chart"},
  {source:"Experiment", relationEn:"PRODUCES", relationZh:"产生", target:"Figure"},
  {source:"AnalyticalMethod", relationEn:"PRODUCES", relationZh:"产生", target:"Chart"},

  // ----------------
  // 公式与领域对象关系
  // ----------------
  {source:"Formula", relationEn:"USED_BY", relationZh:"被……使用", target:"Experiment"},
  {source:"Formula", relationEn:"USED_BY", relationZh:"被……使用", target:"AnalyticalMethod"},
  {source:"Formula", relationEn:"EVALUATES", relationZh:"评价", target:"SourceRock"},
  {source:"Formula", relationEn:"EVALUATES", relationZh:"评价", target:"Reservoir"},
  {source:"Formula", relationEn:"EVALUATES", relationZh:"评价", target:"Seal"},
  {source:"Formula", relationEn:"EVALUATES", relationZh:"评价", target:"Hydrocarbon"},
  {source:"Formula", relationEn:"EVALUATES", relationZh:"评价", target:"PoreStructure"},

  {source:"Experiment", relationEn:"USES_FORMULA", relationZh:"使用公式", target:"Formula"},
  {source:"AnalyticalMethod", relationEn:"USES_FORMULA", relationZh:"使用公式", target:"Formula"},

  // ----------------
  // 跨模态证据关系
  // ----------------
  {source:"Table", relationEn:"PROVIDES_EVIDENCE_FOR", relationZh:"为……提供证据", target:"Figure"},
  {source:"Table", relationEn:"PROVIDES_DATA_FOR", relationZh:"为……提供数据", target:"Chart"},
  {source:"Table", relationEn:"VALIDATES", relationZh:"验证", target:"Formula"},

  {source:"Chart", relationEn:"VISUALIZES", relationZh:"可视化", target:"Table"},
  {source:"Chart", relationEn:"VISUALIZES", relationZh:"可视化", target:"Parameter"},
  {source:"Chart", relationEn:"SUPPORTS", relationZh:"支持", target:"Formula"},

  {source:"Formula", relationEn:"EXPLAINS", relationZh:"解释", target:"DataSeries"},
  {source:"Formula", relationEn:"QUANTIFIES", relationZh:"量化", target:"Parameter"},
  {source:"Formula", relationEn:"VALIDATED_BY", relationZh:"由……验证", target:"Table"},
  {source:"Formula", relationEn:"VISUALIZED_IN", relationZh:"可视化于", target:"Chart"}

] AS row

MATCH (s:EntityConcept {schema: row.source})
MATCH (t:EntityConcept {schema: row.target})

MERGE (s)-[r:SCHEMA_RELATION {
  key: row.source + "_" + row.relationEn + "_" + row.target
}]->(t)

SET
  r.relationEn = row.relationEn,
  r.relationZh = row.relationZh,
  r.updatedAt = datetime();