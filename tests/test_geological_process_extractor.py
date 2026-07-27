"""地质过程与成因模式抽取器的全 mock 离线测试。"""
from __future__ import annotations

import json

from src.extractors.image_extractor import build_image_tasks, extract_from_images
from src.extractors.image_extractor.geological_process import GeologicalProcessExtractor
from src.extractors.image_extractor.schema_models import ImageExtractionContext


def _visual_payload() -> dict:
    """中文说明：模拟指定回流渗透白云石化模式图的 VLM 结构化输出。"""

    lithologies = [
        ("lith_limestone", "灰岩"),
        ("lith_dolostone", "白云岩"),
        ("lith_calcareous_dolostone", "灰质白云岩"),
        ("lith_argillaceous_dolostone", "泥质白云岩"),
        ("lith_gypsiferous_dolostone", "膏质白云岩"),
        ("lith_gypsum", "膏岩"),
        ("lith_salt", "盐岩"),
    ]
    units = [
        ("unit_m1", "马一段", 0, [("lith_salt", "盐岩", "主要")]),
        ("unit_m2", "马二段", 1, [("lith_dolostone", "白云岩", "主要")]),
        ("unit_m3", "马三段", 2, [("lith_salt", "盐岩", "主要")]),
        ("unit_m4", "马四段", 3, [("lith_limestone", "灰岩", "主要")]),
        ("unit_m57_10", "马五7-10", 4, [("lith_dolostone", "白云岩", "主要"), ("lith_gypsum", "膏岩", "夹层")]),
        ("unit_m56", "马五6", 5, [("lith_gypsiferous_dolostone", "膏质白云岩", "主要"), ("lith_salt", "盐岩", "局部")]),
    ]
    return {
        "diagram_summary": "浓缩高密度海水向下进入地层并发生渗透回流白云石化。",
        "coordinate_system": {
            "section_line": "A-A'",
            "horizontal_direction": "向东",
            "vertical_meaning": "地层从下到上依次变新",
            "normal_stratigraphic_order": True,
            "evidence": "右侧层段标签及指向 E 的方向箭头",
        },
        "legend_lithologies": [
            {"id": item_id, "name": name, "visual_pattern": "图例填充", "evidence": f"底部图例{name}", "confidence": 0.96}
            for item_id, name in lithologies
        ],
        "stratigraphic_units": [
            {
                "id": unit_id,
                "name": name,
                "rank": "段",
                "order_bottom_to_top": order,
                "geometry": "剖面右侧可见层段",
                "lithologies": [
                    {
                        "lithology_id": lith_id,
                        "name": lith_name,
                        "role": role,
                        "lateral_variation": "横向厚度变化",
                        "evidence": f"{name}内部填充与{lith_name}图例一致",
                        "confidence": 0.9,
                    }
                    for lith_id, lith_name, role in unit_lithologies
                ],
                "evidence": f"右侧标注{name}",
                "confidence": 0.94,
            }
            for unit_id, name, order, unit_lithologies in units
        ],
        "entities": [
            {"id": "loc_uplift", "name": "中央古隆起", "type": "structure", "attributes": {}, "evidence": "剖面左下部文字", "confidence": 0.96},
            {"id": "loc_sea", "name": "华北海", "type": "location", "attributes": {}, "evidence": "剖面下部文字", "confidence": 0.96},
            {"id": "fluid_seawater", "name": "海水", "type": "fluid", "attributes": {}, "evidence": "水体区域", "confidence": 0.9},
            {"id": "fluid_dense_brine", "name": "浓缩高密度海水", "type": "fluid", "attributes": {}, "evidence": "图中文字及红色向下箭头", "confidence": 0.98},
        ],
        "spatial_relations": [
            {"source": "loc_uplift", "type": "west_of", "target": "loc_sea", "coordinate_frame": "section", "explicit": True, "evidence": "中央古隆起位于华北海西侧", "confidence": 0.9},
            {"source": "fluid_dense_brine", "type": "flows_to", "target": "unit_m57_10", "coordinate_frame": "flow_path", "explicit": True, "evidence": "红色向下箭头", "confidence": 0.95},
        ],
        "temporal_relations": [
            {"source": "event_evaporation", "type": "before", "target": "event_reflux", "explicit": True, "basis": "先蒸发浓缩再回流", "evidence": "蒸发箭头与回流箭头", "confidence": 0.9}
        ],
        "causal_relations": [
            {"source": "event_evaporation", "type": "causes", "target": "fluid_dense_brine", "explicit": True, "evidence": "蒸发形成浓缩海水", "confidence": 0.9}
        ],
        "process_events": [
            {
                "id": "event_evaporation", "name": "蒸发浓缩", "type": "evaporation",
                "participants": ["fluid_seawater", "fluid_dense_brine"], "source": "fluid_seawater", "path": [], "target": "fluid_dense_brine",
                "time_stage": "海退蒸发阶段", "location": "浅水蒸发区", "result": "形成浓缩高密度海水",
                "evidence": "黑色蒸发箭头", "confidence": 0.94,
            },
            {
                "id": "event_reflux", "name": "渗透回流", "type": "reflux",
                "participants": ["fluid_dense_brine", "unit_m57_10"], "source": "fluid_dense_brine", "path": ["unit_m56"], "target": "unit_m57_10",
                "time_stage": "蒸发浓缩之后", "location": "马五段地层", "result": "高密度卤水向下及向西渗流",
                "evidence": "红色向下箭头和橙色渗透回流箭头", "confidence": 0.96,
            },
            {
                "id": "event_dolomitization", "name": "回流渗透白云石化", "type": "dolomitization",
                "participants": ["unit_m57_10", "lith_limestone", "lith_dolostone"], "source": "lith_limestone", "path": ["unit_m57_10"], "target": "lith_dolostone",
                "time_stage": "渗透回流期间", "location": "下伏颗粒滩灰岩", "result": "灰岩白云石化",
                "evidence": "图题和正文参考", "confidence": 0.9,
            },
        ],
        "uncertainties": ["局部层段内岩性横向比例无法精确量化"],
    }


class MockVLM:
    """返回固定视觉 JSON，并记录调用次数。"""

    def __init__(self):
        self.calls = 0

    def describe_image(self, image_path, prompt, **kwargs):
        """中文说明：验证 Prompt 和图片均已传入后返回 mock 结果。"""

        assert image_path.endswith("46e7f59b5f41d22b1828a146e4dd96f9be91ec908bf633a29582d45233a65813.jpg")
        self.calls += 1
        if "第一次识别的层段" in prompt and "corrections" in prompt:
            payload = _visual_payload()
            return json.dumps(
                {
                    "stratigraphic_units": payload["stratigraphic_units"],
                    "corrections": [],
                    "uncertainties": [],
                },
                ensure_ascii=False,
            )
        assert "order_bottom_to_top" in prompt
        return json.dumps(_visual_payload(), ensure_ascii=False)


class MockLLM:
    """模拟关系审查阶段，只增加有正文依据的事件依赖。"""

    def __init__(self):
        self.calls = 0

    def call_openai_json(self, prompt, task_name=""):
        """中文说明：返回渗透回流先于白云石化的审查关系。"""

        assert "不能创造" in prompt
        self.calls += 1
        return {
            "accepted": True,
            "added_relations": [],
            "event_dependencies": [
                {
                    "before": "event_reflux",
                    "after": "event_dolomitization",
                    "basis": "正文说明回流渗透使下伏颗粒滩灰岩白云石化",
                    "evidence": "剩余高度浓缩海水向下回流渗透，使下伏颗粒滩灰岩发生白云石化",
                    "confidence": 0.93,
                }
            ],
            "warnings": [],
        }


def _chunk() -> dict:
    """中文说明：构造用户指定的 A09 单图片 Chunk。"""

    return {
        "id": "鄂尔多斯盆地奥陶系马家沟组白云岩储层特征及成因机制_吴东旭:section:14:image:3",
        "document_id": "鄂尔多斯盆地奥陶系马家沟组白云岩储层特征及成因机制_吴东旭",
        "modality": "image",
        "image_path": [r"C:\Users\nzx\Desktop\Python\毕业论文项目\MulTreeMMKG\data\mineru_output\鄂尔多斯盆地奥陶系马家沟组白云岩储层特征及成因机制_吴东旭\images\46e7f59b5f41d22b1828a146e4dd96f9be91ec908bf633a29582d45233a65813.jpg"],
        "caption": "图9 回流渗透白云石化模式图",
        "references": ["剩余高度浓缩海水向下回流渗透，使下伏颗粒滩灰岩发生白云石化。"],
        "classification": {"primary_code": "A09", "primary_type": "沉积、成岩与孔隙演化模式图"},
    }


def test_geological_process_extractor_builds_lithology_spatial_and_temporal_graph() -> None:
    """指定模式图应得到层段岩性、空间叠置、相对时间和过程事件。"""

    task = build_image_tasks(_chunk())[0]
    vlm = MockVLM()
    llm = MockLLM()
    graph = GeologicalProcessExtractor().extract(task, ImageExtractionContext(llm_client=llm, vlm_client=vlm))

    assert vlm.calls == 2
    assert llm.calls == 1
    assert graph.metadata.extra["status"] == "completed"
    assert graph.metadata.extra["vlm_call_count"] == 2
    assert graph.metadata.extra["algorithm"] == "VLM证据识别 + LLM关系审查 + 地层规则推理"
    names = {entity.name for entity in graph.entities}
    assert {"马五6", "马五7-10", "灰岩", "白云岩", "浓缩高密度海水", "渗透回流"} <= names

    triples = {(relation.source_name, relation.type, relation.target_name) for relation in graph.relations}
    assert ("马五6", "directly_overlies", "马五7-10") in triples
    assert ("马五6", "younger_than", "马五7-10") in triples
    assert ("马五7-10", "has_lithology", "白云岩") in triples
    assert ("浓缩高密度海水", "flows_to", "马五7-10") in triples
    assert ("渗透回流", "before", "回流渗透白云石化") in triples
    assert len(graph.events) == 3
    assert graph.validate_references() == []


def test_pipeline_routes_a09_chunk_to_completed_geological_process_extractor() -> None:
    """图片统一入口应把 A09 Chunk 交给已实现的专用抽取器。"""

    graph = extract_from_images([_chunk()], MockLLM(), MockVLM(), show_progress=False)[0]
    assert graph.metadata.extra["routes"][0]["extractor_kind"] == "geological_process"
    assert graph.metadata.extra["routes"][0]["status"] == "completed"
    assert graph.metadata.extra["model_called"] is True
    assert graph.entities
    assert graph.relations


def _a08_visual_payload() -> dict:
    """中文说明：模拟顺北走滑断裂图的层系、构造分段、油气系统角色和成藏事件。"""

    legend = [
        ("lith_limestone", "石灰岩"),
        ("lith_argillaceous_limestone", "泥质石灰岩"),
        ("lith_calcareous_dolostone", "灰质白云岩"),
        ("lith_dolostone", "白云岩"),
        ("lith_argillaceous_dolostone", "泥质白云岩"),
        ("lith_evaporite", "膏盐岩"),
        ("lith_breccia", "角砾岩"),
    ]
    units = [
        ("unit_c1y", "Є1y烃源岩", 0, "lith_argillaceous_limestone", "泥质石灰岩"),
        ("unit_c2_c1", "Є2—Є1", 1, "lith_limestone", "石灰岩"),
        ("unit_c3ql", "Є3ql", 2, "lith_dolostone", "白云岩"),
        ("unit_o1p", "O1p", 3, "lith_calcareous_dolostone", "灰质白云岩"),
        ("unit_o3", "O3", 4, "lith_limestone", "石灰岩"),
        ("unit_o3s", "O3S泥质岩", 5, "lith_argillaceous_limestone", "泥质石灰岩"),
    ]
    return {
        "diagram_summary": "三种走滑断裂分段切穿寒武系—奥陶系，连接烃源岩与断溶储集体并控制油气富集。",
        "coordinate_system": {
            "section_line": "三维块体",
            "horizontal_direction": "左至右依次为走滑压隆、走滑平移、走滑拉分",
            "vertical_meaning": "从寒武系向上至奥陶系",
            "normal_stratigraphic_order": True,
            "evidence": "图内层系代码的垂向排列",
        },
        "legend_lithologies": [
            {"id": lith_id, "name": name, "visual_pattern": "图例填充", "evidence": f"底部图例{name}", "confidence": 0.96}
            for lith_id, name in legend
        ],
        "stratigraphic_units": [
            {
                "id": unit_id,
                "name": name,
                "rank": "层系",
                "order_bottom_to_top": order,
                "geometry": "横向延伸并被走滑断裂切穿",
                "lithologies": [{
                    "lithology_id": lith_id,
                    "name": lith_name,
                    "role": "主要",
                    "lateral_variation": "断裂带附近破碎并局部角砾化",
                    "lateral_zones": [{"zone": "断裂带外", "role": "主要", "evidence": "层内连续填充"}],
                    "evidence": f"{name}内填充与{lith_name}图例一致",
                    "confidence": 0.88,
                }],
                "evidence": f"图内标注{name}",
                "confidence": 0.92,
            }
            for unit_id, name, order, lith_id, lith_name in units
        ],
        "entities": [
            {"id": "fault_press", "name": "压隆段走滑断裂", "type": "fault", "attributes": {}, "evidence": "左侧红色花状断裂", "confidence": 0.96},
            {"id": "fault_translate", "name": "平移段走滑断裂", "type": "fault", "attributes": {}, "evidence": "中部红色窄断裂", "confidence": 0.96},
            {"id": "fault_pull", "name": "拉分段走滑断裂", "type": "fault", "attributes": {}, "evidence": "右侧红色负花状断裂", "confidence": 0.96},
            {"id": "reservoir_press", "name": "压隆段断溶储集体", "type": "reservoir_body", "attributes": {}, "evidence": "左侧断裂周缘裂缝角砾体", "confidence": 0.9},
            {"id": "reservoir_translate", "name": "平移段断溶储集体", "type": "reservoir_body", "attributes": {}, "evidence": "中部断裂周缘裂缝角砾体", "confidence": 0.9},
            {"id": "reservoir_pull", "name": "拉分段断溶储集体", "type": "reservoir_body", "attributes": {}, "evidence": "右侧断裂周缘裂缝角砾体", "confidence": 0.9},
            {"id": "fluid_hydrocarbon", "name": "油气", "type": "fluid", "attributes": {}, "evidence": "图题与正文", "confidence": 0.95},
        ],
        "structural_segments": [
            {"id": "segment_press", "name": "走滑压隆段", "style": "正花状压隆", "order_left_to_right": 0, "geometry": "宽大分叉断裂网", "structures": ["fault_press", "reservoir_press"], "wells": [{"id": "well_sb5", "name": "SB5", "evidence": "左侧井标", "confidence": 0.9}], "reservoir_characteristics": "宽幅裂缝—角砾岩断溶体", "evidence": "左侧走滑压隆标注", "confidence": 0.95},
            {"id": "segment_translate", "name": "走滑平移段", "style": "近平直窄带", "order_left_to_right": 1, "geometry": "窄直立断裂带", "structures": ["fault_translate", "reservoir_translate"], "wells": [{"id": "well_sb1_8", "name": "SB1-8", "evidence": "中部井标", "confidence": 0.96}], "reservoir_characteristics": "窄幅串珠状断溶体", "evidence": "中部走滑平移标注", "confidence": 0.95},
            {"id": "segment_pull", "name": "走滑拉分段", "style": "负花状拉分", "order_left_to_right": 2, "geometry": "上部分叉、深部贯通", "structures": ["fault_pull", "reservoir_pull"], "wells": [{"id": "well_sb1", "name": "SB1", "evidence": "右侧井标", "confidence": 0.96}, {"id": "well_sb1_10", "name": "SB1-10", "evidence": "右侧井标", "confidence": 0.96}], "reservoir_characteristics": "垂向连通且上部拉分扩展的断溶体", "evidence": "右侧走滑拉分标注", "confidence": 0.95},
        ],
        "petroleum_system": {
            "name": "顺北断溶体油气成藏系统",
            "source_rocks": [{"id": "unit_c1y", "evidence": "图中Є1y烃源岩标签及正文", "confidence": 0.98}],
            "reservoirs": [{"id": item, "evidence": "断裂周缘裂缝和角砾岩体", "confidence": 0.9} for item in ("reservoir_press", "reservoir_translate", "reservoir_pull")],
            "seals": [{"id": "unit_o3s", "evidence": "顶部O3S泥质岩覆盖碳酸盐岩层系", "confidence": 0.86}],
            "migration_paths": [{"id": item, "evidence": "深大断裂从烃源岩向上切穿层系", "confidence": 0.94} for item in ("fault_press", "fault_translate", "fault_pull")],
            "accumulation_zones": [{"id": item, "evidence": "断裂周缘断溶储集体", "confidence": 0.9} for item in ("reservoir_press", "reservoir_translate", "reservoir_pull")],
            "evidence": "图题、断裂几何和正文成藏模式",
        },
        "spatial_relations": [
            {"source": fault_id, "type": "cuts_through", "target": "unit_o3", "coordinate_frame": "structural", "explicit": True, "evidence": "红色断裂穿过O3层系", "confidence": 0.95}
            for fault_id in ("fault_press", "fault_translate", "fault_pull")
        ],
        "temporal_relations": [],
        "causal_relations": [
            {"source": "fault_pull", "type": "controls", "target": "reservoir_pull", "explicit": False, "evidence": "正文说明走滑断裂控储控藏", "confidence": 0.94}
        ],
        "process_events": [
            {"id": "event_generation", "name": "寒武系供烃", "type": "generation", "participants": ["unit_c1y", "fluid_hydrocarbon"], "source": "unit_c1y", "path": [], "target": "fluid_hydrocarbon", "time_stage": "寒武系多期供烃", "location": "Є1y烃源岩", "result": "生成油气", "explicit": False, "evidence": "正文明确陈述", "confidence": 0.96},
            {"id": "event_migration", "name": "油气垂向输导", "type": "migration", "participants": ["fluid_hydrocarbon", "fault_pull"], "source": "fluid_hydrocarbon", "path": ["fault_pull"], "target": "reservoir_pull", "time_stage": "成藏期", "location": "深大通源断裂", "result": "油气进入断溶储集体", "explicit": False, "evidence": "正文明确陈述油气沿断裂垂向输导", "confidence": 0.97},
            {"id": "event_accumulation", "name": "晚期成藏聚集", "type": "accumulation", "participants": ["fluid_hydrocarbon", "reservoir_pull"], "source": "fluid_hydrocarbon", "path": ["fault_pull"], "target": "reservoir_pull", "time_stage": "海西晚期以来", "location": "断溶储集体", "result": "形成超深断溶体油气藏", "explicit": False, "evidence": "正文明确陈述", "confidence": 0.96},
        ],
        "uncertainties": ["图中未给出所有层系颜色的唯一对应边界"],
    }


class A08MockVLM:
    """中文说明：模拟顺北图片的两次 VLM 调用并验证 A08 专项提示词已生效。"""

    def __init__(self) -> None:
        self.calls = 0

    def describe_image(self, image_path, prompt, **kwargs):
        """中文说明：第一次返回完整成藏结构，第二次仅返回动态岩性复核结果。"""

        assert image_path.endswith("f0ac23ffa801dccdccad61dda0ee06d1417576276e440abdd8db77b5a4919ff1.jpg")
        self.calls += 1
        payload = _a08_visual_payload()
        if "第一次识别的层段" in prompt and "corrections" in prompt:
            return json.dumps({"legend_lithologies": payload["legend_lithologies"], "stratigraphic_units": payload["stratigraphic_units"], "corrections": [], "uncertainties": []}, ensure_ascii=False)
        assert "走滑压隆" in prompt
        assert "petroleum_system" in prompt
        return json.dumps(payload, ensure_ascii=False)


class A08MockLLM:
    """中文说明：模拟关系审查，补充供烃、输导和晚期聚集的事件依赖。"""

    def call_openai_json(self, prompt, task_name=""):
        """中文说明：只返回正文直接支持的事件顺序，不产生新实体。"""

        assert "供烃→成储→运移→聚集/富集" in prompt
        return {
            "accepted": True,
            "added_relations": [
                {"source": "event_migration", "type": "migrates_along", "target": "fault_pull", "dimension": "process", "explicit": False, "basis": "正文", "evidence": "油气沿走滑断裂垂向运移输导", "confidence": 0.97}
            ],
            "event_dependencies": [
                {"before": "event_generation", "after": "event_migration", "basis": "供烃先于输导", "evidence": "正文成藏模式", "confidence": 0.92},
                {"before": "event_migration", "after": "event_accumulation", "basis": "输导后进入储集体聚集", "evidence": "正文成藏模式", "confidence": 0.94},
            ],
            "warnings": [],
        }


def _a08_chunk() -> dict:
    """中文说明：构造用户指定的顺北 A08 单图 Chunk。"""

    return {
        "id": "塔里木盆地顺北超深断溶体油藏特征与启示_漆立新:section:10:image:0",
        "document_id": "塔里木盆地顺北超深断溶体油藏特征与启示_漆立新",
        "modality": "image",
        "image_path": [r"C:\Users\nzx\Desktop\Python\毕业论文项目\MulTreeMMKG\data\mineru_output\塔里木盆地顺北超深断溶体油藏特征与启示_漆立新\images\f0ac23ffa801dccdccad61dda0ee06d1417576276e440abdd8db77b5a4919ff1.jpg"],
        "caption": "图6 顺北油田走滑断裂带不同分段样式油气成藏与富集模式",
        "references": ["下寒武统玉尔吐斯组供烃，油气沿深大通源断裂垂向输导，晚期成藏，走滑断裂控富。"],
        "classification": {"primary_code": "A08", "primary_type": "油气藏、成藏与富集模式图"},
    }


def test_a08_extractor_builds_segments_lithologies_and_petroleum_system_roles() -> None:
    """顺北模式图应同时产出逐层岩性、三类构造分段、源储盖输导角色和成藏事件链。"""

    task = build_image_tasks(_a08_chunk())[0]
    graph = GeologicalProcessExtractor().extract(
        task,
        ImageExtractionContext(llm_client=A08MockLLM(), vlm_client=A08MockVLM()),
    )

    assert graph.metadata.extra["status"] == "completed"
    assert graph.metadata.extra["structural_segment_count"] == 3
    assert graph.validate_references() == []
    names = {entity.name for entity in graph.entities}
    assert {"走滑压隆段", "走滑平移段", "走滑拉分段", "SB1-8", "SB1-10", "Є1y烃源岩", "顺北断溶体油气成藏系统"} <= names

    triples = {(relation.source_name, relation.type, relation.target_name) for relation in graph.relations}
    assert ("走滑拉分段", "contains_well", "SB1-10") in triples
    assert ("Є1y烃源岩", "acts_as_source_rock", "顺北断溶体油气成藏系统") in triples
    assert ("压隆段断溶储集体", "acts_as_reservoir", "顺北断溶体油气成藏系统") in triples
    assert ("压隆段走滑断裂", "acts_as_migration_path", "顺北断溶体油气成藏系统") in triples
    assert ("油气垂向输导", "flows_to", "压隆段断溶储集体") in triples
    assert ("油气垂向输导", "flows_to", "平移段断溶储集体") in triples
    assert ("油气垂向输导", "flows_to", "拉分段断溶储集体") in triples
    assert ("油气垂向输导", "before", "晚期成藏聚集") in triples
    assert len(graph.events) == 3
