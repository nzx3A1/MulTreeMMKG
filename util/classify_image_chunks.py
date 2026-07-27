"""对石油地质图片 Chunk 进行统一分类并输出带分类字段的新 JSON。"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "output" / "stage_05_image_chunks.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "stage_05_image_chunks_classified.json"


# 中文说明：分类体系覆盖石油地质论文中常见的地图、剖面、实验图、照片和统计图。
TAXONOMY = [
    {"code": "A01", "name": "区域地质与构造位置图"},
    {"code": "A02", "name": "沉积相与古地理分布图"},
    {"code": "A03", "name": "地层柱状与综合柱状图"},
    {"code": "A04", "name": "地质剖面与连井对比图"},
    {"code": "A05", "name": "地震与地球物理剖面图"},
    {"code": "A06", "name": "测井曲线与测井综合图"},
    {"code": "A07", "name": "储层参数与厚度平面分布图"},
    {"code": "A08", "name": "油气藏、成藏与富集模式图"},
    {"code": "A09", "name": "沉积、成岩与孔隙演化模式图"},
    {"code": "A10", "name": "岩心、露头与手标本照片"},
    {"code": "A11", "name": "岩石薄片与显微照片"},
    {"code": "A12", "name": "扫描电镜与微观孔隙图"},
    {"code": "A13", "name": "CT、核磁与孔隙结构实验图"},
    {"code": "A14", "name": "地球化学谱图与实验曲线"},
    {"code": "A15", "name": "统计分布与组成图"},
    {"code": "A16", "name": "参数相关性与散点关系图"},
    {"code": "A17", "name": "时间序列与变化趋势图"},
    {"code": "A18", "name": "勘探成果与有利区预测图"},
    {"code": "A19", "name": "储集空间与岩性综合图版"},
    {"code": "A20", "name": "其他石油地质综合图"},
]
TYPE_BY_CODE = {item["code"]: item["name"] for item in TAXONOMY}


def _manual_visual_review() -> dict[str, dict[str, Any]]:
    """返回无图注 Chunk 的人工视觉复核结果。"""

    return {
        "吉木萨尔凹陷芦草沟组页岩油储层发育特征与成藏条件研究_李杰:section:1:image:2":
            _result("A03", [], 0.98, "图中包含深度、地层层级、岩性柱、甜点段与油层列。", "人工视觉复核"),
        "川南地区龙马溪组页岩气富集高产主控因素分析_蒲泊伶:section:4:image:2":
            _result("A06", [], 0.99, "图中为深度、岩性、伽马、中子孔隙度和密度测井道。", "人工视觉复核"),
        "川南地区龙马溪组页岩气富集高产主控因素分析_蒲泊伶:section:4:image:3":
            _result("A06", [], 0.99, "图中为深度、岩性、伽马、中子孔隙度和密度测井道。", "人工视觉复核"),
        "川南地区龙马溪组页岩气富集高产主控因素分析_蒲泊伶:section:4:image:4":
            _result("A06", [], 0.99, "两张子图均为含深度与多条物性曲线的测井综合图。", "人工视觉复核"),
        "川南地区龙马溪组页岩气富集高产主控因素分析_蒲泊伶:section:4:image:0":
            _result("A07", [], 0.99, "图中为带厚度等值线和城市位置的页岩厚度平面分布图。", "人工视觉复核"),
        "川南地区龙马溪组页岩气富集高产主控因素分析_蒲泊伶:section:6:image:0":
            _result("A12", ["A11"], 0.95, "灰度微观照片显示矿物基质、孔隙和有机质形态。", "人工视觉复核"),
        "川南地区龙马溪组页岩气富集高产主控因素分析_蒲泊伶:section:6:image:1":
            _result("A12", ["A11"], 0.95, "灰度微观照片显示有机质孔和孔隙结构。", "人工视觉复核"),
        "川南地区龙马溪组页岩气富集高产主控因素分析_蒲泊伶:section:6:image:2":
            _result("A12", [], 0.98, "扫描电镜图显示片状矿物及微米级孔隙。", "人工视觉复核"),
        "川南地区龙马溪组页岩气富集高产主控因素分析_蒲泊伶:section:6:image:3":
            _result("A12", ["A11"], 0.95, "灰度微观图显示矿物颗粒与粒间孔。", "人工视觉复核"),
        "川南地区龙马溪组页岩气富集高产主控因素分析_蒲泊伶:section:6:image:4":
            _result("A12", [], 0.98, "扫描电镜图显示黏土矿物集合体和微孔。", "人工视觉复核"),
        "川南地区龙马溪组页岩气富集高产主控因素分析_蒲泊伶:section:6:image:5":
            _result("A12", [], 0.98, "扫描电镜图显示片状矿物及粒间孔隙。", "人工视觉复核"),
        "川南地区龙马溪组页岩气富集高产主控因素分析_蒲泊伶:section:6:image:6":
            _result("A10", [], 0.99, "图中为带层理和裂缝特征的圆柱岩心照片。", "人工视觉复核"),
        "川南地区龙马溪组页岩气富集高产主控因素分析_蒲泊伶:section:6:image:7":
            _result("A10", [], 0.99, "图中为发育水平层理的黑色页岩岩心照片。", "人工视觉复核"),
        "自贡地区五峰组—龙马溪组页岩储层特征和含气性_朱龙飞:section:8:image:5":
            _result("A16", [], 0.99, "两张子图均为含回归线和决定系数的参数散点关系图。", "人工视觉复核"),
        "自贡地区五峰组—龙马溪组页岩储层特征和含气性_朱龙飞:section:8:image:6":
            _result("A16", [], 0.99, "图中展示总含气量与 TOC 的回归关系。", "人工视觉复核"),
        "自贡地区五峰组—龙马溪组页岩储层特征和含气性_朱龙飞:section:8:image:8":
            _result("A16", [], 0.99, "图中展示总含气量与地层压力的线性关系。", "人工视觉复核"),
        "鄂尔多斯盆地奥陶系马家沟组白云岩储层特征及成因机制_吴东旭:section:11:image:0":
            _result("A19", ["A11", "A10"], 0.98, "复合图版包含铸模孔薄片、晶间孔和岩心孔洞照片。", "人工视觉复核"),
        "鄂尔多斯盆地奥陶系马家沟组白云岩储层特征及成因机制_吴东旭:section:15:image:1":
            _result("A02", [], 0.96, "图中为含井位与微相色区的沉积相平面分布图。", "人工视觉复核"),
        "鄂尔多斯盆地延安地区马五段上组合细粒白云岩储层特征及有利储层控制因素_伊硕:section:6:image:0":
            _result("A15", ["A16"], 0.98, "三张子图包含孔隙度和渗透率频率柱状图及孔渗散点关系图。", "人工视觉复核"),
    }


def _result(
    primary_code: str,
    secondary_codes: list[str],
    confidence: float,
    reason: str,
    basis: str,
) -> dict[str, Any]:
    """按统一字段组装单个 Chunk 的分类结果。"""

    return {
        "primary_code": primary_code,
        "primary_type": TYPE_BY_CODE[primary_code],
        "secondary_codes": secondary_codes,
        "secondary_types": [TYPE_BY_CODE[code] for code in secondary_codes],
        "confidence": confidence,
        "reason": reason,
        "basis": basis,
    }


def _contains(text: str, pattern: str) -> bool:
    """以忽略大小写的正则表达式匹配图注和引用上下文。"""

    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def classify_from_caption(chunk: dict[str, Any]) -> dict[str, Any]:
    """根据图注和引用文本的石油地质术语判定图片类型。"""

    caption = str(chunk.get("caption") or "").strip()
    # 中文说明：分类优先依据图注，避免正文引用中的孔隙、成岩等术语覆盖真实图件类型。
    text = caption

    # 中文说明：组合图版、特殊仪器图和模式图优先匹配，避免被通用的“分布/关系”词覆盖。
    rules = [
        ("A18", [], r"勘探成果|有利区|甜点区预测|exploration outcome|favorable area", "图注指向勘探成果或有利区预测。"),
        ("A05", ["A04"], r"地震|seismic|AFE属性|单频剖面|S变换", "图注包含地震或地球物理剖面术语。"),
        ("A07", [], r"厚度分布|厚度等值线|总厚度.*优质页岩厚度|孔隙度平面|储层分布图|参数分布|含量分布|等值线|isopleth|content distribution|thickness.*(distribution|profile)|属性平面", "图注描述厚度、储层或参数的平面分布。"),
        ("A13", [], r"3D.?CT|CT扫描|核磁|NMR|T\s*_?\s*2谱|孔喉.*模型|node.?link|孔径分布", "图注包含 CT、核磁或孔隙结构实验术语。"),
        ("A14", [], r"色谱|chromatograph|pyrolysis|热解|同位素|isotopic|稀土元素|谱图|地球化学|饱和烃|芳香烃|主峰碳|热模拟|hydrocarbon.*yield", "图注包含地球化学谱图或实验分析术语。"),
        ("A03", [], r"综合柱状图|地层柱状图|stratigraphic column|comprehensive column|柱状剖面", "图注明确为地层或综合柱状图。"),
        ("A16", [], r"关系图|相关性|correlation|relationship|与.*关系|对.*影响|散点|回归|R\s*\^?2", "图注描述参数相关性、回归或散点关系。"),
        ("A15", [], r"直方图|柱状图|条形图|频率分布|饼状图|三角图|pie chart|histogram|bar chart|frequency distribution", "图注明确为统计分布或组成图。"),
        ("A19", ["A11", "A10"], r"储集空间类型|储层特征及主要储集空间|储集空间特征|典型岩石学特征|成岩作用特征|岩性.*岩心.*薄片|剖面.*岩心.*薄片|孔隙和裂缝中.*发育特征", "图注描述储集空间、岩心和显微照片组成的综合图版。"),
        ("A09", [], r"白云石化模式|成岩.*模式|孔隙演化|成因模式|沉积模式|顶.*底板模式|发育过程|dolomitization|diagenesis|pore evolution", "图注描述沉积、成岩或孔隙演化过程。"),
        ("A12", ["A11"], r"扫描电镜|\bSEM\b|场发射|有机质孔|晶间孔|微观孔隙", "图注包含扫描电镜或微观孔隙表征术语。"),
        ("A11", [], r"薄片|显微|microphoto|lamella|铸体|正交光|单偏光|阴极发光", "图注包含岩石薄片或显微观察术语。"),
        ("A10", [], r"岩心|露头|手标本|core|outcrop|岩芯", "图注包含岩心、露头或手标本照片术语。"),
        ("A08", [], r"成藏.*模式|富集模式|富集区分布模式|油气藏类型|油气成藏|accumulation|enrichment pattern|reservoir types", "图注描述油气藏、成藏或富集模式。"),
        ("A03", [], r"综合柱状图|地层柱状图|stratigraphic column|comprehensive column|柱状剖面", "图注明确为地层或综合柱状图。"),
        ("A06", [], r"测井|logging|伽马|GAPI|中子孔隙度|密度曲线", "图注包含测井曲线或综合测井术语。"),
        ("A04", [], r"连井|剖面图|地质剖面|profile|cross.?section|井.*对比", "图注明确为地质剖面或连井对比图。"),
        ("A16", [], r"关系图|相关性|correlation|relationship|与.*关系|散点|回归|R\s*\^?2", "图注描述参数相关性、回归或散点关系。"),
        ("A17", ["A14"], r"随.*变化|变化图|variation|change.*times|解吸日期|演化曲线", "图注描述随时间、次数或过程变化的趋势。"),
        ("A15", [], r"直方图|柱状图|条形图|频率分布|饼状图|三角图|组成统计|矿物含量.*统计|占比|含量图|含量差异|脆性指数|brittleness ind|histogram|bar chart|pie chart|frequency distribution|composition", "图注明确为统计分布、组成或指标对比图。"),
        ("A07", [], r"厚度分布|厚度等值线|总厚度.*优质页岩厚度|孔隙度平面|储层分布图|参数分布|含量分布|等值线|isopleth|content distribution|thickness.*distribution|属性平面", "图注描述厚度、储层或参数的平面分布。"),
        ("A02", [], r"沉积相|沉积微相|古地理|sedimentary facies|facies distribution", "图注描述沉积相、微相或古地理分布。"),
        ("A01", ["A03"], r"位置图.*地层.*油气层", "复合图包含钻井位置以及地层和油气层分布。"),
        ("A01", [], r"构造区划|构造位置|研究区位置|井位分布|断裂分布|断裂带空间结构|构造样式|地质图|古构造图|古地貌图|tectonic zoning|structural zoning|location.*map|geological map|fault.*distribution|strike slip fault", "图注描述区域位置、构造、断裂、古构造或古地貌。"),
    ]
    for code, secondary, pattern, reason in rules:
        if _contains(text, pattern):
            return _result(code, secondary, 0.93 if caption else 0.78, reason, "图注与上下文语义规则")

    return _result("A20", [], 0.60, "图注信息不足或同时包含多类图件特征。", "图注与上下文语义规则")


def classify_chunks(data: dict[str, Any]) -> dict[str, Any]:
    """复制原阶段五结构，为每个图片 Chunk 增加分类结果和汇总统计。"""

    manual = _manual_visual_review()
    classified_chunks = []
    category_counter: Counter[str] = Counter()
    basis_counter: Counter[str] = Counter()
    for raw_chunk in data.get("chunks", []):
        chunk = dict(raw_chunk)
        classification = manual.get(str(chunk.get("id"))) or classify_from_caption(chunk)
        chunk["classification"] = classification
        classified_chunks.append(chunk)
        category_counter[classification["primary_code"]] += 1
        basis_counter[classification["basis"]] += 1

    result = dict(data)
    result["_description"] = "118 个石油地质图片 ImageChunk 及其统一图件分类结果。"
    result["_produced_by"] = "util/classify_image_chunks.py"
    result["taxonomy"] = TAXONOMY
    result["chunks"] = classified_chunks
    result["classification_statistics"] = {
        "classified_chunk_count": len(classified_chunks),
        "category_counts": {
            f"{code} {TYPE_BY_CODE[code]}": category_counter.get(code, 0)
            for code in TYPE_BY_CODE
            if category_counter.get(code, 0) > 0
        },
        "basis_counts": dict(basis_counter),
    }
    return result


def main() -> None:
    """读取图片 Chunk JSON，执行分类并写出新的 UTF-8 JSON 文件。"""

    parser = argparse.ArgumentParser(description="分类石油地质图片 Chunk")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="待分类的图片 Chunk JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="分类结果输出 JSON")
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    result = classify_chunks(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"图片 Chunk 分类完成：{result['classification_statistics']['classified_chunk_count']} 个，"
        f"输出：{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
