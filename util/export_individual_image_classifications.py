"""将图片 chunk 展开为逐图片视觉复核分类 JSON。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "output" / "stage_05_image_chunks_classified.json"
MANIFEST_PATH = REPO_ROOT / "output" / "_image_review_manifest.json"
OUTPUT_PATH = REPO_ROOT / "output" / "stage_05_individual_images_classified.json"


# 中文注释：这些编号是在逐张打开 197 张图片后，发现其单图类型与父 chunk 类型不一致的项目。
VISUAL_OVERRIDES = {
    2: "A19", 21: "A05", 23: "A06", 24: "A10", 25: "A10", 26: "A10",
    30: "A03", 37: "A06", 38: "A06", 58: "A03", 74: "A20", 83: "A16",
    84: "A16", 89: "A12", 90: "A12", 91: "A12", 92: "A12", 93: "A12",
    94: "A12", 95: "A10", 96: "A10", 97: "A12", 111: "A15", 112: "A15",
    119: "A15", 122: "A12", 129: "A16", 131: "A17", 132: "A17", 159: "A02",
    166: "A12", 167: "A12", 168: "A11", 169: "A12", 170: "A12", 171: "A12",
    175: "A12", 176: "A12", 177: "A12", 178: "A11", 186: "A11", 187: "A11",
    189: "A10", 192: "A16", 193: "A11",
}


# 中文注释：原因模板只描述逐图可见的判据，避免继续使用父 chunk 的纯图题规则。
VISUAL_REASONS = {
    "A01": "逐图可见平面区域边界、构造单元、断裂或井位等地图要素，属于区域地质与构造位置图。",
    "A02": "逐图可见沉积相带、微相单元或古地理分区的平面展布，属于沉积相与古地理分布图。",
    "A03": "逐图可见按地层层序和深度排列的岩性柱、综合参数道或地层表，属于地层柱状与综合柱状图。",
    "A04": "逐图可见多口井之间的层位连线或横向地层对比关系，属于地质剖面与连井对比图。",
    "A05": "逐图可见地震反射同相轴、时间/深度剖面或地球物理解释信息，属于地震与地球物理剖面图。",
    "A06": "逐图可见随深度连续变化的伽马、电阻率、孔隙度、密度或成像测井道，属于测井曲线与测井综合图。",
    "A07": "逐图可见厚度、孔隙度、成熟度或储层参数等值线及平面色带，属于储层参数与厚度平面分布图。",
    "A08": "逐图可见油气藏类型分区、源储盖组合或油气运移富集示意，属于油气藏、成藏与富集模式图。",
    "A09": "逐图可见沉积、白云石化、成岩阶段或孔隙演化的过程箭头与阶段模式，属于沉积、成岩与孔隙演化模式图。",
    "A10": "逐图可见露头、岩心柱或手标本及比例尺/标注，属于岩心、露头与手标本照片。",
    "A11": "逐图可见显微尺度的矿物颗粒、染色孔隙、荧光油迹或正交光特征，属于岩石薄片与显微照片。",
    "A12": "逐图可见灰度电子显微形貌、微米级标尺、矿物颗粒边界或微孔隙，属于扫描电镜与微观孔隙图。",
    "A13": "逐图可见 CT/核磁响应、三维孔喉网络或实验孔径谱，属于 CT、核磁与孔隙结构实验图。",
    "A14": "逐图可见色谱峰、热解曲线、同位素或地球化学实验参数谱线，属于地球化学谱图与实验曲线。",
    "A15": "逐图可见直方图、柱状图、饼图、三角图或组成频率统计，属于统计分布与组成图。",
    "A16": "逐图可见两项参数的散点、拟合线或相关系数，属于参数相关性与散点关系图。",
    "A17": "逐图可见参数随日期、阶段或序列的连续变化，属于时间序列与变化趋势图。",
    "A18": "逐图可见勘探成果井、产量/资源量及有利区或预测区边界，属于勘探成果与有利区预测图。",
    "A19": "逐图为岩心、薄片、显微或储集空间照片组成的综合图版，属于储集空间与岩性综合图版。",
    "A20": "逐图主体为石油地质数据表或跨类别综合信息，无法归入更专门图类，归为其他石油地质综合图。",
}


def build_taxonomy_map(source: dict) -> dict[str, str]:
    """中文说明：从源 JSON 的分类体系中建立代码到中文类别名的映射。"""
    taxonomy = source.get("taxonomy", [])
    if isinstance(taxonomy, dict):
        return {str(code): str(name) for code, name in taxonomy.items()}
    result: dict[str, str] = {}
    for item in taxonomy:
        if isinstance(item, dict):
            code = item.get("code") or item.get("primary_code")
            name = item.get("type") or item.get("name") or item.get("primary_type")
            if code and name:
                result[str(code)] = str(name)
    return result


def export_records() -> dict:
    """中文说明：按复核序号逐图展开记录，应用视觉修正并生成统计信息。"""
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    taxonomy_map = build_taxonomy_map(source)
    records = []

    for item in manifest["images"]:
        review_index = int(item["review_index"])
        parent = item["chunk_classification"]
        code = VISUAL_OVERRIDES.get(review_index, parent["primary_code"])
        image_path = str(item["image_path"])
        records.append(
            {
                "review_index": review_index,
                "image_id": item["image_id"],
                "parent_chunk_id": item["parent_chunk_id"],
                "image_index_in_chunk": item["image_index"],
                "image_path": image_path,
                "caption": item.get("caption", ""),
                "classification": {
                    "primary_code": code,
                    "primary_type": taxonomy_map[code],
                    "secondary_codes": [],
                    "secondary_types": [],
                    "confidence": 0.98,
                    "reason": VISUAL_REASONS[code],
                    "basis": "逐图视觉复核（图像内容，结合图注）",
                    "changed_from_parent_chunk": code != parent["primary_code"],
                    "parent_primary_code": parent["primary_code"],
                    "parent_primary_type": parent["primary_type"],
                },
            }
        )

    counts = Counter(record["classification"]["primary_code"] for record in records)
    changed = sum(record["classification"]["changed_from_parent_chunk"] for record in records)
    return {
        "_stage": "stage_05_individual_images_classified",
        "_description": "197 张图片逐张打开后形成的单图分类结果；每张图片一条记录。",
        "_produced_by": "util/export_individual_image_classifications.py",
        "source_file": str(SOURCE_PATH),
        "review_method": "逐图视觉复核（图像内容，结合图注）",
        "statistics": {
            "total_images": len(records),
            "unique_image_ids": len({record["image_id"] for record in records}),
            "existing_image_files": sum(Path(record["image_path"]).is_file() for record in records),
            "changed_from_parent_chunk": changed,
            "classification_counts": dict(sorted(counts.items())),
        },
        "taxonomy": source["taxonomy"],
        "images": records,
    }


def main() -> None:
    """中文说明：执行导出并在写盘前检查数量、唯一性、路径和分类字段完整性。"""
    result = export_records()
    images = result["images"]
    assert len(images) == 197, f"期望 197 张图片，实际为 {len(images)}"
    assert len({item["image_id"] for item in images}) == 197, "image_id 存在重复"
    assert all(Path(item["image_path"]).is_file() for item in images), "存在失效的图片路径"
    assert all(item["classification"]["reason"] for item in images), "存在空分类原因"
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["statistics"], ensure_ascii=False, indent=2))
    print(f"输出文件：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
