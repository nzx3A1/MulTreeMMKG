"""使用项目 VLM 对两张定量图表示例执行真实抽取。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extractors.image_extractor.quantitative_chart import extract_quantitative_chart_chunks
from src.utils.vlm_client import VLMClient


IMAGE_NAMES = (
    "9b16185cc734f608d2165f15b6d01c439e6cd2540d49f05255e35f34d0515749.jpg",
    "2a4d07419ff7b3dba84d05c84936e2d3a5b880e004b3dfd3b1ed3a603ecae8cf.jpg",
)


def _find_image(name: str) -> str:
    matches = list((ROOT / "data" / "mineru_output").glob(f"*/images/{name}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"预期唯一匹配图片 {name}，实际找到 {len(matches)} 个")
    return str(matches[0])


def build_example_chunks() -> list[dict[str, object]]:
    """返回用户给出的两个 ImageChunk，路径按当前仓库动态解析。"""

    return [
        {
            "id": "2.2.1:image:1",
            "order": 3,
            "modality": "image",
            "image_path": [_find_image(IMAGE_NAMES[0])],
            "caption": (
                "图4 两类白云岩储层的孔渗特征对比（膏模孔：N=1730，粒（晶）间孔：N=784）\n"
                "Fig. 4 The contrast of porosity-permeability relationship between anhydrite mold pores "
                "(N=1730) and intergranular and intercrystalline pores (N=784)"
            ),
            "references": [
                "在孔隙度小于 2.5% 的范围内，白云岩粒（晶）间孔的渗透率明显大于膏模孔的渗透率。"
                "随着孔隙度增大，孤立膏模孔相互之间产生联络，导致渗透率上升，与粒（晶）间孔隙相当。"
            ],
        },
        {
            "id": "3.1:image:1",
            "order": 4,
            "modality": "image",
            "image_path": [_find_image(IMAGE_NAMES[1])],
            "caption": (
                "A: 盟6井，马五，粉晶白云岩；B: 莲70井，马五，粗粉晶白云岩；"
                "C: 莲55，马五，粗粉晶白云岩\n图7 白云岩稀土元素分布特征\n"
                "Fig. 7 Distribution characteristics of the rare earth elements in the dolomite"
            ),
            "references": [
                "均一化后的 REE 分布特征与近海平面海水相类似，具有明显的 Eu 负异常和 Ce 略微负异常。"
            ],
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "src/extractors/image_extractor/quantitative_chart/result/quantitative_chart_live_results.json",
    )
    parser.add_argument("--max-points", type=int, default=16)
    args = parser.parse_args()
    graphs = extract_quantitative_chart_chunks(
        build_example_chunks(),
        VLMClient(),
        output_path=args.output,
        max_points_per_series=args.max_points,
    )
    summary = [
        {
            "chunk_id": graph.metadata.chunk_id,
            "status": graph.metadata.extra.get("status"),
            "entity_count": len(graph.entities),
            "relation_count": len(graph.relations),
            "reference_errors": graph.validate_references(),
            "routes": graph.metadata.extra.get("routes"),
        }
        for graph in graphs
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果已写入：{args.output}")
    return 0 if all(item["status"] == "completed" for item in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
