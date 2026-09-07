"""将 MapKurator Spotter 的 JSON 结果绘制到原图，并导出便于查看的 CSV。"""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    """解析输入图像、推理 JSON、可视化图片和 CSV 的路径参数。"""
    parser = argparse.ArgumentParser(description="可视化 MapKurator Spotter 推理结果")
    parser.add_argument("--image", required=True, help="原始图片路径")
    parser.add_argument("--json", required=True, help="Spotter 输出的 JSON 路径")
    parser.add_argument("--output-image", required=True, help="标注图片输出路径")
    parser.add_argument("--output-csv", required=True, help="识别结果 CSV 输出路径")
    parser.add_argument("--font", required=True, help="支持中文的字体路径")
    return parser.parse_args()


def load_rows(json_path):
    """把 Pandas columns 格式的 JSON 转换为逐条检测记录。"""
    with Path(json_path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    rows = []
    for index_text, text in data["text"].items():
        rows.append(
            {
                "index": int(index_text),
                "text": text,
                "score": float(data["score"][index_text]),
                "polygon_x": data["polygon_x"][index_text],
                "polygon_y": data["polygon_y"][index_text],
            }
        )
    return rows


def draw_predictions(image_path, rows, font_path, output_path):
    """在原图上绘制检测多边形，并用编号标注每条识别结果。"""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, 16)

    for row in rows:
        points = list(zip(row["polygon_x"], row["polygon_y"]))
        if not points:
            continue
        draw.line(points + [points[0]], fill=(255, 0, 0), width=2)
        x, y = points[0]
        # 图上只显示编号和置信度，完整 OCR 文本放在 CSV 中，避免地图过度遮挡。
        label = f'{row["index"]} ({row["score"]:.2f})'
        box = draw.textbbox((x, y), label, font=font, stroke_width=1)
        draw.rectangle(box, fill=(255, 255, 255))
        draw.text((x, y), label, fill=(200, 0, 0), font=font, stroke_width=1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_csv(rows, output_path):
    """按置信度导出文本、外接框、中心点和完整多边形像素坐标。"""
    with Path(output_path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "index",
                "text",
                "score",
                "center_x",
                "center_y",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "polygon_pixels",
            ],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["score"], reverse=True):
            points = list(zip(row["polygon_x"], row["polygon_y"]))
            x_values, y_values = zip(*points)
            writer.writerow(
                {
                    "index": row["index"],
                    "text": row["text"],
                    "score": row["score"],
                    "center_x": round(sum(x_values) / len(x_values), 2),
                    "center_y": round(sum(y_values) / len(y_values), 2),
                    "x_min": round(min(x_values), 2),
                    "y_min": round(min(y_values), 2),
                    "x_max": round(max(x_values), 2),
                    "y_max": round(max(y_values), 2),
                    "polygon_pixels": json.dumps(points, ensure_ascii=False),
                }
            )


def main():
    """执行结果读取、图片绘制和 CSV 导出。"""
    args = parse_args()
    rows = load_rows(args.json)
    draw_predictions(args.image, rows, args.font, args.output_image)
    write_csv(rows, args.output_csv)
    print(f"已生成 {len(rows)} 条检测结果")


if __name__ == "__main__":
    main()
