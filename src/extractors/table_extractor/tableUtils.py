import json
import re
import uuid


COMMON_UNITS = [
    "mmol/g",
    "mol/kg",
    "cm3/g",
    "cm³/g",
    "m3/t",
    "m³/t",
    "m2/g",
    "m²/g",
    "g/cm3",
    "g/cm³",
    "mg/g",
    "kg/m3",
    "kg/m³",
    "MPa",
    "kPa",
    "Pa",
    "mD",
    "md",
    "μm",
    "µm",
    "nm",
    "km",
    "m",
    "cm",
    "mm",
    "%",
    "℃",
    "°C",
    "K",
    "Ma",
]


def _parse_header_unit(field_name):
    text = str(field_name or "").strip()
    if not text:
        return "", ""

    match = re.match(r"^(.+?)[（(]\s*([^()（）]+?)\s*[)）]$", text)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    match = re.match(r"^(.+?)\s*/\s*([A-Za-z%℃°μµ][A-Za-z0-9%℃°μµ/\-·.^³²]*)$", text)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    return text, ""


def _to_float(text):
    if text is None:
        return None
    value = str(text).strip().replace(",", "")
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    try:
        return float(value)
    except ValueError:
        return None


def _extract_unit_from_value(text):
    compact = str(text or "").strip()
    for unit in sorted(COMMON_UNITS, key=len, reverse=True):
        if unit and unit in compact:
            return unit

    match = re.search(r"[-+]?\d+(?:\.\d+)?\s*([A-Za-z%℃°μµ][A-Za-z0-9%℃°μµ/\-·.^³²]*)", compact)
    if match:
        return match.group(1).strip()
    return ""


def normalize_numeric_value(value, field_name=""):
    """
    将表格单元格中的数值、范围、单位归一化，返回可直接写入 properties 的结构。
    不识别为数值时返回空字典。
    """
    if value is None:
        return {}
    if isinstance(value, (dict, list)):
        return {}

    raw_text = str(value).strip()
    if not raw_text:
        return {}

    normalized_text = raw_text.replace("－", "-").replace("–", "-").replace("—", "-")
    normalized_text = normalized_text.replace("～", "~").replace("至", "~")
    normalized_text = re.sub(r"\s+", "", normalized_text)

    clean_field, header_unit = _parse_header_unit(field_name)
    unit = _extract_unit_from_value(normalized_text) or header_unit

    comparator = ""
    comparator_match = re.match(r"^(>=|<=|>|<|≥|≤|约|大于|小于|不小于|不大于)", normalized_text)
    if comparator_match:
        comparator = comparator_match.group(1)

    range_match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*[~]\s*([-+]?\d+(?:\.\d+)?)", normalized_text)
    if range_match:
        min_value = _to_float(range_match.group(1))
        max_value = _to_float(range_match.group(2))
        if min_value is None or max_value is None:
            return {}
        return {
            "field": clean_field or str(field_name or ""),
            "raw_value": raw_text,
            "min_value": min_value,
            "max_value": max_value,
            "unit": unit,
            "is_range": True,
            "comparator": comparator,
        }

    number_match = re.search(r"[-+]?\d+(?:\.\d+)?", normalized_text)
    if not number_match:
        return {}

    numeric_value = _to_float(number_match.group(0))
    if numeric_value is None:
        return {}

    return {
        "field": clean_field or str(field_name or ""),
        "raw_value": raw_text,
        "numeric_value": numeric_value,
        "unit": unit,
        "is_range": False,
        "comparator": comparator,
    }


def normalize_table_properties(properties):
    if not isinstance(properties, dict):
        return {}

    normalized = dict(properties)
    numeric_fields = []
    for key, value in properties.items():
        item = normalize_numeric_value(value, key)
        if item:
            numeric_fields.append(item)

    if numeric_fields:
        normalized["normalized_values"] = numeric_fields
    return normalized


def transform_to_graph(data):
    nodes = []
    relations = []

    def process_item(item, label, parent_id=None, parent_name=None):
        """
        递归处理JSON元素
        :param item: 当前处理的数据对象 (dict, list, or scalar)
        :param label: 当前节点的名称/标签
        :param parent_id: 父节点的UUID
        :param parent_name: 父节点的名称
        """
        node_id = str(uuid.uuid4())
        current_node_name = str(label)
        properties = {}
        children_to_process = []

        if isinstance(item, dict):
            for k, v in item.items():
                # 如果值是基本类型，存入当前节点属性
                if not isinstance(v, (dict, list)):
                    properties[k] = v
                else:
                    # 如果值是复杂类型，准备作为子节点处理
                    children_to_process.append((k, v))

        elif isinstance(item, list):
            for i, v in enumerate(item):
                # 列表元素作为子节点
                child_label = f"{label}_{i}"
                children_to_process.append((child_label, v))

        else:
            # 基本数据类型直接作为属性
            properties["value"] = item

        # 1. 创建当前节点
        properties = normalize_table_properties(properties)
        nodes.append({
            "id": node_id,
            "name": current_node_name,
            "properties": properties,
            "confidence": 0.78,
            "evidence_source": "table",
            "source_modality": "table",
            "evidence_span": json.dumps(properties, ensure_ascii=False) if properties else current_node_name,
        })

        # 2. 创建与父节点的关系
        if parent_id:
            relations.append({
                "source": parent_name,  # 填入父节点名字
                "source_id": parent_id,
                "target": current_node_name,  # 填入当前节点名字
                "target_id": node_id,
                "relation": "包含",
                "type": "belongTable",
                "description": "None None None",
                "confidence": 0.78,
                "evidence_source": "table",
            })

        # 3. 递归处理子项，将当前的 id 和 name 作为下一层的 parent
        for child_label, child_value in children_to_process:
            process_item(child_value, child_label, node_id, current_node_name)

    # 初始入口逻辑判断
    if isinstance(data, list):
        for i, entry in enumerate(data):
            # 对于根列表，我们可以定义一个虚拟的根名称或者直接以索引命名
            process_item(entry, f"Root_{i}")
    elif isinstance(data, dict):
        for k, v in data.items():
            process_item(v, k)
    else:
        process_item(data, "Root")

    return nodes, relations


# --- 测试代码 ---

data_samples = [
    # 样本1：嵌套地质数据
    {'奥陶系': {'中统': {'平凉': {'主要岩性': '灰、深灰色泥晶灰岩'}}}},
    # 样本3：列表嵌套结构
    [{'table': '表格', 'rows': [{'row': 1, 'cells': [{'cell': 'H/m', 'value': '1831.90'}]}]}]
]

if __name__ == "__main__":
    for i, sample in enumerate(data_samples):
        nodes, relations = transform_to_graph(sample)
        print(f"\n--- 测试样本 {i + 1} ---")

        # 打印关系部分，查看 source 和 target 是否已填入
        if relations:
            print("关系示例 (带名称):")
            # 取最后一个关系（通常是嵌套最深层的）作为示例
            print(json.dumps(relations[-1], ensure_ascii=False, indent=2))
