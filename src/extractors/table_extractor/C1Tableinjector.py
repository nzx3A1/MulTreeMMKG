# 调用TableModalProcessor将解析好的表格table_extraction_results.json文件数据，注入到对应的toc_with_summaries.json文件中
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
explicitkg_root = current_file.parent.parent.parent
if str(explicitkg_root) not in sys.path:
    sys.path.append(str(explicitkg_root))

from modalprocessor.table.TableModalProcessor import TableModalProcessor
from config import get_section
from config.top_schema_config import RelationType


MAX_WORKERS = int(get_section("APIConfig", {}).get("MAX_CONCURRENT_REQUESTS", 4))


def _inject_table_cell_relations(result: dict) -> dict:
    if not isinstance(result, dict):
        return result

    chunk = result.get("chunk")
    entities = result.get("entities")
    relations = result.get("relations")

    if not isinstance(chunk, dict):
        return result
    if not isinstance(entities, list):
        entities = []
        result["entities"] = entities
    if not isinstance(relations, list):
        relations = []
        result["relations"] = relations

    chunk_name = str(chunk.get("name", "")).strip()
    chunk_id = str(chunk.get("id", "")).strip()
    if not chunk_name:
        return result

    existing_keys = set()
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        existing_keys.add((
            str(rel.get("source", "")).strip(),
            str(rel.get("target", "")).strip(),
            str(rel.get("relation", "")).strip(),
        ))

    for entity in entities:
        if not isinstance(entity, dict):
            continue
        if str(entity.get("type", "")).strip() != "TableCell":
            continue

        source_name = str(entity.get("name", "")).strip()
        source_id = str(entity.get("id", "")).strip()
        if not source_name:
            continue

        key = (source_name, chunk_name, RelationType.HAS_TABLE_CELL)
        if key in existing_keys:
            continue

        relations.append({
            "source": source_name,
            "source_id": source_id,
            "relation": RelationType.HAS_TABLE_CELL,
            "target": chunk_name,
            "target_id": chunk_id,
            "type": RelationType.HAS_TABLE_CELL,
            "description": f"实体 '{source_name}' 来源于 chunk '{chunk_name}'",
            "confidence": entity.get("confidence", 0.8),
            "evidence_source": chunk.get("source_chapter_title") or chunk_name,
        })
        existing_keys.add(key)

    return result


def main():
    """
    Main execution function for testing the TableModalProcessor.
    Loads the TOC structure, processes tables, and saves results.
    """
    # Define paths
    # 定义路径
    base_dir = Path(__file__).resolve().parent.parent.parent
    input_file = base_dir / "output" / "toc_structure.json"
    output_file = base_dir / "output" / "modal" / "table_extraction_results.json"

    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return

    # Load TOC structure
    # 加载 TOC 结构
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize processor
    # 初始化处理器
    try:
        processor = TableModalProcessor()
        print("TableModalProcessor initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize processor: {e}")
        return

    table_tasks = []

    # helper for recursion
    # 递归辅助函数
    def process_toc_nodes(nodes):
        for node in nodes:
            # Check for tables in current node
            # 检查当前节点中的表格
            if "table" in node and isinstance(node["table"], list):
                for table_data in node["table"]:
                    # Ensure table data has minimal required fields or provide defaults
                    # 确保表格数据具有所需的最小字段或提供默认值
                    if not table_data.get("content"):
                        continue

                    task_data = dict(table_data)
                    task_data["title"] = node.get("title", "")
                    table_tasks.append(task_data)

            # Recurse into children
            # 递归处理子节点
            if "children" in node and isinstance(node["children"], list):
                process_toc_nodes(node["children"])

    # Start processing from root TOC
    # 开始从根 TOC 处理
    if "toc" in data:
        process_toc_nodes(data["toc"])

    def process_one_table(table_data: dict) -> dict:
        caption = table_data.get("caption", "Untitled")
        print(f"Processing table: {caption}")
        result = processor.process_table(table_data)
        result = _inject_table_cell_relations(result)
        print(f"Successfully processed table: {caption}")
        return result

    all_table_results = []
    print(f"Processing {len(table_tasks)} tables with {MAX_WORKERS} worker threads.")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_one_table, task): task for task in table_tasks}
        for future in as_completed(futures):
            table_data = futures[future]
            try:
                all_table_results.append(future.result())
            except Exception as e:
                print(f"Error processing table {table_data.get('caption', 'Untitled')}: {e}")

    # Save results
    # 保存结果
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_table_results, f, ensure_ascii=False, indent=2)

    print(f"Processing complete. Found and processed {len(all_table_results)} tables.")
    print(f"Results saved to: {output_file}")



if __name__ == '__main__':
    main()
