import sys
import os
import json
import time
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# 将父目录添加到路径，以便正确导入提示
# 获取当前文件(2.py)的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录路径（即1.py所在的目录）
current_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(current_dir)
# 将父目录添加到Python的模块搜索路径中
sys.path.append(parent_dir)

from LLMPrompt.prompt import table_prompt, ENTITY_PROMPT2,table_2_json_md
from modalprocessor.BaseModel import BaseModel
from modalprocessor.embedding_utils import attach_embedding_to_node, attach_embeddings_to_nodes
from config.top_schema_config import NodeLabel, RelationType

class TableModalProcessor(BaseModel):


##############################################################
    def modal_caption_func(self, table_img_path: str, table_caption: str, table_body: str, table_footnote: str, context: str) -> Tuple[str, Dict]:
        """
        Step 2: AI 分析与描述生成
        3 构建 Prompt：将表格内容和上下文填入 table_prompt
        调用 LLM，让 AI 生成一份 详细描述 (detailed_description) 和 实体元数据 (entity_info)。
        
        Args:
           table_img_path (str): 表格图像路径（如果存在）。
           table_caption (str): 表格标题。
           table_body (str): 表格 Markdown 内容。
           table_footnote (str): 表格脚注。
           context (str): 表格周围的上下文文本。
           
        Returns:
            Tuple[str, Dict]: 详细描述文本和实体信息字典。
        """
        entity_name = table_caption if table_caption else "Unknown Table"
        
        # 3 构建 Prompt
        prompt_content = table_prompt(
            entity_name=entity_name,
            table_img_path=table_img_path,
            table_caption=table_caption,
            table_body=table_body,
            table_footnote=table_footnote,
            context=context
        )
        
        # 调用 LLM
        result_json = self.call_openai_json(prompt_content)
        
        detailed_description = result_json.get("detailed_description", "No description generated.")
        entity_info = result_json.get("entity_info", {
            "entity_name": entity_name,
            "entity_type": "table",
            "summary": ""
        })
        
        return detailed_description, entity_info

    def _create_entity_and_chunk(self, table_md: str, detailed_description: str, entity_info: Dict, title: str = "") -> Dict:
        """
        Step 4: 实体与文本块创建
        构建文本块: 将原始表格 Markdown 和 AI 生成的详细描述拼合为一个完整的 chunk。
        
        Args:
            table_md (str): 原始表格 Markdown。
            detailed_description (str): AI 生成的详细描述。
            entity_info (Dict): 实体信息。
            title (str): 表格所属的小节对应标题。
            
        Returns:
            Dict: 包含组合文本、实体信息和原始数据的字典。
        """
        # Combine Markdown and Description
        # 组合 Markdown 和 描述
        chunk_text = f"{table_md}\n\n### Detailed Analysis\n{detailed_description}"
        
        chunk_id = str(uuid.uuid4())
        chunk_name = entity_info.get("entity_name") or title or f"table_chunk_{chunk_id[:8]}"
        summary = entity_info.get("summary", "") if isinstance(entity_info, dict) else ""
        chunk_data = {
            "id": chunk_id,
            "name": chunk_name,
            "type": NodeLabel.TABLE,
            "label_zh": "表格",
            "chunk_text": chunk_text,
            "summary": summary,
            "original_md": table_md,
            "description": detailed_description,
            "source_chapter_title": title,
            "source_modality": "table",
            "evidence_source": title or chunk_name,
            "confidence": 0.82,
        }
        attach_embedding_to_node(chunk_data)
        return chunk_data


    def _fallback_table_node_name(self, properties: Dict, index: int, table_caption: str = "", title: str = "") -> str:
        """
        根据表格行/单元格属性生成兜底实体名；当 LLM 命名失败时，优先使用非空文本字段拼出可读名称。
        """
        if isinstance(properties, dict):
            values = []
            for value in properties.values():
                if isinstance(value, (dict, list)):
                    continue
                text = str(value or "").strip()
                if text:
                    values.append(text)
                if len(values) >= 3:
                    break
            if values:
                return " - ".join(values)

        prefix = str(table_caption or title or "table_item").strip()
        return f"{prefix}_{index + 1}" if prefix else f"table_item_{index + 1}"

    def _build_table_node_name_prompt(
        self,
        nodes: List[Dict],
        table_body: str,
        table_caption: str = "",
        title: str = "",
        context: str = "",
    ) -> str:
        """
        构造表格节点命名 Prompt，要求模型基于表题、章节、上下文和节点属性返回稳定且简短的实体名。
        """
        node_payload = []
        for index, node in enumerate(nodes):
            node_payload.append({
                "index": index,
                "current_name": node.get("name", ""),
                "properties": node.get("properties", {}),
                "evidence_span": node.get("evidence_span", ""),
            })

        return f"""
你是石油地质知识图谱实体命名专家。请根据表格标题、章节标题、上下文和每个表格节点的属性，为每个表格节点生成一个合适的 name。

命名要求：
1. name 必须表示该表格项描述的实体或对象，不要使用 item、Root、table_item、UUID 等临时名称。
2. 优先使用表格行/单元格中最能代表实体的字段；必要时结合表题或章节补全语义。
3. 名称应简短、稳定、可读，通常 4 到 30 个中文字符；不要加入解释、编号或多余标点。
4. 如果表格项描述的是层级分类，名称可以由关键层级组合而成，例如“碳酸盐岩台地-台内盆地”。
5. 只返回 JSON 对象，不要返回 Markdown 或额外说明。

输入信息：
章节标题：{title}
表格标题：{table_caption}
上下文：{context[:1200]}
表格内容：{table_body[:3000]}
待命名节点：
{json.dumps(node_payload, ensure_ascii=False)}

输出格式：
{{
  "names": [
    {{"index": 0, "name": "实体名称"}},
    {{"index": 1, "name": "实体名称"}}
  ]
}}
"""

    def _apply_llm_table_node_names(
        self,
        nodes: List[Dict],
        relations: List[Dict],
        table_body: str,
        table_caption: str = "",
        title: str = "",
        context: str = "",
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        调用 LLM 为表格内容节点批量生成实体名，并同步修正关系中的 source/target 名称引用。
        """
        if not nodes:
            return nodes, relations

        old_names = [str(node.get("name", "")).strip() for node in nodes]
        generated_names: Dict[int, str] = {}

        try:
            prompt_content = self._build_table_node_name_prompt(nodes, table_body, table_caption, title, context)
            result_json = self.call_openai_json(prompt_content)
            name_items = result_json.get("names", []) if isinstance(result_json, dict) else []
            if isinstance(name_items, list):
                for item in name_items:
                    if not isinstance(item, dict):
                        continue
                    try:
                        index = int(item.get("index"))
                    except (TypeError, ValueError):
                        continue
                    name = str(item.get("name", "")).strip()
                    if 0 <= index < len(nodes) and name:
                        generated_names[index] = name
        except Exception as exc:
            print(f"Table node naming failed, using fallback names: {exc}")

        used_names = set()
        rename_map = {}
        for index, node in enumerate(nodes):
            properties = node.get("properties", {})
            new_name = generated_names.get(index) or self._fallback_table_node_name(
                properties, index, table_caption, title
            )
            base_name = new_name
            suffix = 2
            while new_name in used_names:
                new_name = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(new_name)

            old_name = old_names[index]
            if old_name:
                rename_map[old_name] = new_name
            node["name"] = new_name

        for rel in relations:
            if not isinstance(rel, dict):
                continue
            source_name = str(rel.get("source", "")).strip()
            target_name = str(rel.get("target", "")).strip()
            if source_name in rename_map:
                rel["source"] = rename_map[source_name]
            if target_name in rename_map:
                rel["target"] = rename_map[target_name]

        return nodes, relations

    def _process_table_json(self, text: str, table_caption: str = "", title: str = "", context: str = "") -> Dict:
        """
             使用大语言模型处理单个表格的转为json，再进行json嵌套得到节点和关系。

             这里使用包路径导入 tableUtils，保证该处理器被多树流水线从项目根目录调用时也能正常工作。
        """
        from modalprocessor.table.tableUtils import normalize_table_properties, transform_to_graph
        #首先使用规则判断，是否为简单行表格或者列表格
        #如果是简单表格，直接使用规则转换为json
        result=self.analyze_table(text)
        if result.get("result")=="简单表格":
            nodes = []
            relations = []
            for entity in result["data"].get("entities", []):
                nodes.append({
                    "id": str(uuid.uuid4()),
                    "label": "tableNode",
                    "label_zh": "表格单元",
                    "name": "item"+str(uuid.uuid4()) ,
                    "properties": normalize_table_properties(entity),
                    "raw_content": "",
                    "type": NodeLabel.TABLE_CELL,
                    "source_modality": "table",
                    "evidence_source": "table",
                    "evidence_span": json.dumps(entity, ensure_ascii=False),
                    "confidence": 0.8,
                })
            return self._apply_llm_table_node_names(nodes, relations, text, table_caption, title, context)


        #拿到表格内容
        prompt_content = table_2_json_md(
            table_body=text
        )

        # 调用 LLM
        result_json = self.call_openai_json(prompt_content)


        print(result_json)
        nodes, relations = transform_to_graph(result_json)
        return self._apply_llm_table_node_names(nodes, relations, text, table_caption, title, context)

    def process_table(self, table_data: Dict) -> Dict:
        """
        处理单个表格的主入口点。
        
        Args:
            table_data: 包含 'content' (md), 'caption', 'context' 等信息的字典。
            
        Returns:
            Dict: 包含创建的 chunk、实体列表（包括主表格实体和提取的子实体）以及关系列表。
        """
        # 1 输入准备
        table_body = table_data.get("content", "")
        table_caption = table_data.get("caption", "")
        context = table_data.get("context", "")
        title = table_data.get("title", "")
        table_footnote = "" # Assuming footnote is not explicitly separated or is part of content/context
        table_img_path = "" # Assuming no image path in this specific data structure or processed elsewhere
        
        # 2 AI 分析与描述生成
        detailed_description, entity_info = self.modal_caption_func(
            table_img_path, table_caption, table_body, table_footnote, context
        )
        
        # 3 (Internal Step) Create Chunk
        # 4 实体与文本块创建
        chunk_data = self._create_entity_and_chunk(table_body, detailed_description, entity_info, title)
        
        # 5 知识图谱 (Knowledge Graph)
        # Create Main Entity metadata only for relation filtering.
        # 主表实体不再写入 entities，只用于过滤关系。
        main_entity_name = entity_info.get("entity_name") or table_caption or "Table Entity"
        
        # Deep Relation Extraction on Description
        # 对描述进行深度关系抽取
        # extracted_entities, extracted_relations = self._process_chunk_for_extraction(detailed_description)

        extracted_entities, extracted_relations = self._process_table_json(
            table_body,
            table_caption=table_caption,
            title=title,
            context=context,
        )
        
        for entity in extracted_entities:
            entity["label"] = "tableNode"
            entity["label_zh"] = "表格单元"
            entity["type"] = NodeLabel.TABLE_CELL
            entity["source_modality"] = "table"
            entity.setdefault("evidence_source", table_caption or title or "table")
            entity.setdefault("confidence", 0.8)
            if not entity.get("evidence_span"):
                entity["evidence_span"] = json.dumps(entity.get("properties", entity), ensure_ascii=False)

        for entity in extracted_entities:
            if not entity.get("id"):
                entity["id"] = str(uuid.uuid4())

        attach_embeddings_to_nodes(extracted_entities)

        # Add source relations from every entity to current chunk node.
        source_relations = []
        chunk_id = chunk_data.get("id")
        chunk_name = chunk_data.get("name")
        all_entities_for_source = extracted_entities
        for entity in all_entities_for_source:
            entity_name = entity.get("name")
            entity_id = entity.get("id")
            if not entity_name or not entity_id or not chunk_id or not chunk_name:
                continue
            source_relations.append({
                "source": entity_name,
                "source_id": entity_id,
                "relation": RelationType.SOURCE,
                "target": chunk_name,
                "target_id": chunk_id,
                "type": RelationType.SOURCE,
                "description": f"实体 '{entity_name}' 来源于 chunk '{chunk_name}'",
                "confidence": entity.get("confidence", 0.8),
                "evidence_source": table_caption or title or chunk_name,
            })
        
        # entities 仅保留表格中抽取的 item；过滤层级/挂载类关系。
        all_entities = extracted_entities
        all_relations = []
        for rel in extracted_relations + source_relations:
            relation_value = str(rel.get("relation", "")).lower()
            type_value = str(rel.get("type", "")).lower()
            source_value = str(rel.get("source", ""))
            target_value = str(rel.get("target", ""))
            is_hierarchical = (
                "hierarchical" in relation_value
                or "hierarchical" in type_value
                or "belong" in relation_value
                or "belong" in type_value
                or "has_table_cell" in relation_value
                or "has_table_cell" in type_value
                or relation_value == "包含"
                or type_value == "包含"
            )
            references_main_table = (
                source_value == main_entity_name
                or target_value == main_entity_name
            )
            if is_hierarchical or references_main_table:
                continue
            all_relations.append(rel)
        
        return {
            "chunk": chunk_data,
            "entities": all_entities,
            "relations": all_relations,
            #"main_entity": main_entity
        }

    def _parse_html_table_to_grid(self, table_html: str) -> List[List[str]]:
        """
        Parses HTML table to a 2D grid, handling rowspan and colspan using strict rules.
        Rowspan cells are filled down with the same value.
        """
        # Clean comments
        table_html = re.sub(r'<!--.*?-->', '', table_html, flags=re.DOTALL)
        
        # Extract rows
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.IGNORECASE | re.DOTALL)
        
        grid = {} # (r, c) -> value
        max_col = 0
        
        for r, row_content in enumerate(rows):
            # Extract cells: td or th
            # We use a pattern that captures attributes and content
            # pattern: <t[dh] (attributes) > (content) </t[dh]>
            cells = re.findall(r'<t[dh]([^>]*)>(.*?)</t[dh]>', row_content, re.IGNORECASE | re.DOTALL)
            
            current_col = 0
            for attrs, content in cells:
                # Skip occupied cells (from previous rowspans)
                while (r, current_col) in grid:
                    current_col += 1
                
                # Parse attributes
                rowspan = 1
                colspan = 1
                
                rs_match = re.search(r'rowspan=["\']?(\d+)["\']?', attrs, re.IGNORECASE)
                if rs_match:
                    rowspan = int(rs_match.group(1))
                    
                cs_match = re.search(r'colspan=["\']?(\d+)["\']?', attrs, re.IGNORECASE)
                if cs_match:
                    colspan = int(cs_match.group(1))
                
                # Clean content: remove tags, convert entities if needed (keeping simple for now)
                clean_content = re.sub(r'<[^>]+>', '', content).strip()
                # Remove extra whitespace
                clean_content = re.sub(r'\s+', ' ', clean_content)

                # Fill the grid
                # Note: We replicate the value in ALL spanned cells to make extraction easier.
                # For example, if "Region" spans 3 rows, all 3 rows will have "Region" in that column.
                for i in range(rowspan):
                    for j in range(colspan):
                        grid[(r + i, current_col + j)] = clean_content
                
                current_col += colspan
            
            if current_col > max_col:
                max_col = current_col
        
        # Build list of lists
        result = []
        for r in range(len(rows)):
            row_data = []
            for c in range(max_col):
                row_data.append(grid.get((r, c), ""))
            result.append(row_data)
            
        return result

    def analyze_table(self, table_html):
        """
        解析表格：结合大模型（识别结构）和规则代码（解析数据，支持跨行跨列）。
        Refactored: Split into two prompts (Classification -> Structure).
        """
        # --- Step 1: 判定复杂度 (Simple vs Complex) ---
        prompt_classify = f"""
        你的任务是判断 HTML 表格的复杂度。只做二分类判断。

        【判断逻辑】
        1. **复杂表格**：
           - 如果表格结构极度混乱、包含多个独立且结构不同的子区域、或者无法映射为规则的“实体-属性”关系，请标记为 "复杂表格"。
           - 注意：仅仅包含合并单元格（rowspan/colspan）**不一定**是复杂表格，只要能清晰地对应每一行数据，仍视为简单表格。
        
        2. **简单表格**：
           - **横向**：通常第一行（或前几行）为表头，后续每一行代表一个数据对象。
           - **竖向**：第一列通常为属性名，第二列为属性值，整体代表一个对象。

        【输入表格】
        {table_html}

        【输出格式】
        JSON Only:
        {{
            "type": "简单表格" | "复杂表格",
            "reason": "简要说明原因"
        }}
        """

        try:
            # 1. 调用大模型进行分类
            classify_res = self.call_openai_json(prompt_classify)
            table_type = classify_res.get("type", "复杂表格")

            if table_type == "复杂表格":
                return {"result": "复杂表格", "reason": classify_res.get("reason", "AI判定为结构复杂")}

            # --- Step 2: 简单表格结构分析 (Orientation & Headers) ---
            prompt_structure = f"""
            你的任务是分析简单表格的结构，判断是横向还是竖向，并提取完整的属性头（Headers）。

            【输入表格】
            {table_html}

            【分析任务】
            1. **判断方向 (Orientation)**：
               - "横向" (Horizontal)：属性名在表头行，每一行存数据。
               - "竖向" (Vertical)：属性名在第一列，后续列存数据。

            2. **提取表头 (Headers)**：
               - **横向**：提取表头行（第一行或合并的前几行）的所有列名。如果有层级表头，合并名称。确保列出**所有列**的属性名。
               - **竖向**：提取第一列的所有单元格文本作为属性名。确保列出**所有行**的属性头。

            3. **确定数据位置 (Indices)**：
               - **data_start_row**: 数据（或属性值）真正开始的行索引（0-based）。
                 - 横向：跳过表头行后的第一行索引。
                 - 竖向：通常为0（如果第一行就是属性-值），如果有"属性|值"这样的标题行，则是1。
               - **data_start_col**: 数据值开始的列索引（0-based）。
                 - 横向：通常为0。
                 - 竖向：通常为1（第0列是属性名）。

            【输出格式】
            JSON Only:
            {{
                "orientation": "横向" | "竖向",
                "headers": ["header1", "header2", ...],
                "data_start_row": int,
                "data_start_col": int
            }}
            """
            
            structure_res = self.call_openai_json(prompt_structure)
            orientation = structure_res.get("orientation", "横向")
            headers = structure_res.get("headers", [])
            start_row = structure_res.get("data_start_row", 0)
            start_col = structure_res.get("data_start_col", 0 if orientation == "横向" else 1)

            # --- Step 3: 结合规则抓取数据 ---
            # 使用规则代码将 HTML 解析为二维数组 grid
            grid = self._parse_html_table_to_grid(table_html)
            entities = []

            if orientation == "横向":
                # start_row 指向第一条数据行
                # start_col 通常是 0
                for i in range(start_row, len(grid)):
                    row_data = grid[i]
                    if not row_data: 
                        continue
                    
                    entity = {}
                    # 按照 headers 顺序映射
                    for idx, header in enumerate(headers):
                        # 考虑 start_col 偏移（极为罕见，横向表格数据一般从第0列开始，但为了兼容 prompt）
                        col_idx = user_col = idx + start_col
                        if user_col < len(row_data):
                            entity[header] = row_data[user_col]
                        else:
                            entity[header] = ""
                    if entity:
                        entities.append(entity)

            elif orientation == "竖向":
                # 竖向表格：
                # Headers 列表对应 Grid 的某些行（从 start_row 开始？）
                # 这种情况下，headers 应该就是 keys。
                # 值在 columns start_col, start_col+1 ...
                
                if not grid: 
                    pass
                else:
                    num_cols = len(grid[0])
                    # 遍历每一列（作为独立实体或合并实体）
                    # 从 start_col 开始是数据列
                    for c in range(start_col, num_cols):
                        entity = {}
                        # 遍历 headers。假设 headers 顺序完全对应 grid 从 start_row 开始的行
                        for r_idx, header in enumerate(headers):
                            grid_r = start_row + r_idx
                            if grid_r < len(grid):
                                val = grid[grid_r][c]
                                entity[header] = val
                        if entity:
                            entities.append(entity)

            # 包装结果
            result_json = {
                "type": orientation,
                "headers": headers,
                "entities": entities
            }
            
            return {"result": "简单表格", "data": result_json}

        except Exception as e:
            return {"result": "错误", "reason": str(e)}








