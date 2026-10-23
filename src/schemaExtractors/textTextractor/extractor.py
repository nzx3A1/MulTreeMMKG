"""调用统一 LLMClient，依次完成单个文本 Chunk 的实体、关系和独立属性补充。"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from src.utils.llm_client import LLMClient

from ..schema_loader import MarkdownSchema, load_schema
from src.utils.llm_client import _extract_json_fragment
from .attribute_extraction import merge_attribute_updates, validate_attribute_updates
from .evidence import evidence_sentences
from .prompts import PROMPT_VERSION, attribute_messages, entity_messages, relation_messages
from .validator import VALIDATOR_VERSION, ValidationResult, validate_entities, validate_relations


ATTRIBUTE_BATCH_SIZE = 2


class SchemaTextExtractor:
    """组合 Schema、中文提示词和验证反馈，支持真实客户端及离线测试注入。"""

    def __init__(
        self, schema: MarkdownSchema | None = None, llm_client: Any | None = None,
        *, validation_retries: int = 1, max_tokens: int = 8192,
    ) -> None:
        """保存抽取依赖和有限纠错次数，不在初始化时发起模型请求。"""
        if validation_retries < 0 or max_tokens < 1:
            raise ValueError("validation_retries 必须非负，max_tokens 必须为正数")
        self.schema = schema if schema is not None else load_schema()
        self.client = llm_client if llm_client is not None else LLMClient()
        self.validation_retries = validation_retries
        self.max_tokens = max_tokens

    def fingerprint(self, chunk: dict[str, Any]) -> str:
        """将输入、Schema、模型和规则版本绑定，避免恢复已经过期的检查点。"""
        config = getattr(self.client, "config", None)
        payload = {
            "chunk": chunk, "schema": self.schema.fingerprint,
            "prompt": PROMPT_VERSION, "validator": VALIDATOR_VERSION,
            "client": f"{type(self.client).__module__}.{type(self.client).__qualname__}",
            "model": getattr(config, "model", "injected"),
            "endpoint": getattr(config, "base_url", ""),
            "enable_thinking": getattr(config, "enable_thinking", None),
            "validation_retries": self.validation_retries, "max_tokens": self.max_tokens,
            "attribute_batch_size": ATTRIBUTE_BATCH_SIZE,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _request(
        self, messages: list[dict[str, str]], key: str,
        validator: Callable[[Any], ValidationResult], history: list[dict[str, Any]],
    ) -> ValidationResult:
        """检查响应结构与语义约束，将验证错误反馈给模型进行有限次完整重抽。"""
        conversation = list(messages)
        for attempt in range(self.validation_retries + 1):
            raw_response = None
            chat_json_with_raw = getattr(self.client, "chat_json_with_raw", None)
            if callable(chat_json_with_raw):
                payload, raw_response = chat_json_with_raw(
                    conversation, temperature=0, max_tokens=self.max_tokens,
                )
            else:
                payload = self.client.chat_json(conversation, temperature=0, max_tokens=self.max_tokens)
            result = validator(payload)
            attempt_record = {"phase": key, "attempt": attempt + 1, "accepted_count": len(result.records),
                              "rejected_count": result.rejected_count, "issues": result.issues}
            if raw_response is not None and any(issue["code"] == "invalid_response" for issue in result.issues):
                # 中文说明：仅在响应结构错误时保存短预览，帮助区分空输出、解析失败和错误 JSON 结构。
                attempt_record["response_diagnostic"] = {
                    "raw_length": len(raw_response),
                    "raw_preview": raw_response[:4000],
                    "parsed_type": type(payload).__name__,
                    "parsed_keys": list(payload)[:30] if isinstance(payload, dict) else None,
                }
                try:
                    json.loads(_extract_json_fragment(raw_response))
                except json.JSONDecodeError as exc:
                    attempt_record["response_diagnostic"]["json_parse_error"] = {
                        "message": exc.msg, "line": exc.lineno, "column": exc.colno,
                    }
            history.append(attempt_record)
            if not result.has_errors or attempt == self.validation_retries:
                return result
            feedback = "上一轮结果未通过校验。请依据原文和原有 Schema 修复以下问题，并重新返回包含所有有效记录的完整 JSON，不要仅返回补丁：\n" + json.dumps(result.issues, ensure_ascii=False)
            if key == "attributes" and any(issue["code"] == "invalid_response" for issue in result.issues):
                # 中文说明：明确唯一顶层字段和合法空结果，避免拼接多个 JSON 对象或重复生成根节点。
                feedback += (
                    '\n属性阶段必须只输出一个可解析 JSON 对象，根节点严格为 '
                    '{"entity_attributes":[...]}；不得重复输出多个 entity_attributes 根字段或附加解释。'
                    '若本批实体确无可补充属性，唯一合法空结果为 {"entity_attributes":[]}。'
                )
            if key != "attributes" and any(issue["code"] == "invalid_provenance" for issue in result.issues):
                # 中文说明：证据复制失败时改用编号引用；成功结果的格式和恢复规则保持兼容。
                source_text = json.loads(messages[1]["content"])["source_text"]
                feedback += (
                    "\n为避免改写含公式或特殊符号的证据，当前原文已按句子编号。每条记录可省略 provenance，"
                    "改为返回 evidence_sentence_ids 字符串数组，例如 [\"sentence_1\"]；"
                    "多句只能引用连续升序编号，程序会从原文回填 provenance。不能引用不支持该记录的句子。\n"
                    + json.dumps(evidence_sentences(source_text), ensure_ascii=False)
                )
            conversation.extend([
                {"role": "assistant", "content": raw_response if isinstance(raw_response, str)
                 else json.dumps(payload, ensure_ascii=False)},
                {"role": "user", "content": feedback},
            ])
        raise AssertionError("不可达的模型重试状态")

    def extract_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        """按实体、关系、补充属性三阶段抽取，失败时保留已完成阶段的记录。"""
        graph: dict[str, Any] = {
            "metadata": {key: chunk.get(key, "") for key in
                         ("document_id", "document_title", "section_id", "section_title", "source_file")},
            "source_text": chunk["text"], "entities": [], "relations": [], "status": "failed",
            "validation": {"issues": [], "attempt_history": []},
        }
        graph["metadata"].update({"chunk_id": chunk["id"], "modality": "text",
                                  "input_fingerprint": self.fingerprint(chunk)})
        history = graph["validation"]["attempt_history"]
        phase = "entities"
        try:
            entity_result = self._request(
                entity_messages(self.schema, chunk), "entities",
                lambda payload: validate_entities(payload, self.schema, chunk["text"]), history,
            )
            graph["entities"] = entity_result.records
            graph["validation"]["issues"].extend(entity_result.issues)
            graph["validation"]["rejected_entity_count"] = entity_result.rejected_count
            # 中文说明：错误响应与合法空数组不同，前者不能标记为成功完成。
            if any(issue["code"] == "invalid_response" for issue in entity_result.issues):
                graph["validation"]["failed_phase"] = phase
                return graph
            phase = "relations"
            if self.schema.candidate_relations(entity_result.records):
                relation_result = self._request(
                    relation_messages(self.schema, chunk, entity_result.records), "relations",
                    lambda payload: validate_relations(payload, self.schema, chunk["text"], entity_result.records), history,
                )
            else:
                relation_result = ValidationResult()
            graph["relations"] = relation_result.records
            graph["validation"]["issues"].extend(relation_result.issues)
            graph["validation"]["rejected_relation_count"] = relation_result.rejected_count
            if any(issue["code"] == "invalid_response" for issue in relation_result.issues):
                graph["validation"]["failed_phase"] = phase
                return graph
            phase = "attributes"
            if graph["entities"]:
                # 中文说明：分批复查实体，缩小单次提示和输出，降低长 JSON 截断或漏掉顶层数组的概率。
                attribute_result = ValidationResult()
                attribute_responses_invalid = False
                for batch_index, start in enumerate(range(0, len(graph["entities"]), ATTRIBUTE_BATCH_SIZE), start=1):
                    batch = graph["entities"][start:start + ATTRIBUTE_BATCH_SIZE]
                    history_start = len(history)
                    batch_result = self._request(
                        attribute_messages(self.schema, chunk, batch, graph["relations"]), phase,
                        lambda payload, batch=batch: validate_attribute_updates(
                            payload, self.schema, chunk["text"], batch,
                        ), history,
                    )
                    for attempt_record in history[history_start:]:
                        attempt_record["batch_index"] = batch_index
                        attempt_record["batch_entity_ids"] = [entity["id"] for entity in batch]
                    attribute_result.records.extend(batch_result.records)
                    attribute_result.issues.extend(batch_result.issues)
                    attribute_result.rejected_count += batch_result.rejected_count
                    attribute_responses_invalid = attribute_responses_invalid or any(
                        issue["code"] == "invalid_response" for issue in batch_result.issues
                    )
            else:
                attribute_result = ValidationResult()
            graph["validation"]["issues"].extend(attribute_result.issues)
            graph["validation"]["rejected_attribute_count"] = attribute_result.rejected_count
            # 中文说明：部分批次成功时仍合并其合格属性；失败批次不阻止保留已验证的增量。
            graph["entities"] = merge_attribute_updates(graph["entities"], attribute_result.records)
            if graph["entities"] and attribute_responses_invalid:
                graph["validation"]["failed_phase"] = phase
                return graph
            graph["validation"]["attribute_stage_completed"] = True
            # 中文说明：属性补齐后重新检查关系端点和层级约束。
            checked_relations = validate_relations(graph, self.schema, chunk["text"], graph["entities"])
            graph["relations"] = checked_relations.records
            graph["validation"]["issues"].extend(checked_relations.issues)
            graph["validation"]["rejected_relation_count"] += checked_relations.rejected_count
            issues = graph["validation"]["issues"]
            graph["status"] = "needs_review" if any(issue["severity"] == "error" for issue in issues) else (
                "completed_with_warnings" if issues else "completed"
            )
        except Exception as exc:
            # 中文说明：保留实体阶段已完成的有效记录，失败 Chunk 在续跑时会重新处理。
            graph["validation"]["failed_phase"] = phase
            graph["validation"]["issues"].append({"severity": "error", "code": "extraction_failed",
                                                     "location": phase, "message": f"{type(exc).__name__}: {exc}"})
        return graph
