"""第三步：基于多模态 Chunk 的自底向上文档摘要。

本模块读取第二阶段 ``Document -> Section -> Chunk`` 的 JSON 目录树。叶子章节
直接将其文本、表格、公式和图片交给视觉语言模型；父章节则以已完成的子章节摘要
为主要依据继续归纳，最终在 ``document.summary`` 写入全文摘要。
"""
from __future__ import annotations

import argparse
import copy
import html
import json
import os
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Protocol, Sequence

from neo4j import GraphDatabase
from tqdm import tqdm

# 允许以 ``python src/summarizer/bottom_up_summarizer.py`` 方式直接执行。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.model_config import settings
from config.neo4j_config import settings as neo4j_settings
from prompts.summary_prompts import AGG_PROMPT, DOC_AGG_PROMPT, LEAF_PROMPT
from src.utils.llm_client import LLMClient
from src.utils.json_io import read_json, write_json
from src.utils.logger import get_logger
from src.utils.vlm_client import VLMClient


logger = get_logger(__name__)
"""第三步摘要日志记录器，同时写入控制台与项目日志文件。"""


class ChatClient(Protocol):
    """约束可注入的模型客户端接口，方便在测试中替换为假客户端。"""

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """接收 OpenAI 兼容 messages，并返回纯文本模型结果。"""


class SummaryGenerationError(RuntimeError):
    """封装章节摘要调用失败，避免将失败静默写入阶段产物。"""


class SummaryModelRouter:
    """按消息是否含图片，在高速文本模型与视觉模型之间自动路由。"""

    def __init__(
        self,
        text_client: ChatClient,
        vision_client: ChatClient,
        text_model: str,
        fallback_text_client: ChatClient | None = None,
        request_interval_secs: float = 0.0,
    ) -> None:
        """保存主、备用文本客户端与视觉客户端，文字章节优先走高速模型。"""

        self._text_client = text_client
        self._vision_client = vision_client
        self._text_model = text_model
        self._fallback_text_client = fallback_text_client or text_client
        self._request_interval_secs = max(0.0, request_interval_secs)
        self._last_request_at: float | None = None

    def _wait_for_rate_limit(self) -> None:
        """在调用大模型前等待，保证相邻请求达到配置的最小时间间隔。"""

        if self._last_request_at is not None:
            remaining = self._request_interval_secs - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at = time.monotonic()

    @staticmethod
    def _contains_image(messages: Sequence[Mapping[str, Any]]) -> bool:
        """判断 OpenAI 兼容消息是否包含真实的 image_url 内容块。"""

        for message in messages:
            content = message.get("content")
            if isinstance(content, list) and any(isinstance(part, Mapping) and part.get("type") == "image_url" for part in content):
                return True
        return False

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """将含图请求发送给视觉模型，其余请求发送给 MiniMax 高速文本模型。"""

        self._wait_for_rate_limit()
        if self._contains_image(messages):
            return self._vision_client.chat(messages, **kwargs)
        # MiniMax 官方仅允许 M3 关闭 thinking；M2.x 传入该参数也不会生效。
        if self._text_model == "MiniMax-M3":
            extra_body = dict(kwargs.pop("extra_body", {}))
            extra_body.setdefault("thinking", {"type": "disabled"})
            kwargs["extra_body"] = extra_body
        return self._text_client.chat(messages, **kwargs)

    def retry_after_empty(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """主模型只返回思考或空内容时重试；文本请求切换到关闭思考的 M3 兜底。"""

        self._wait_for_rate_limit()
        if self._contains_image(messages):
            return self._vision_client.chat(messages, **kwargs)
        return self._fallback_text_client.chat(messages, **kwargs)


DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "output" / "stage_03_document_summary.json"
"""第三步的默认输出位置；输入文件不会被原地覆盖。"""

MAX_TEXT_PER_CHUNK = 4_000
"""单个非图片 Chunk 放入提示词前的最大字符数，防止异常长段落挤占上下文。"""

MAX_SECTION_CONTEXT_CHARS = int(os.getenv("SUMMARY_MAX_CONTEXT_CHARS", "12000"))
"""单章发送给模型的文本上下文上限，避免过长输入拖慢模型首 token 返回。"""

MAX_IMAGES_PER_SECTION = int(os.getenv("SUMMARY_MAX_IMAGES_PER_SECTION", "3"))
"""单章最多传入的图片数；图注始终保留，图片只作为必要的视觉补充。"""

SCHEMA_KEYS = (
    "盆地", "构造单元", "凹陷/坳陷", "洼陷", "隆起", "斜坡", "构造带", "断层", "断裂带", "区块", "研究区",
    "地质时代", "组", "段", "亚段/小层", "层/单层", "目的层/储层段", "地层界面", "不整合面", "层序单元",
    "层序界面", "体系域", "岩性", "矿物", "岩石结构/组构", "沉积构造", "沉积相", "亚相", "微相", "沉积环境",
    "沉积体系", "成岩作用", "成岩阶段", "成岩相", "含油气系统/成藏系统", "烃源岩", "储层/储集层", "盖层",
    "圈闭", "运移通道", "输导体系", "成藏过程", "油气田", "油气藏", "烃类/油气", "流体界面", "孔隙结构",
    "裂缝", "溶蚀孔洞/洞穴", "甜点", "有机质", "干酪根类型", "生物标志化合物", "同位素", "地震剖面", "井",
    "测井资料", "岩心", "薄片", "露头", "地质图", "测试/试采", "样品", "实验", "分析方法",
)
"""章节 schemaKeys 的唯一合法词表；模型输出会再次经过程序校验。"""

CONCEPT_CATEGORIES_CYPHER = """
MATCH (n:ConceptCategory)
WHERE n.name IS NOT NULL
RETURN n.name AS name
"""
"""从 Neo4j Schema 库读取全部概念类别名称的只读查询。"""


def _load_concept_categories() -> tuple[str, ...]:
    """查询并去重 ConceptCategory 名称，为本次总结任务提供统一候选集。"""

    config = neo4j_settings.schema_db
    driver = GraphDatabase.driver(
        config.uri,
        auth=(config.username, config.password),
        connection_timeout=10.0,
        connection_acquisition_timeout=10.0,
    )
    try:
        driver.verify_connectivity()
        with driver.session(database=config.database) as session:
            rows = session.run(CONCEPT_CATEGORIES_CYPHER)
            category_names = (str(row["name"]).strip() for row in rows)
            categories = tuple(dict.fromkeys(name for name in category_names if name))
    finally:
        driver.close()
    if not categories:
        raise RuntimeError("Neo4j Schema 库中没有可用的 ConceptCategory.name，无法生成概念类别。")
    logger.info(f"已从 Neo4j 读取 {len(categories)} 个 ConceptCategory 候选名称。")
    return categories


def _section_count(sections: Sequence[Mapping[str, Any]]) -> int:
    """统计递归章节数，为 tqdm 提供准确的自底向上处理总量。"""

    count = 0
    for section in sections:
        count += 1
        children = section.get("children") or []
        if isinstance(children, list):
            count += _section_count([child for child in children if isinstance(child, Mapping)])
    return count


def _sections_grouped_by_height(sections: Sequence[MutableMapping[str, Any]]) -> list[list[MutableMapping[str, Any]]]:
    """按距叶子节点的高度分组，保证同组章节可并发且父章节晚于所有子章节。"""

    groups: dict[int, list[MutableMapping[str, Any]]] = {}

    def visit(section: MutableMapping[str, Any]) -> int:
        """递归计算单个章节高度，并将它放入对应的自底向上批次。"""

        children = section.get("children") or []
        valid_children = [child for child in children if isinstance(child, MutableMapping)] if isinstance(children, list) else []
        height = 0 if not valid_children else 1 + max(visit(child) for child in valid_children)
        groups.setdefault(height, []).append(section)
        return height

    for section in sections:
        visit(section)
    return [groups[height] for height in sorted(groups)]


def _start_section_progress(progress: Any, section: Mapping[str, Any]) -> None:
    """在提交模型请求前刷新进度条和日志，让长请求期间也能看到当前章节。"""

    title = str(section.get("title") or section.get("id") or "未命名章节")
    progress.set_postfix_str(f"正在生成：{title}", refresh=True)
    logger.info(f"开始生成章节摘要：{title}")


def _finish_section_progress(progress: Any, section: Mapping[str, Any]) -> None:
    """更新章节进度条并输出一条简洁日志，向使用者反馈当前处理位置。"""

    title = str(section.get("title") or section.get("id") or "未命名章节")
    progress.update(1)
    progress.set_postfix_str(f"当前：{title}", refresh=False)
    logger.info(f"章节摘要已完成（{progress.n}/{progress.total}）：{title}")


def _run_section_with_progress(
    section: MutableMapping[str, Any],
    client: ChatClient,
    concept_categories: Sequence[str],
    progress: Any,
) -> str:
    """生成单章摘要，并把本次任务的概念类别候选传给模型提示词。"""

    _start_section_progress(progress, section)
    summary = _summarize_section(section, client, concept_categories)
    _finish_section_progress(progress, section)
    return summary


def _clean_text(value: Any) -> str:
    """清理模型结果和 Chunk 文本中的 HTML、空白与常见 Markdown 包裹。"""

    text = "" if value is None else str(value)
    # M2.x 的原生兼容接口可能把思考内容放入 content 的 think 标签，摘要产物不应保留它。
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    text = re.sub(r"^\s*(?:摘要|总结)\s*[:：]\s*", "", text.strip())
    return re.sub(r"\s+", " ", text).strip()


def _ordered(items: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """按 JSON 中的同级顺序字段稳定排序，缺失顺序时保持原有相对位置。"""

    return sorted(items, key=lambda item: (int(item.get("order", 0)), str(item.get("id", ""))))


def _shorten(text: str, limit: int = MAX_TEXT_PER_CHUNK) -> str:
    """保留 Chunk 的开头语义片段，确保单次请求处于可控的上下文长度。"""

    cleaned = _clean_text(text)
    return cleaned if len(cleaned) <= limit else f"{cleaned[:limit].rstrip()}……"


def _text_part(chunk: Mapping[str, Any]) -> str:
    """将非视觉模态统一转换为带来源标记的模型文本片段。"""

    modality = str(chunk.get("modality", ""))
    chunk_id = str(chunk.get("id", "未命名 Chunk"))
    if modality == "text":
        content = chunk.get("text")
        label = "正文"
    elif modality == "table":
        content = "\n".join(filter(None, (str(chunk.get("caption") or ""), str(chunk.get("markdown") or ""))))
        label = "表格"
    elif modality == "formula":
        content = "\n".join(filter(None, (str(chunk.get("caption") or ""), str(chunk.get("latex") or ""))))
        label = "公式"
    elif modality == "image":
        content = chunk.get("caption") or ""
        label = "图片图注"
    else:
        return ""
    content = _shorten(str(content))
    return f"[{label} | {chunk_id}]\n{content}" if content else ""


def _image_parts(chunk: Mapping[str, Any]) -> list[dict[str, Any]]:
    """把存在的图片路径转换成 OpenAI 兼容 image_url 内容块。"""

    if str(chunk.get("modality", "")) != "image":
        return []
    raw_paths = chunk.get("image_path") or []
    paths = [raw_paths] if isinstance(raw_paths, (str, Path)) else raw_paths
    image_parts: list[dict[str, Any]] = []
    for raw_path in paths:
        value = str(raw_path)
        # URL、data URL 可直接发送；本地路径必须存在才进行 base64 编码。
        if value.startswith(("http://", "https://", "data:image/")) or Path(value).is_file():
            image_parts.append({"type": "image_url", "image_url": {"url": VLMClient.normalize_image_ref(value)}})
    return image_parts


def _chunk_content_parts(chunks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """构建有总量上限的多模态消息内容，减少输入传输和视觉处理耗时。"""

    parts: list[dict[str, Any]] = []
    text_char_count = 0
    image_count = 0
    for chunk in _ordered(chunks):
        text = _text_part(chunk)
        if text and text_char_count < MAX_SECTION_CONTEXT_CHARS:
            remaining = MAX_SECTION_CONTEXT_CHARS - text_char_count
            bounded_text = text if len(text) <= remaining else f"{text[:remaining].rstrip()}……"
            parts.append({"type": "text", "text": bounded_text})
            text_char_count += len(bounded_text)
        for image_part in _image_parts(chunk):
            if image_count >= MAX_IMAGES_PER_SECTION:
                break
            parts.append(image_part)
            image_count += 1
    return parts


def _summary_prompt(
    title: str,
    child_summaries: Sequence[tuple[str, str]],
    has_chunks: bool,
    concept_categories: Sequence[str],
    is_document: bool = False,
) -> str:
    """按总结层级组装提示词，并加入 Neo4j 概念类别候选集。"""

    child_text = "\n".join(f"- 《{child_title}》：{summary}" for child_title, summary in child_summaries)
    if is_document:
        template = DOC_AGG_PROMPT
        content = child_text
    elif child_summaries:
        template = AGG_PROMPT
        direct_chunk_note = "\n\n直属多模态 Chunk 将作为本消息后续内容块提供。" if has_chunks else ""
        content = f"{child_text}{direct_chunk_note}"
    else:
        template = LEAF_PROMPT
        content = "文本、表格、公式和图片将作为本消息后续内容块提供。"
    prompt = template.format(title=title or "未命名章节", content=content)
    schema_options = json.dumps(SCHEMA_KEYS, ensure_ascii=False)
    category_options = json.dumps(list(concept_categories), ensure_ascii=False)
    return f"""{prompt}

Output Format:
请严格只输出一个 JSON 对象，不要使用 Markdown 代码块：
{{"summary":"一个自然段摘要","schemaKeys":["最相关词1","最相关词2"],"ConceptCategories":["概念类别1","概念类别2","概念类别3"]}}
schemaKeys 是后续 Schema 召回的重要依据。请逐项检查直属 Chunk 中明确出现或具有直接语义对应的实体；宁可保留多个有直接证据的候选。仅在全部白名单词都无直接依据时返回空数组。每个词必须原样取自以下白名单，不得创造、拆分或改写：
{schema_options}

ConceptCategories 用于标识当前章节或全文的核心概念领域。请从下列 Neo4j ConceptCategory.name 候选中选择与当前内容最相关的三个，必须原样返回，不得创造、拆分或改写：
{category_options}
"""


def _parse_summary_result(value: Any, concept_categories: Sequence[str]) -> tuple[str, list[str], list[str]]:
    """解析模型结果，并过滤 schemaKeys 与 ConceptCategories 中的非法值。"""

    cleaned = _clean_text(value)
    if not cleaned:
        return "", [], []
    json_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        payload = json.loads(json_text)
    except (json.JSONDecodeError, TypeError):
        # 保留纯文本摘要供调用层识别，并由调用层针对缺失的结构化字段发起重试。
        return cleaned, [], []
    if not isinstance(payload, Mapping):
        return "", [], []
    summary = _clean_text(payload.get("summary"))
    raw_keys = payload.get("schemaKeys")
    valid_keys: list[str] = []
    if isinstance(raw_keys, list):
        allowed = set(SCHEMA_KEYS)
        for key in raw_keys:
            normalized = str(key).strip()
            if normalized in allowed and normalized not in valid_keys:
                valid_keys.append(normalized)
    raw_categories = payload.get("ConceptCategories")
    valid_categories: list[str] = []
    if isinstance(raw_categories, list):
        allowed_categories = set(concept_categories)
        for category in raw_categories:
            normalized = str(category).strip()
            if normalized in allowed_categories and normalized not in valid_categories:
                valid_categories.append(normalized)
            if len(valid_categories) == 3:
                break
    return summary, valid_keys, valid_categories


def _call_model(
    client: ChatClient,
    title: str,
    child_summaries: Sequence[tuple[str, str]],
    chunks: Sequence[Mapping[str, Any]],
    concept_categories: Sequence[str],
    is_document: bool = False,
) -> tuple[str, list[str], list[str]]:
    """向模型发送章节材料，返回摘要及经过候选集校验的两个数组字段。"""

    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": _summary_prompt(
            title, child_summaries, bool(chunks), concept_categories, is_document=is_document
        ),
    }]
    content.extend(_chunk_content_parts(chunks))
    messages = [{"role": "user", "content": content}]
    try:
        result = client.chat(messages)
    except Exception as exc:  # 调用失败时保留章节上下文，便于定位坏图片或网络问题。
        raise SummaryGenerationError(f"章节《{title or '未命名章节'}》摘要生成失败：{exc}") from exc
    summary, schema_keys, selected_categories = _parse_summary_result(result, concept_categories)
    expected_category_count = min(3, len(concept_categories))
    if not summary or len(selected_categories) != expected_category_count:
        logger.warning(
            f"章节《{title or '未命名章节'}》首次模型响应缺少有效摘要或未返回 "
            f"{expected_category_count} 个合法 ConceptCategories，正在自动重试。"
        )
        try:
            if isinstance(client, SummaryModelRouter):
                result = client.retry_after_empty(messages)
            else:
                result = client.chat(messages)
        except Exception as exc:
            raise SummaryGenerationError(f"章节《{title or '未命名章节'}》摘要重试失败：{exc}") from exc
        summary, schema_keys, selected_categories = _parse_summary_result(result, concept_categories)
    if not summary:
        raise SummaryGenerationError(f"章节《{title or '未命名章节'}》摘要生成失败：重试后模型仍返回空内容。")
    if len(selected_categories) != expected_category_count:
        raise SummaryGenerationError(
            f"章节《{title or '未命名章节'}》摘要生成失败：重试后仅返回 "
            f"{len(selected_categories)} 个合法 ConceptCategories，预期 {expected_category_count} 个。"
        )
    return summary, schema_keys, selected_categories


def _summarize_section(
    section: MutableMapping[str, Any],
    client: ChatClient,
    concept_categories: Sequence[str],
) -> str:
    """生成一个章节摘要；调用前其子章节已经在上一并发批次中完成。"""

    children = section.get("children") or []
    if not isinstance(children, list):
        raise ValueError(f"章节 {section.get('id', '未知')} 的 children 必须是列表。")
    child_summaries: list[tuple[str, str]] = []
    for child in _ordered(child for child in children if isinstance(child, MutableMapping)):
        child_summary = str(child.get("summary") or "")
        if not child_summary:
            raise RuntimeError(f"子章节 {child.get('id', '未知')} 尚未生成摘要，不能聚合父章节。")
        child_summaries.append((str(child.get("title") or child.get("id") or "未命名小节"), child_summary))

    chunks = section.get("chunks") or []
    if not isinstance(chunks, list):
        raise ValueError(f"章节 {section.get('id', '未知')} 的 chunks 必须是列表。")
    valid_chunks = [chunk for chunk in chunks if isinstance(chunk, Mapping)]
    section_title = str(section.get("title") or section.get("id") or "未命名章节")
    # 空章节不调用模型，仍写入 summary 以满足输出树中每个章节都有该字段的约定。
    if not child_summaries and not valid_chunks:
        section["summary"] = "本章节未包含可用于生成摘要的内容。"
        section["schemaKeys"] = []
        section["ConceptCategories"] = []
        logger.warning(f"章节《{section_title}》没有可摘要内容，已写入提示文本。")
    else:
        section["summary"], section["schemaKeys"], section["ConceptCategories"] = _call_model(
            client, str(section.get("title") or ""), child_summaries, valid_chunks, concept_categories
        )
    return str(section["summary"])


def summarize_document_tree(
    stage02: Mapping[str, Any],
    llm_client: ChatClient | None = None,
    show_progress: bool = True,
    concept_categories: Sequence[str] | None = None,
) -> dict[str, Any]:
    """为目录树添加摘要、schemaKeys 和 ConceptCategories，并返回独立深拷贝。"""

    result = copy.deepcopy(dict(stage02))
    document = result.get("document")
    if not isinstance(document, MutableMapping):
        raise ValueError("第二阶段输入必须包含对象类型的 document 字段。")
    sections = document.get("sections")
    if not isinstance(sections, list):
        raise ValueError("document.sections 必须是列表。")
    valid_sections = [section for section in sections if isinstance(section, MutableMapping)]
    category_options = (
        _load_concept_categories()
        if concept_categories is None
        else tuple(dict.fromkeys(str(item).strip() for item in concept_categories if str(item).strip()))
    )
    if not category_options:
        raise ValueError("ConceptCategory 候选列表不能为空。")
    # 纯文本请求使用 deepseek 高速模型，含图片请求保留原有视觉模型，兼顾吞吐量和多模态能力。
    client = llm_client or SummaryModelRouter(
        text_client=LLMClient(config=replace(settings.llm, model=settings.summary.model, max_tokens=settings.summary.max_tokens)),
        vision_client=LLMClient(config=replace(settings.vlm, max_tokens=settings.summary.max_tokens)),
        text_model=settings.summary.model,
        fallback_text_client=LLMClient(
            config=replace(settings.llm, model=settings.summary.fallback_model, max_tokens=settings.summary.max_tokens)
        ),
        request_interval_secs=settings.summary.request_interval_secs,
    )
    section_total = _section_count(valid_sections)
    logger.info(
        f"开始自底向上摘要：共 {section_total} 个章节，文本模型={settings.summary.model}，"
        f"兜底模型={settings.summary.fallback_model}，串行执行，"
        f"请求间隔={settings.summary.request_interval_secs} 秒，单次输出上限={settings.summary.max_tokens} tokens。"
    )
    progress = tqdm(
        total=section_total,
        desc="生成章节摘要",
        unit="章",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    top_summaries: list[tuple[str, str]] = []
    try:
        # 按章节高度自底向上串行执行，避免同时向大模型服务发送多个请求。
        for section_batch in _sections_grouped_by_height(valid_sections):
            for section in _ordered(section_batch):
                _run_section_with_progress(section, client, category_options, progress)

        for section in _ordered(valid_sections):
            top_summaries.append((str(section.get("title") or section.get("id") or "未命名章节"), str(section["summary"])))

        document_title = str(document.get("title") or "全文")
        if top_summaries:
            logger.info("章节摘要全部完成，开始生成全文摘要。")
            document["summary"], document["schemaKeys"], document["ConceptCategories"] = _call_model(
                client, document_title, top_summaries, [], category_options, is_document=True
            )
            logger.info(f"全文摘要已完成：{document_title}")
        else:
            document["summary"] = "文档未包含可用于生成摘要的章节。"
            document["schemaKeys"] = []
            document["ConceptCategories"] = []
            logger.warning("文档未包含章节，未调用模型生成全文摘要。")
    finally:
        progress.close()
    return result


def summarize_stage02_file(
    input_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    llm_client: ChatClient | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """读取阶段二 JSON、生成自底向上摘要，并以 UTF-8 JSON 写入第三阶段文件。"""

    stage02 = read_json(input_path)
    if not isinstance(stage02, Mapping):
        raise ValueError("第二阶段输入 JSON 根节点必须是对象。")
    logger.info(f"读取第二阶段目录树：{Path(input_path).resolve()}")
    result = summarize_document_tree(stage02, llm_client=llm_client, show_progress=show_progress)
    write_json(output_path, result)
    logger.info(f"摘要结果已写入：{Path(output_path).resolve()}")
    return result


def summarize_bottom_up(
    stage02: Mapping[str, Any],
    llm_client: ChatClient | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """保留原有函数名，作为 ``summarize_document_tree`` 的兼容入口。"""

    return summarize_document_tree(stage02, llm_client=llm_client, show_progress=show_progress)


def main() -> None:
    """提供命令行入口，默认从 output/stage_02_document_tree.json 读取。"""

    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="第三步：按章节树自底向上生成多模态摘要")
    parser.add_argument("--input", type=Path, default=project_root / "output" / "stage_02_document_tree.json", help="第二阶段目录树 JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="写入带 summary 的目录树 JSON")
    parser.add_argument("--no-progress", action="store_true", help="关闭 tqdm 章节进度条")
    args = parser.parse_args()
    result = summarize_stage02_file(args.input, args.output, show_progress=not args.no_progress)
    print(f"第三步摘要完成：{sum(1 for _ in _iter_sections(result['document']['sections']))} 个章节，输出文件：{args.output.resolve()}")


def _iter_sections(sections: Sequence[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    """深度优先遍历章节，仅供命令行统计最终写入了多少条章节摘要。"""

    for section in sections:
        yield section
        children = section.get("children") or []
        if isinstance(children, list):
            yield from _iter_sections([child for child in children if isinstance(child, Mapping)])


if __name__ == "__main__":
    main()
