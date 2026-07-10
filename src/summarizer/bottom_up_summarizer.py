"""第三步：基于多模态 Chunk 的自底向上文档摘要。

本模块读取第二阶段 ``Document -> Section -> Chunk`` 的 JSON 目录树。叶子章节
直接将其文本、表格、公式和图片交给视觉语言模型；父章节则以已完成的子章节摘要
为主要依据继续归纳，最终在 ``document.summary`` 写入全文摘要。
"""
from __future__ import annotations

import argparse
import copy
import html
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Protocol, Sequence

from tqdm import tqdm

# 允许以 ``python src/summarizer/bottom_up_summarizer.py`` 方式直接执行。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.model_config import settings
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
    ) -> None:
        """保存主、备用文本客户端与视觉客户端，文字章节优先走高速模型。"""

        self._text_client = text_client
        self._vision_client = vision_client
        self._text_model = text_model
        self._fallback_text_client = fallback_text_client or text_client

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

SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "MiniMax-M2.7-highspeed")
"""文本摘要默认模型；可改为 MiniMax-M3 以启用关闭思考模式。"""

SUMMARY_FALLBACK_MODEL = os.getenv("SUMMARY_FALLBACK_MODEL", "MiniMax-M3")
"""高速模型只输出思考或空内容时使用的文本兜底模型。"""

SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "800"))
"""摘要输出长度上限；与压缩型提示词匹配，避免沿用全局 8192 token 配额。"""

DEFAULT_SUMMARY_WORKERS = int(os.getenv("SUMMARY_WORKERS", "3"))
"""默认并发请求数；同一层级互不依赖的章节可同时生成摘要。"""


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
    is_document: bool = False,
) -> str:
    """按叶子、章节聚合或全文聚合场景加载石油地质领域提示词。"""

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
    return template.format(title=title or "未命名章节", content=content)


def _call_model(
    client: ChatClient,
    title: str,
    child_summaries: Sequence[tuple[str, str]],
    chunks: Sequence[Mapping[str, Any]],
    is_document: bool = False,
) -> str:
    """向视觉语言模型发送章节材料并规范化其摘要结果。"""

    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": _summary_prompt(title, child_summaries, bool(chunks), is_document=is_document),
    }]
    content.extend(_chunk_content_parts(chunks))
    messages = [{"role": "user", "content": content}]
    try:
        result = client.chat(messages)
    except Exception as exc:  # 调用失败时保留章节上下文，便于定位坏图片或网络问题。
        raise SummaryGenerationError(f"章节《{title or '未命名章节'}》摘要生成失败：{exc}") from exc
    summary = _clean_text(result)
    if not summary:
        logger.warning(f"章节《{title or '未命名章节'}》首次模型响应为空或仅含思考内容，正在自动重试。")
        try:
            if isinstance(client, SummaryModelRouter):
                result = client.retry_after_empty(messages)
            else:
                result = client.chat(messages)
        except Exception as exc:
            raise SummaryGenerationError(f"章节《{title or '未命名章节'}》摘要重试失败：{exc}") from exc
        summary = _clean_text(result)
    if not summary:
        raise SummaryGenerationError(f"章节《{title or '未命名章节'}》摘要生成失败：重试后模型仍返回空内容。")
    return summary


def _summarize_section(section: MutableMapping[str, Any], client: ChatClient) -> str:
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
        logger.warning(f"章节《{section_title}》没有可摘要内容，已写入提示文本。")
    else:
        section["summary"] = _call_model(client, str(section.get("title") or ""), child_summaries, valid_chunks)
    return str(section["summary"])


def summarize_document_tree(
    stage02: Mapping[str, Any],
    llm_client: ChatClient | None = None,
    show_progress: bool = True,
    max_workers: int = DEFAULT_SUMMARY_WORKERS,
) -> dict[str, Any]:
    """为第二阶段目录树添加章节和全文 ``summary``，并返回独立的深拷贝结果。"""

    result = copy.deepcopy(dict(stage02))
    document = result.get("document")
    if not isinstance(document, MutableMapping):
        raise ValueError("第二阶段输入必须包含对象类型的 document 字段。")
    sections = document.get("sections")
    if not isinstance(sections, list):
        raise ValueError("document.sections 必须是列表。")
    if max_workers < 1:
        raise ValueError("max_workers 必须大于或等于 1。")

    valid_sections = [section for section in sections if isinstance(section, MutableMapping)]
    # 纯文本请求使用 MiniMax 高速模型，含图片请求保留原有视觉模型，兼顾吞吐量和多模态能力。
    client = llm_client or SummaryModelRouter(
        text_client=LLMClient(config=replace(settings.llm, model=SUMMARY_MODEL, max_tokens=SUMMARY_MAX_TOKENS)),
        vision_client=LLMClient(config=replace(settings.vlm, max_tokens=SUMMARY_MAX_TOKENS)),
        text_model=SUMMARY_MODEL,
        fallback_text_client=LLMClient(config=replace(settings.llm, model=SUMMARY_FALLBACK_MODEL, max_tokens=SUMMARY_MAX_TOKENS)),
    )
    section_total = _section_count(valid_sections)
    logger.info(
        f"开始自底向上摘要：共 {section_total} 个章节，文本模型={SUMMARY_MODEL}，兜底模型={SUMMARY_FALLBACK_MODEL}，"
        f"并发数={max_workers}，单次输出上限={SUMMARY_MAX_TOKENS} tokens。"
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
        # 同一高度的章节没有上下游依赖，可安全并发；父章节会在下一批次等待子摘要完成。
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="summary") as executor:
            for section_batch in _sections_grouped_by_height(valid_sections):
                ordered_batch = _ordered(section_batch)
                futures = []
                for section in ordered_batch:
                    _start_section_progress(progress, section)
                    futures.append((section, executor.submit(_summarize_section, section, client)))
                # 按稳定章节顺序收集结果，使日志和进度条输出可预测。
                for section, future in futures:
                    future.result()
                    _finish_section_progress(progress, section)

        for section in _ordered(valid_sections):
            top_summaries.append((str(section.get("title") or section.get("id") or "未命名章节"), str(section["summary"])))

        document_title = str(document.get("title") or "全文")
        if top_summaries:
            logger.info("章节摘要全部完成，开始生成全文摘要。")
            document["summary"] = _call_model(client, document_title, top_summaries, [], is_document=True)
            logger.info(f"全文摘要已完成：{document_title}")
        else:
            document["summary"] = "文档未包含可用于生成摘要的章节。"
            logger.warning("文档未包含章节，未调用模型生成全文摘要。")
    finally:
        progress.close()
    return result


def summarize_stage02_file(
    input_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    llm_client: ChatClient | None = None,
    show_progress: bool = True,
    max_workers: int = DEFAULT_SUMMARY_WORKERS,
) -> dict[str, Any]:
    """读取阶段二 JSON、生成自底向上摘要，并以 UTF-8 JSON 写入第三阶段文件。"""

    stage02 = read_json(input_path)
    if not isinstance(stage02, Mapping):
        raise ValueError("第二阶段输入 JSON 根节点必须是对象。")
    logger.info(f"读取第二阶段目录树：{Path(input_path).resolve()}")
    result = summarize_document_tree(stage02, llm_client=llm_client, show_progress=show_progress, max_workers=max_workers)
    write_json(output_path, result)
    logger.info(f"摘要结果已写入：{Path(output_path).resolve()}")
    return result


def summarize_bottom_up(
    stage02: Mapping[str, Any],
    llm_client: ChatClient | None = None,
    show_progress: bool = True,
    max_workers: int = DEFAULT_SUMMARY_WORKERS,
) -> dict[str, Any]:
    """保留原有函数名，作为 ``summarize_document_tree`` 的兼容入口。"""

    return summarize_document_tree(stage02, llm_client=llm_client, show_progress=show_progress, max_workers=max_workers)


def main() -> None:
    """提供命令行入口，默认从 output/stage_02_document_tree.json 读取。"""

    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="第三步：按章节树自底向上生成多模态摘要")
    parser.add_argument("--input", type=Path, default=project_root / "output" / "stage_02_document_tree.json", help="第二阶段目录树 JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="写入带 summary 的目录树 JSON")
    parser.add_argument("--no-progress", action="store_true", help="关闭 tqdm 章节进度条")
    parser.add_argument("--workers", type=int, default=DEFAULT_SUMMARY_WORKERS, help="同层级章节摘要的并发请求数，默认 3")
    args = parser.parse_args()
    result = summarize_stage02_file(args.input, args.output, show_progress=not args.no_progress, max_workers=args.workers)
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
