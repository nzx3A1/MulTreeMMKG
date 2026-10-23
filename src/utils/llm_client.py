"""文本大模型客户端。

实现沿用提供脚本的重试、限流、思考标签清理与宽容 JSON 解析策略，并兼容工程原有的
``chat(messages)``、``chat_json(messages)`` 调用方式。
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import deque
from typing import Any, Optional, Sequence

from config.model_config import OpenAICompatibleConfig, settings
from .llm_rate_limiter import GLOBAL_LLM_RATE_LIMITER, qps_to_min_interval


logger = logging.getLogger(__name__)


def prompt_to_text(prompt: Any) -> str:
    """将字符串或结构化 Prompt 统一转换为模型可读的文本。"""

    return prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False, indent=2)


def _strip_code_fences(content: str) -> str:
    """去除模型返回值最外层的 Markdown 代码围栏。"""

    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_fragment(content: str) -> str:
    """从含有说明文字的模型输出中截取最可能的 JSON 对象或数组。"""

    text = _strip_code_fences(content)
    if not text or text[0] in "[{":
        return text
    # 中文说明：缺少根对象左括号时保留首个属性名，避免误取其内部数组而丢失外层字段。
    if re.match(r'^"(?:[^"\\]|\\.)*"\s*:', text):
        return text
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        return text
    start = min(starts)
    end = text.rfind("}" if text[start] == "{" else "]")
    return text[start : end + 1].strip() if end > start else text


def _normalize_json_punctuation(content: str) -> str:
    """仅转换字符串外的全角结构符号，保留证据、名称及转义字符串的原始内容。"""

    punctuation = str.maketrans("｛｝［］，：", "{}[],:")
    output: list[str] = []
    in_string = escaped = False
    for char in content.lstrip("\ufeff"):
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            output.append(char.translate(punctuation))
            if char == '"':
                in_string = True
    return "".join(output)


def _json_structure_tokens(content: str) -> tuple[str, ...] | None:
    """提取完整 JSON 词元；遇到截断字符串、非法字面量等内容错误时放弃修复。"""

    pattern = re.compile(
        r'\s*("(?:[^"\\]|\\.)*"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?'
        r'|true|false|null|[{}\[\],:])'
    )
    tokens: list[str] = []
    position = 0
    while position < len(content):
        match = pattern.match(content, position)
        if match is None:
            return tuple(tokens) if not content[position:].strip() else None
        token = match.group(1)
        # 中文说明：不能把 01、1e 等非法或截断数值拆成新字段来伪造修复结果。
        if token[0] not in '"{}[],:' and match.end() < len(content):
            if content[match.end()] not in ' \t\r\n{}[],:"':
                return None
        tokens.append(token)
        position = match.end()
    return tuple(tokens)


def _repair_json_structure(content: str) -> tuple[str, int] | None:
    """在有限搜索内只增删结构符号，修复多余或遗漏括号及分隔符，不改写任何数据词元。"""

    tokens = _json_structure_tokens(content)
    if not tokens:
        return None
    closing = {"{": "}", "[": "]"}
    pending = deque([(tokens, 0)])
    visited = {tokens}
    # 中文说明：限制编辑次数和候选数量，避免严重损坏的长输出耗尽解析时间。
    max_edits, max_candidates = 6, 128
    examined = 0
    while pending and examined < max_candidates:
        current, edits = pending.popleft()
        examined += 1
        candidate = " ".join(current)
        try:
            json.loads(candidate)
            return candidate, edits
        except json.JSONDecodeError as exc:
            if edits >= max_edits:
                continue
            offset = 0
            positions: dict[int, int] = {}
            for index, token in enumerate(current):
                positions[offset] = index
                offset += len(token) + 1
            positions[len(candidate)] = len(current)
            index = positions.get(exc.pos)
            if index is None:
                # 中文说明：错误位于字符串或数值内部，不靠更改括号来掩盖内容错误。
                continue
            proposals: list[tuple[str, ...]] = []
            token = current[index] if index < len(current) else ""
            previous = current[index - 1] if index else ""
            # 中文说明：在值位置出现“字符串 :”时，可确定缺少对象左括号。
            if token == ":" and previous.startswith('"'):
                proposals.append(current[:index - 1] + ("{",) + current[index - 1:])
            if token in ("{", "}", "[", "]"):
                proposals.append(current[:index] + current[index + 1:])
            if previous in ("{", "["):
                proposals.append(current[:index - 1] + current[index:])
            stack: list[str] = []
            for structural in current[:index]:
                if structural in closing:
                    stack.append(structural)
                elif stack and structural == closing[stack[-1]]:
                    stack.pop()
            if stack and (index == len(current) or token in ("}", "]")):
                proposals.append(current[:index] + (closing[stack[-1]],) + current[index:])
            if previous == "," and (index == len(current) or token in ("}", "]")):
                proposals.append(current[:index - 1] + current[index:])
            if exc.msg == "Expecting ',' delimiter" and token and token not in ("}", "]", ":", ","):
                proposals.append(current[:index] + (",",) + current[index:])
            if exc.msg == "Expecting ':' delimiter" and token and token not in ("}", "]", ":", ","):
                proposals.append(current[:index] + (":",) + current[index:])
            for proposal in proposals:
                if proposal not in visited and len(visited) < max_candidates:
                    visited.add(proposal)
                    pending.append((proposal, edits + 1))
    return None


def safe_json_loads(content: str) -> Any:
    """优先正常解析，再保守修复 JSON 结构；缺失数据或无法修复时仍返回空对象。"""

    try:
        fragment = _extract_json_fragment(_normalize_json_punctuation(_strip_code_fences(content)))
        try:
            return json.loads(fragment)
        except json.JSONDecodeError as exc:
            repaired = _repair_json_structure(fragment)
            if repaired is not None:
                repaired_text, edits = repaired
                logger.warning(
                    "[LLM] 已自动修复 JSON 结构：编辑 %s 处，原解析错误：%s（第 %s 行，第 %s 列）",
                    edits, exc.msg, exc.lineno, exc.colno,
                )
                return json.loads(repaired_text)
    except (TypeError, ValueError, RecursionError):
        pass
    return {}


class LLMClient:
    """OpenAI 兼容的文本模型客户端，提供串行、限流和自动重试请求。"""

    def __init__(
        self,
        config: Optional[OpenAICompatibleConfig] = None,
        client: Any = None,
        rate_limit_qps: Optional[float] = None,
    ) -> None:
        """保存模型配置和可选 SDK 注入客户端，SDK 在首次调用时才创建。"""

        self.config = config or settings.llm
        self._client = client
        configured_qps = float(os.getenv("LLM_RATE_LIMIT_QPS", "0"))
        effective_qps = configured_qps if rate_limit_qps is None else rate_limit_qps
        self.min_request_interval = qps_to_min_interval(float(effective_qps or 0))
        self.retry_attempts = max(1, int(os.getenv("LLM_RETRY_ATTEMPTS", "3")))
        self.retry_backoff_base = float(os.getenv("LLM_RETRY_BACKOFF_BASE", "2"))
        self.extra_throttle_sec = float(os.getenv("LLM_EXTRA_THROTTLE_SEC", "0"))

    def _get_client(self) -> Any:
        """延迟创建 OpenAI SDK 客户端，便于测试注入假客户端。"""

        if self._client is None:
            if not self.config.api_key:
                raise ValueError("LLM_API_KEY or OPENAI_API_KEY is required to call LLM service")
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_secs,
            )
        return self._client

    def _rate_limit(self) -> None:
        """在每次请求开始前执行全局 QPS 限流。"""

        GLOBAL_LLM_RATE_LIMITER.wait(self.min_request_interval)

    def _thinking_extra_body(self, model: str) -> dict[str, Any]:
        """按当前文本模型的接口契约生成思考开关参数。"""

        # 中文说明：内网 Qwen 3.6 通过 chat_template_kwargs 接收思考开关；若放在顶层，
        # 服务端会忽略该参数，短输出上限可能被 reasoning 内容耗尽而返回空 content。
        normalized_model = str(model).lower()
        # 中文说明：CUP DeepSeek 接口要求将 thinking 放入 chat_template_kwargs，避免影响其他模型。
        if "deepseek" in normalized_model or "qwen-3.6" in normalized_model:
            thinking_key = "thinking" if "deepseek" in normalized_model else "enable_thinking"
            return {"chat_template_kwargs": {thinking_key: self.config.enable_thinking}}
        return {"enable_thinking": self.config.enable_thinking}

    @staticmethod
    def _normalize_messages(prompt_or_messages: Any) -> list[dict[str, Any]]:
        """兼容原有 messages 调用与提供脚本的单个 Prompt 调用。"""

        if isinstance(prompt_or_messages, Sequence) and not isinstance(prompt_or_messages, (str, bytes)):
            return [dict(message) for message in prompt_or_messages]
        return [{"role": "user", "content": prompt_to_text(prompt_or_messages)}]

    def chat_with_retry(self, prompt: Any, task_name: str = "", **kwargs: Any) -> str:
        """按指数退避重试模型请求，并清理返回中的 ``<think>`` 内容。"""

        messages = self._normalize_messages(prompt)
        model = kwargs.pop("model", self.config.model)
        extra_body = self._thinking_extra_body(model)
        extra_body.update(kwargs.pop("extra_body", {}))
        request = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.config.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.config.max_tokens),
            "timeout": kwargs.pop("timeout", self.config.timeout_secs),
            **kwargs,
        }
        if extra_body:
            request["extra_body"] = extra_body

        last_error: Exception | None = None
        for attempt in range(self.retry_attempts):
            try:
                self._rate_limit()
                if self.extra_throttle_sec > 0:
                    time.sleep(self.extra_throttle_sec)
                response = self._get_client().chat.completions.create(**request)
                text = (response.choices[0].message.content or "").strip()
                return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
            except Exception as exc:  # 网络或服务端异常均按同一策略重试。
                last_error = exc
                if attempt < self.retry_attempts - 1:
                    logger.warning(
                        "[LLM] 请求失败，准备第 %s/%s 次重试：%s",
                        attempt + 2,
                        self.retry_attempts,
                        exc,
                    )
                    time.sleep(self.retry_backoff_base**attempt)

        label = f"（{task_name}）" if task_name else ""
        raise RuntimeError(f"LLM 请求失败{label}，已重试 {self.retry_attempts} 次：{last_error}") from last_error

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """发送 OpenAI 兼容消息并返回清理后的纯文本响应。"""

        return self.chat_with_retry(messages, **kwargs)

    def call_json(self, prompt: Any, task_name: str = "") -> Any:
        """调用模型并以提供脚本的宽容方式解析 JSON 响应。"""

        return safe_json_loads(self.chat_with_retry(prompt, task_name=task_name))

    def call_openai_json(self, prompt: Any, task_name: str = "") -> Any:
        """保留 BaseModel 风格方法名，方便旧脚本迁移到本工程。"""

        return self.call_json(prompt, task_name=task_name)

    def chat_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any] | list[Any]:
        """请求并解析 JSON，自动修复可恢复的结构错误，修复失败时返回空对象。"""

        payload, _ = self.chat_json_with_raw(messages, **kwargs)
        return payload

    def chat_json_with_raw(
        self, messages: list[dict[str, Any]], **kwargs: Any,
    ) -> tuple[dict[str, Any] | list[Any], str]:
        """返回解析后的 JSON 和原始文本，供抽取校验失败时定位模型输出问题。"""

        kwargs.setdefault("response_format", {"type": "json_object"})
        raw_response = self.chat(messages, **kwargs)
        return safe_json_loads(raw_response), raw_response


# 中文说明：旧抽取脚本可继续使用 LLMJsonClient 名称，无需改动调用点。
LLMJsonClient = LLMClient


class DisabledLLMClient:
    """用于只校验流程的空实现，避免触发真实模型请求。"""

    def chat(self, messages: Any, **kwargs: Any) -> str:
        """返回空 JSON 字符串以兼容文本调用接口。"""

        _ = messages, kwargs
        return "{}"

    def chat_json(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        """返回空对象以兼容结构化抽取接口。"""

        _ = messages, kwargs
        return {}
