from __future__ import annotations

import _bootstrap  # noqa: F401

from src.utils.llm_client import LLMClient


def test_llm_simple_question():
    """向大模型提问一个简单问题，验证能正常返回回答。"""
    llm = LLMClient()
    response = llm.chat([{"role": "user", "content": "你是谁？"}])
    print(f"\n大模型回答: {response}")


def main() -> None:
    print("开始 LLM 连通性测试...")
    test_llm_simple_question()
    print("LLM 连通性测试通过！")


def test_llm_simple_question_with_number():
    from openai import OpenAI

    client = OpenAI(
        api_key="sk-lwctfhzpjhwclurfgdtpkwynqkawporxgvrhkjrtbuujayij",
        base_url="https://api.siliconflow.cn/v1"
    )

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4-Flash",
        messages=[
            {"role": "system", "content": "你是一个有用的助手"},
            {"role": "user", "content": "你好，请介绍一下你自己"}
        ]
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()