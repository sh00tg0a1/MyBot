"""
大模型调用工具 - 支持 OpenAI 和 Anthropic
"""
import os
from typing import Generator

import anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLM:
    """OpenAI 大模型客户端"""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "MiniMax-M1")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def chat(
        self,
        message: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        单轮对话

        Args:
            message: 用户消息
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大 token 数

        Returns:
            模型回复内容
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def chat_stream(
        self,
        message: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Generator[str, None, None]:
        """
        流式对话

        Args:
            message: 用户消息
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大 token 数

        Yields:
            模型回复内容片段
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def chat_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        多轮对话

        Args:
            messages: 消息历史列表 [{"role": "user/assistant", "content": "..."}]
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大 token 数

        Returns:
            模型回复内容
        """
        all_messages = [{"role": "system", "content": system_prompt}] + messages

        response = self.client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class AnthropicLLM:
    """Anthropic 大模型客户端（支持 thinking 流式输出）"""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "MiniMax-M2.1")

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url

        self.client = anthropic.Anthropic(**kwargs)

    def chat(
        self,
        message: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        单轮对话

        Args:
            message: 用户消息
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大 token 数

        Returns:
            模型回复内容
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": message}]},
            ],
        )

        # 提取文本内容
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text
        return text

    def chat_stream(
        self,
        message: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        include_thinking: bool = False,
    ) -> Generator[str, None, None]:
        """
        流式对话

        Args:
            message: 用户消息
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大 token 数
            include_thinking: 是否包含思考过程

        Yields:
            模型回复内容片段
        """
        stream = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": message}]},
            ],
            stream=True,
        )

        for chunk in stream:
            if chunk.type == "content_block_delta":
                if hasattr(chunk, "delta") and chunk.delta:
                    if chunk.delta.type == "thinking_delta" and include_thinking:
                        if chunk.delta.thinking:
                            yield chunk.delta.thinking
                    elif chunk.delta.type == "text_delta":
                        if chunk.delta.text:
                            yield chunk.delta.text

    def chat_stream_with_thinking(
        self,
        message: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Generator[dict, None, None]:
        """
        流式对话（分别返回 thinking 和 text）

        Args:
            message: 用户消息
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大 token 数

        Yields:
            {"type": "thinking"|"text", "content": str}
        """
        stream = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": message}]},
            ],
            stream=True,
        )

        for chunk in stream:
            if chunk.type == "content_block_delta":
                if hasattr(chunk, "delta") and chunk.delta:
                    if chunk.delta.type == "thinking_delta" and chunk.delta.thinking:
                        yield {"type": "thinking", "content": chunk.delta.thinking}
                    elif chunk.delta.type == "text_delta" and chunk.delta.text:
                        yield {"type": "text", "content": chunk.delta.text}

    def chat_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        多轮对话

        Args:
            messages: 消息历史列表 [{"role": "user/assistant", "content": "..."}]
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大 token 数

        Returns:
            模型回复内容
        """
        # 转换消息格式
        formatted = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            formatted.append({"role": msg["role"], "content": content})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=formatted,
        )

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text
        return text


# 便捷函数
def chat_completion(
    message: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 2000,
) -> str:
    """
    快速调用大模型

    Args:
        message: 用户消息
        system_prompt: 系统提示词
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大 token 数

    Returns:
        模型回复内容
    """
    llm = LLM(model=model)
    return llm.chat(
        message=message,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
