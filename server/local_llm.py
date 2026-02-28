import json
import logging
import re
from dataclasses import dataclass, field
from typing import AsyncIterator, Union

from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    text: str


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: str


StreamEvent = Union[TextChunk, ToolCallRequest]


class LocalLLM:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
        # 文末句読点で分割（TTS単位を小さくして初期応答を早める）
        self._sentence_pattern = re.compile(r"(?<=[。！？\.\!\?])\s*")

    async def generate_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        logger.info(f"LLMリクエスト (モデル: {settings.llm_model})")

        kwargs: dict = {
            "model": settings.llm_model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 512,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = await self.client.chat.completions.create(**kwargs)

        buffer = ""
        # tool_calls断片を組み立てるためのバッファ
        tool_calls_buf: dict[int, dict] = {}

        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            # テキスト応答の処理
            if delta.content:
                buffer += delta.content

                parts = self._sentence_pattern.split(buffer)
                for sentence in parts[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        logger.debug(f"LLMチャンク: '{sentence}'")
                        yield TextChunk(text=sentence)

                buffer = parts[-1]

            # tool_calls断片の組み立て
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_buf:
                        tool_calls_buf[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }
                    entry = tool_calls_buf[idx]
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            entry["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            entry["arguments"] += tc_delta.function.arguments

            # finish_reason確認
            if choice.finish_reason == "tool_calls":
                # 残りテキストがあればフラッシュ
                if buffer.strip():
                    yield TextChunk(text=buffer.strip())
                    buffer = ""
                # 組み立て済みtool_callsをyield
                for idx in sorted(tool_calls_buf.keys()):
                    entry = tool_calls_buf[idx]
                    logger.info(
                        f"ツール呼び出し: {entry['name']}({entry['arguments']})"
                    )
                    yield ToolCallRequest(
                        id=entry["id"],
                        name=entry["name"],
                        arguments=entry["arguments"],
                    )
                tool_calls_buf.clear()

        # 残りのバッファを出力
        if buffer.strip():
            logger.debug(f"LLMラスト: '{buffer.strip()}'")
            yield TextChunk(text=buffer.strip())
