import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Union

import litellm

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
        # 文末句読点で分割（TTS単位を小さくして初期応答を早める）
        self._sentence_pattern = re.compile(r"(?<=[。！？\.\!\?])\s*")

    @staticmethod
    def _strip_think_tags(text: str, in_think: bool) -> tuple[str, bool]:
        """テキストから<think>...</think>タグ部分を除去する。

        Returns:
            (除去済みテキスト, thinkタグ内にいるかのフラグ)
        """
        result: list[str] = []
        pos = 0
        while pos < len(text):
            if in_think:
                end = text.find("</think>", pos)
                if end == -1:
                    # タグが閉じていない → 残り全部破棄
                    break
                pos = end + len("</think>")
                in_think = False
            else:
                start = text.find("<think>", pos)
                if start == -1:
                    result.append(text[pos:])
                    break
                result.append(text[pos:start])
                pos = start + len("<think>")
                in_think = True
        return "".join(result), in_think

    @staticmethod
    def _split_json_objects(raw: str) -> list[str]:
        """結合されたJSON文字列 ({...}{...}) を個別のJSONに分離する。"""
        raw = raw.strip()
        if not raw:
            return ["{}"]
        # まず正常なJSONかチェック
        try:
            json.loads(raw)
            return [raw]
        except json.JSONDecodeError:
            pass
        # JSONデコーダで先頭から1つずつ切り出す
        results: list[str] = []
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(raw):
            # 空白をスキップ
            while pos < len(raw) and raw[pos] in " \t\n\r":
                pos += 1
            if pos >= len(raw):
                break
            try:
                _, end = decoder.raw_decode(raw, pos)
                results.append(raw[pos:end])
                pos = end
            except json.JSONDecodeError:
                # パースできない残りは最後の結果に結合するか無視
                logger.warning(f"JSON分離失敗 (pos={pos}): {raw[pos:]}")
                break
        return results if results else [raw]

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

        stream = await litellm.acompletion(**kwargs)

        buffer = ""
        in_think = False
        raw_content = ""  # デバッグ用: LLM生出力の記録
        # tool_calls断片を組み立てるためのバッファ
        tool_calls_buf: dict[int, dict] = {}

        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            # テキスト応答の処理
            if delta.content:
                raw_content += delta.content
                clean, in_think = self._strip_think_tags(
                    delta.content, in_think
                )
                if clean:
                    buffer += clean

                if buffer and not in_think:
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

        # LLM生出力をログ（thinkタグ除去前）
        if raw_content and not buffer.strip():
            logger.warning(f"LLM生出力あり(thinkタグ等で除去済み): {raw_content[:200]}")

        # 残りのバッファを出力
        if buffer.strip():
            logger.info(f"LLMラスト: '{buffer.strip()}'")
            yield TextChunk(text=buffer.strip())

        # 組み立て済みtool_callsをyield
        # (finish_reasonが"tool_calls"以外でも対応するためストリーム終了後に処理)
        if tool_calls_buf:
            for idx in sorted(tool_calls_buf.keys()):
                entry = tool_calls_buf[idx]
                # 結合されたJSON ({...}{...}) を分離して個別のツール呼び出しにする
                split_args = self._split_json_objects(entry["arguments"])
                for i, args in enumerate(split_args):
                    tc_id = entry["id"] if i == 0 else f"call_{uuid.uuid4().hex[:24]}"
                    logger.info(
                        f"ツール呼び出し: {entry['name']}({args})"
                    )
                    yield ToolCallRequest(
                        id=tc_id,
                        name=entry["name"],
                        arguments=args,
                    )
