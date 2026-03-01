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
    def _strip_think_tags(
        text: str, in_think: bool,
    ) -> tuple[str, bool, str]:
        """テキストから<think>...</think>タグ部分を除去する。

        チャンク境界でタグが分割されるケースに対応するため、
        末尾にタグの部分一致がある場合は pending として返す。

        Returns:
            (除去済みテキスト, thinkタグ内にいるか, 次チャンクへ持ち越すpending)
        """
        _OPEN = "<think>"
        _CLOSE = "</think>"
        result: list[str] = []
        pos = 0
        while pos < len(text):
            if in_think:
                end = text.find(_CLOSE, pos)
                if end == -1:
                    # 末尾が </think> の部分一致なら pending へ
                    for i in range(min(len(_CLOSE) - 1, len(text) - pos), 0, -1):
                        if text[-i:] == _CLOSE[:i]:
                            return "".join(result), True, text[-i:]
                    return "".join(result), True, ""
                pos = end + len(_CLOSE)
                in_think = False
            else:
                start = text.find(_OPEN, pos)
                if start == -1:
                    # 末尾が <think> の部分一致なら pending へ
                    remaining = text[pos:]
                    for i in range(min(len(_OPEN) - 1, len(remaining)), 0, -1):
                        if remaining[-i:] == _OPEN[:i]:
                            result.append(remaining[:-i])
                            return "".join(result), False, remaining[-i:]
                    result.append(remaining)
                    return "".join(result), False, ""
                result.append(text[pos:start])
                pos = start + len(_OPEN)
                in_think = True
        return "".join(result), in_think, ""

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
        pending = ""  # チャンク境界のタグ部分一致を持ち越すバッファ
        raw_content = ""  # デバッグ用: LLM生出力の記録
        # tool_calls断片を組み立てるためのバッファ
        tool_calls_buf: dict[int, dict] = {}

        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            # テキスト応答の処理
            if delta.content:
                raw_content += delta.content
                text = pending + delta.content
                clean, in_think, pending = self._strip_think_tags(
                    text, in_think
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

        # pendingに残ったテキストをフラッシュ（タグ未完成＝通常テキスト）
        if pending and not in_think:
            buffer += pending

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
