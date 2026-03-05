import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Union

from openai import AsyncOpenAI

from config import llm_config

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
        api_key = llm_config.api_key or "no-key"
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=llm_config.base_url or None,
        )
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
                # <think> と </think>(迷子) の両方を探す
                start = text.find(_OPEN, pos)
                stray_close = text.find(_CLOSE, pos)

                # 迷子の </think> が先に見つかった場合（<think>なしで閉じタグだけ来るケース）
                if stray_close != -1 and (start == -1 or stray_close < start):
                    result.append(text[pos:stray_close])
                    pos = stray_close + len(_CLOSE)
                    continue

                if start == -1:
                    # 末尾が <think> または </think> の部分一致なら pending へ
                    remaining = text[pos:]
                    best_len = 0
                    for tag in (_OPEN, _CLOSE):
                        for i in range(min(len(tag) - 1, len(remaining)), 0, -1):
                            if remaining[-i:] == tag[:i] and i > best_len:
                                best_len = i
                                break
                    if best_len:
                        result.append(remaining[:-best_len])
                        return "".join(result), False, remaining[-best_len:]
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
        logger.info(f"LLMリクエスト (モデル: {llm_config.model})")

        kwargs: dict = {
            "model": llm_config.model,
            "input": messages,
            "stream": True,
            "temperature": 0.7,
            "max_output_tokens": 512,
        }
        if llm_config.reasoning:
            kwargs["reasoning"] = {"effort": llm_config.reasoning}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = await self._client.responses.create(**kwargs)

        buffer = ""
        in_think = False
        pending = ""  # チャンク境界のタグ部分一致を持ち越すバッファ
        raw_content = ""  # デバッグ用: LLM生出力の記録
        # 完成したfunction_callを蓄積するリスト
        tool_calls_list: list[dict] = []
        # output_item.addedで取得したcall_id/nameをoutput_indexで引くマップ
        tool_call_meta: dict[int, dict] = {}  # {call_id, name}

        async for event in stream:
            event_type = event.type

            # テキスト応答の処理
            if event_type == "response.output_text.delta":
                delta_text = event.delta
                if delta_text:
                    raw_content += delta_text
                    text = pending + delta_text
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

            # function_callアイテム追加（call_idとnameの早期取得）
            elif event_type == "response.output_item.added":
                item = event.item
                if item.type == "function_call":
                    tool_call_meta[event.output_index] = {
                        "call_id": item.call_id,
                        "name": item.name,
                    }

            # function_call引数の確定
            elif event_type == "response.function_call_arguments.done":
                meta = tool_call_meta.get(event.output_index, {})
                tool_calls_list.append({
                    "call_id": meta.get("call_id", ""),
                    "name": meta.get("name", "") or getattr(event, "name", ""),
                    "arguments": event.arguments,
                })

        # pendingに残ったテキストをフラッシュ（タグ未完成＝通常テキスト）
        if pending and not in_think:
            buffer += pending

        # LLM生出力をログ（thinkタグ除去前）
        if raw_content and not buffer.strip():
            logger.debug(f"LLM生出力あり(thinkタグ等で除去済み): {raw_content[:200]}")

        # 残りのバッファを出力
        if buffer.strip():
            logger.info(f"LLMラスト: '{buffer.strip()}'")
            yield TextChunk(text=buffer.strip())

        # 完成したtool_callsをyield
        for entry in tool_calls_list:
            # 結合されたJSON ({...}{...}) を分離して個別のツール呼び出しにする
            split_args = self._split_json_objects(entry["arguments"])
            for i, args in enumerate(split_args):
                tc_id = entry["call_id"] if i == 0 else f"call_{uuid.uuid4().hex[:24]}"
                logger.info(
                    f"ツール呼び出し: {entry['name']}({args})"
                )
                yield ToolCallRequest(
                    id=tc_id,
                    name=entry["name"],
                    arguments=args,
                )
