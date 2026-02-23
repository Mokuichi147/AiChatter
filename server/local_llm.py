import logging
import re
from typing import AsyncIterator

from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)


class LocalLLM:
    def __init__(self) -> None:
        # Ollama OpenAI互換エンドポイントを使用
        self.client = AsyncOpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama",  # Ollamaはapi_keyを使わないためダミー値
        )
        # 文末句読点で分割（TTS単位を小さくして初期応答を早める）
        self._sentence_pattern = re.compile(r"(?<=[。！？\.\!\?])\s*")

    async def generate_stream(
        self,
        text: str,
        history: list[dict],
    ) -> AsyncIterator[str]:
        messages = [{"role": "system", "content": settings.system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": text})

        logger.info(f"LLMリクエスト: '{text}' (モデル: {settings.llm_model})")

        stream = await self.client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=512,
        )

        buffer = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue

            buffer += delta

            # 文末（。！？.!?）で区切ってTTSに渡す
            parts = self._sentence_pattern.split(buffer)
            for sentence in parts[:-1]:
                sentence = sentence.strip()
                if sentence:
                    logger.debug(f"LLMチャンク: '{sentence}'")
                    yield sentence

            buffer = parts[-1]

        # 残りのバッファを出力
        if buffer.strip():
            logger.debug(f"LLMラスト: '{buffer.strip()}'")
            yield buffer.strip()
