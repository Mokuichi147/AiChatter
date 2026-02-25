import asyncio
import logging
import struct
from typing import Awaitable, Callable, Optional

from local_asr import LocalASR
from local_llm import LocalLLM
from local_tts import LocalTTS

logger = logging.getLogger(__name__)

# WebSocketメッセージタイプ (Server → ESP32)
MSG_TTS_CHUNK = 0x02
MSG_TTS_END = 0x03

HEADER_SIZE = 7  # [type:1][seq:2][payload_len:4]


def make_header(msg_type: int, seq: int, payload_len: int) -> bytes:
    return struct.pack(">BHI", msg_type, seq & 0xFFFF, payload_len)


class AudioPipeline:
    def __init__(
        self,
        send_fn: Callable[[bytes], Awaitable[None]],
        asr: LocalASR,
        llm: LocalLLM,
        tts: LocalTTS,
    ) -> None:
        self.send_fn = send_fn
        self.asr = asr
        self.llm = llm
        self.tts = tts

        self._audio_buffer = bytearray()
        self._seq: int = 0
        self._history: list[dict] = []
        self._interrupted: bool = False
        self._current_task: Optional[asyncio.Task] = None

    def _next_seq(self) -> int:
        self._seq = (self._seq + 1) & 0xFFFF
        return self._seq

    async def process_audio_chunk(self, payload: bytes) -> None:
        self._audio_buffer.extend(payload)

    async def process_end_of_speech(self) -> None:
        audio_data = bytes(self._audio_buffer)
        self._audio_buffer.clear()
        self._interrupted = False

        # 前のタスクをキャンセル
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass

        if not audio_data:
            logger.warning("音声データが空のためスキップ")
            return

        self._current_task = asyncio.create_task(self._run_pipeline(audio_data))

    async def process_interrupt(self) -> None:
        logger.info("バージイン割り込み受信")
        self._interrupted = True

        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass

        self._audio_buffer.clear()

    async def _run_pipeline(self, audio_data: bytes) -> None:
        try:
            # --- ASR ---
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, self.asr.transcribe, audio_data)

            if not text:
                logger.info("ASR: 空の認識結果、TTS_END送信")
                header = make_header(MSG_TTS_END, self._next_seq(), 0)
                await self.send_fn(header)
                return

            logger.info(f"ASR認識: '{text}'")

            # --- LLM + TTS ストリーミング ---
            full_response = ""

            async for sentence in self.llm.generate_stream(text, self._history):
                if self._interrupted:
                    logger.info("割り込みによりパイプライン中断")
                    break

                full_response += sentence
                logger.info(f"TTS合成: '{sentence}'")

                # TTS一括生成してすぐに送信
                chunks = await loop.run_in_executor(
                    None,
                    lambda s=sentence: list(self.tts.synthesize_chunks(s)),
                )

                for chunk in chunks:
                    if self._interrupted:
                        break
                    MAX_CHUNK = 4096
                    offset = 0
                    while offset < len(chunk):
                        part = chunk[offset:offset + MAX_CHUNK]
                        header = make_header(
                            MSG_TTS_CHUNK, self._next_seq(), len(part)
                        )
                        await self.send_fn(header + part)
                        offset += MAX_CHUNK

            if not self._interrupted:
                # TTS終了通知
                header = make_header(MSG_TTS_END, self._next_seq(), 0)
                await self.send_fn(header)
                logger.info("TTS完了送信")

                # 会話履歴を更新（最大10往復 = 20メッセージ）
                self._history.append({"role": "user", "content": text})
                self._history.append({"role": "assistant", "content": full_response})
                if len(self._history) > 20:
                    self._history = self._history[-20:]

        except asyncio.CancelledError:
            logger.info("パイプラインタスクキャンセル")
            raise
        except Exception as e:
            logger.error(f"パイプラインエラー: {e}", exc_info=True)
