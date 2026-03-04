import asyncio
import json
import logging
import struct
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, ClassVar, Optional

from local_asr import LocalASR
from local_llm import LocalLLM, TextChunk, ToolCallRequest
from local_tts import LocalTTS
import config
from config import character_data_path, prompt_config, settings

logger = logging.getLogger(__name__)

# MLX (Metal GPU) 推論のグローバルロック
# ASR/TTSは run_in_executor でスレッドプールから実行されるが、
# Metal GPUコマンドバッファはスレッドセーフではないためシリアライズが必要
_gpu_lock = asyncio.Lock()

# WebSocketメッセージタイプ (Server → ESP32)
MSG_TTS_CHUNK = 0x02
MSG_TTS_END = 0x03
MSG_SLEEP = 0x04
MSG_WAKE = 0x05
MSG_DISPLAY_TEXT = 0x20
MSG_DISPLAY_IMAGE_BLOCK = 0x21

HEADER_SIZE = 7  # [type:1][seq:2][payload_len:4]
MAX_TOOL_ROUNDS = 5
MAX_CHUNK = 4096
DISPLAY_IMAGE_META_SIZE = 8


def make_header(msg_type: int, seq: int, payload_len: int) -> bytes:
    return struct.pack(">BHI", msg_type, seq & 0xFFFF, payload_len)


class AudioPipeline:
    _cached_history: ClassVar[Optional[list[dict]]] = None
    _cached_at: ClassVar[float] = 0.0
    _CACHE_TTL: ClassVar[float] = 120.0  # 2分以内 → キャッシュ利用

    def __init__(
        self,
        send_fn: Callable[[bytes], Awaitable[None]],
        asr: LocalASR,
        llm: LocalLLM,
        tts: LocalTTS,
        tool_registry=None,
        memory_store=None,
    ) -> None:
        self.send_fn = send_fn
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.tool_registry = tool_registry
        self.memory_store = memory_store

        self._audio_buffer = bytearray()
        self._seq: int = 0
        self._history: list[dict] = self._restore_history()
        self._interrupted: bool = False
        self._current_task: Optional[asyncio.Task] = None
        self._pipeline_lock = asyncio.Lock()
        self._device_sleeping: bool = False
        self._sleep_after_tts: bool = False
        self._ws_closed: bool = False

    @staticmethod
    def _history_path() -> Path:
        path = Path(character_data_path("history.json"))
        if not path.is_absolute():
            path = Path(__file__).parent / path
        return path

    @staticmethod
    def _load_history() -> list[dict]:
        """永続化された会話履歴から直近N往復を復元する。"""
        path = AudioPipeline._history_path()
        if not path.exists():
            return []
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
            # 直近N往復(=2*N メッセージ)を復元
            count = settings.history_restore_count * 2
            restored = entries[-count:] if len(entries) > count else entries
            logger.info(f"会話履歴を復元: {len(restored) // 2}往復 ({path})")
            return restored
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"会話履歴の読み込み失敗: {e}")
            return []

    @classmethod
    def _restore_history(cls) -> list[dict]:
        """キャッシュが有効ならキャッシュから、そうでなければファイルから履歴を復元する。"""
        if cls._cached_history and (time.monotonic() - cls._cached_at) < cls._CACHE_TTL:
            history = list(cls._cached_history)
            cls._cached_history = None
            logger.info(f"キャッシュから会話履歴を復元: {len(history) // 2}往復")
            return history
        cls._cached_history = None
        return cls._load_history()

    def _save_history(self, new_entries: list[dict]) -> None:
        """新しい会話エントリをファイルに追記する。"""
        path = self._history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = []
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
            existing.extend(new_entries)
            path.write_text(
                json.dumps(existing, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning(f"会話履歴の保存失敗: {e}")

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

    async def close(self) -> None:
        """パイプラインの進行中処理を停止し、参照を解放する。"""
        # 再接続に備えて履歴をキャッシュ
        if self._history:
            AudioPipeline._cached_history = list(self._history)
            AudioPipeline._cached_at = time.monotonic()
            logger.info(f"会話履歴をキャッシュ: {len(self._history) // 2}往復")

        self._interrupted = True
        self._ws_closed = True
        self._sleep_after_tts = False
        self._audio_buffer.clear()

        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"パイプラインクローズ時のタスク停止失敗: {e}")
        self._current_task = None

    async def _safe_send(self, data: bytes) -> bool:
        """WebSocketへ安全に送信する。失敗時はFalseを返す。"""
        try:
            await self.send_fn(data)
            return True
        except Exception:
            self._ws_closed = True
            return False

    async def _synthesize_and_send(self, sentence: str) -> None:
        """テキストをTTS合成してWebSocketで送信する。"""
        # 改行が含まれる場合は行単位で分割して順に合成する
        segments = [sentence]
        if "\n" in sentence or "\r" in sentence:
            segments = [seg.strip() for seg in sentence.splitlines() if seg.strip()]

        loop = asyncio.get_event_loop()
        for segment in segments:
            async with _gpu_lock:
                done_event = threading.Event()
                def _tts_work(s=segment):
                    try:
                        return list(self.tts.synthesize_chunks(s))
                    finally:
                        done_event.set()
                try:
                    chunks = await loop.run_in_executor(None, _tts_work)
                except asyncio.CancelledError:
                    done_event.wait(timeout=30)
                    raise
            for chunk in chunks:
                if self._interrupted or self._ws_closed:
                    break
                offset = 0
                while offset < len(chunk):
                    part = chunk[offset : offset + MAX_CHUNK]
                    header = make_header(MSG_TTS_CHUNK, self._next_seq(), len(part))
                    if not await self._safe_send(header + part):
                        return
                    offset += MAX_CHUNK

    def _build_system_prompt(self, extra_instruction: str = "") -> str:
        """システムプロンプトを組み立てる。"""
        system_prompt = config.character.persona.system_prompt or settings.system_prompt

        # 共通フォーマット指示を追加
        if prompt_config.output_rules:
            system_prompt += "\n\n" + prompt_config.output_rules.strip()

        # ツール有効時はツール使用ガイドを追加
        if (
            self.tool_registry
            and not self.tool_registry.is_empty
            and settings.tools_enabled
        ):
            system_prompt += "\n\n" + prompt_config.tool_guide.strip()

        if extra_instruction:
            system_prompt += "\n\n" + extra_instruction

        # コンテキスト変数を展開（全セクション結合後に一括置換）
        now = datetime.now()
        system_prompt = system_prompt.replace(
            "{{DATETIME}}", now.strftime("%Y年%m月%d日 %H:%M")
        )

        return system_prompt

    async def _process_llm_and_tts(
        self, messages: list[dict], user_text: str
    ) -> None:
        """LLMストリーミング → TTS合成 → 送信 → 履歴更新。"""
        # ツール設定
        tools = None
        if (
            self.tool_registry
            and not self.tool_registry.is_empty
            and settings.tools_enabled
        ):
            tools = self.tool_registry.to_openai_tools()

        # --- LLM + TTS ストリーミング (ツール実行ループ) ---
        full_response = ""

        for round_num in range(MAX_TOOL_ROUNDS):
            tool_call_requests: list[ToolCallRequest] = []

            async for event in self.llm.generate_stream(messages, tools):
                if self._interrupted or self._ws_closed:
                    logger.info("割り込みまたは接続断によりパイプライン中断")
                    break

                if isinstance(event, TextChunk):
                    full_response += event.text
                    logger.info(f"TTS合成: '{event.text}'")
                    await self._synthesize_and_send(event.text)
                elif isinstance(event, ToolCallRequest):
                    tool_call_requests.append(event)

            if self._interrupted or self._ws_closed:
                break

            # ツール呼び出しがなければ終了
            if not tool_call_requests:
                # 空応答（thinkタグ除去等）の場合、ツールなしで再試行
                if not full_response.strip() and not self._interrupted and not self._ws_closed:
                    logger.warning("LLM空応答を検出、ツールなしで再試行")
                    messages.append({"role": "user", "content": "返答をお願いします。"})
                    async for event in self.llm.generate_stream(messages, tools=None):
                        if self._interrupted or self._ws_closed:
                            break
                        if isinstance(event, TextChunk):
                            full_response += event.text
                            logger.info(f"TTS合成(リトライ): '{event.text}'")
                            await self._synthesize_and_send(event.text)
                break

            # assistantメッセージ（tool_calls付き）を追加
            assistant_msg = {"role": "assistant", "content": full_response or None}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in tool_call_requests
            ]
            messages.append(assistant_msg)

            # ツール実行
            for tc in tool_call_requests:
                logger.info(f"ツール実行: {tc.name}")
                result = await self.tool_registry.execute(tc.name, tc.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.content,
                    }
                )

            # 次のラウンドへ（LLM再呼び出し）
            full_response = ""

        if not self._interrupted and not self._ws_closed:
            # TTS終了通知
            header = make_header(MSG_TTS_END, self._next_seq(), 0)
            await self._safe_send(header)
            logger.info("TTS完了送信")

            # スリープ予約がある場合、TTS完了後に送信
            # (デバイス側で再生完了を待ってからスリープに入る)
            if self._sleep_after_tts:
                await self._send_sleep_now()

            # 会話履歴を更新（最終テキスト応答のみ記録、最大10往復=20メッセージ）
            if full_response:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                user_entry = {"role": "user", "content": user_text, "created_at": now_str}
                assistant_entry = {"role": "assistant", "content": full_response, "created_at": now_str}
                self._history.append(user_entry)
                self._history.append(assistant_entry)
                # メモリ上は直近分のみ保持、ファイルには全件蓄積
                if len(self._history) > 20:
                    self._history = self._history[-20:]
                self._save_history([user_entry, assistant_entry])

    async def _run_pipeline(self, audio_data: bytes) -> None:
        tts_end_sent = False
        try:
            # --- ASR ---
            loop = asyncio.get_event_loop()
            async with _gpu_lock:
                done_event = threading.Event()
                def _asr_work():
                    try:
                        return self.asr.transcribe(audio_data)
                    finally:
                        done_event.set()
                try:
                    text = await loop.run_in_executor(None, _asr_work)
                except asyncio.CancelledError:
                    done_event.wait(timeout=30)
                    raise

            if not text or text == "はい。":
                logger.info("ASR: 空の認識結果、TTS_END送信")
                header = make_header(MSG_TTS_END, self._next_seq(), 0)
                await self._safe_send(header)
                tts_end_sent = True
                return

            logger.info(f"ASR認識: '{text}'")

            async with self._pipeline_lock:
                # --- メッセージ組み立て ---
                system_prompt = self._build_system_prompt()
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(
                    {
                        "role": h["role"],
                        "content": f"[{h['created_at']}] {h['content']}" if h["role"] == "user" and h.get("created_at") else h["content"],
                    }
                    for h in self._history
                )
                messages.append({"role": "user", "content": text})

                await self._process_llm_and_tts(messages, text)
                tts_end_sent = True

        except asyncio.CancelledError:
            logger.info("パイプラインタスクキャンセル")
            raise
        except Exception as e:
            logger.error(f"パイプラインエラー: {e}", exc_info=True)
        finally:
            # エラー時もTTS_ENDを送信してESP32がPROCESSINGで停止するのを防ぐ
            if not tts_end_sent and not self._ws_closed:
                header = make_header(MSG_TTS_END, self._next_seq(), 0)
                await self._safe_send(header)
                logger.info("TTS_END送信 (エラーリカバリ)")

    async def send_sleep(self) -> None:
        """TTS完了後にデバイスをスリープさせるフラグを立てる。"""
        self._sleep_after_tts = True
        logger.info("スリープ予約 (TTS完了後に送信)")

    async def _send_sleep_now(self) -> None:
        """MSG_SLEEPを即座に送信する。"""
        header = make_header(MSG_SLEEP, self._next_seq(), 0)
        if await self._safe_send(header):
            self._device_sleeping = True
            self._sleep_after_tts = False
            logger.info("MSG_SLEEP送信")

    async def send_wake(self) -> None:
        """デバイスにウェイク指示を送信する。"""
        header = make_header(MSG_WAKE, self._next_seq(), 0)
        if await self._safe_send(header):
            self._device_sleeping = False
            logger.info("MSG_WAKE送信")

    async def send_display_text(
        self,
        text: str,
        size: int = 1,
        x: int = 0,
        y: int = 10,
        clear: bool = True,
    ) -> None:
        """デバイス画面にテキストを表示する。"""
        if text is None:
            text = ""
        size = max(1, min(4, int(size)))
        x = max(0, min(134, int(x)))
        y = max(0, min(239, int(y)))

        text_bytes = text.encode("utf-8")
        if len(text_bytes) > 512:
            text_bytes = text_bytes[:512]

        payload = struct.pack(">BBHH", size, 1 if clear else 0, x, y) + text_bytes
        header = make_header(MSG_DISPLAY_TEXT, self._next_seq(), len(payload))
        await self._safe_send(header + payload)
        logger.info(f"MSG_DISPLAY_TEXT送信 size={size} x={x} y={y} clear={clear}")

    async def send_display_image(
        self, rgb565: bytes, width: int, height: int, x: int = 0, y: int = 0
    ) -> None:
        """RGB565画像を複数ブロックに分割して送信する。"""
        width = int(width)
        height = int(height)
        x = int(x)
        y = int(y)

        if width <= 0 or height <= 0:
            raise ValueError("width/height は1以上で指定してください。")
        if x < 0 or y < 0 or x + width > 135 or y + height > 240:
            raise ValueError("描画領域が画面範囲外です。")

        expected = width * height * 2
        if len(rgb565) != expected:
            raise ValueError(
                f"RGB565データ長が不正です。expected={expected}, actual={len(rgb565)}"
            )

        bytes_per_row = width * 2
        max_rows = (MAX_CHUNK - DISPLAY_IMAGE_META_SIZE) // bytes_per_row
        if max_rows < 1:
            raise ValueError("画像の横幅が大きすぎて送信できません。")

        for row in range(0, height, max_rows):
            block_rows = min(max_rows, height - row)
            begin = row * bytes_per_row
            end = begin + block_rows * bytes_per_row
            block = rgb565[begin:end]
            payload = (
                struct.pack(">HHHH", x, y + row, width, block_rows) + block
            )
            header = make_header(
                MSG_DISPLAY_IMAGE_BLOCK, self._next_seq(), len(payload)
            )
            if not await self._safe_send(header + payload):
                return

        logger.info(f"MSG_DISPLAY_IMAGE_BLOCK送信 width={width} height={height} x={x} y={y}")

    async def process_button_press(self) -> None:
        """ボタン押下をLLMに伝えて応答を生成する。"""
        # 進行中のパイプラインを停止してから実行
        await self.process_interrupt()
        await self.generate_from_text(
            "[ボタン押下] ユーザーがボタンを押しました。反応してください。"
        )

    async def generate_from_text(self, text: str) -> None:
        """ASRをスキップし、指定テキストを直接LLM→TTS処理する（通知用）。"""
        tts_end_sent = False
        try:
            # スリープ中なら先にウェイクして安定待ち
            if self._device_sleeping:
                await self.send_wake()
                await asyncio.sleep(0.5)

            async with self._pipeline_lock:
                instruction = f"[通知] {text} — ユーザーはこの結果を待っています。内容を要約してキャラクターらしく必ず伝えてください。"
                system_prompt = self._build_system_prompt()
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(
                    {
                        "role": h["role"],
                        "content": f"[{h['created_at']}] {h['content']}" if h["role"] == "user" and h.get("created_at") else h["content"],
                    }
                    for h in self._history
                )
                messages.append({"role": "user", "content": instruction})

                self._interrupted = False
                await self._process_llm_and_tts(messages, instruction)
                tts_end_sent = True

        except asyncio.CancelledError:
            logger.info("通知タスクキャンセル")
            raise
        except Exception as e:
            logger.error(f"通知パイプラインエラー: {e}", exc_info=True)
        finally:
            if not tts_end_sent and not self._ws_closed:
                header = make_header(MSG_TTS_END, self._next_seq(), 0)
                await self._safe_send(header)
                logger.info("TTS_END送信 (通知エラーリカバリ)")
