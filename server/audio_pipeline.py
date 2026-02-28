import asyncio
import json
import logging
import struct
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Optional

from local_asr import LocalASR
from local_llm import LocalLLM, TextChunk, ToolCallRequest
from local_tts import LocalTTS
from config import character, settings

logger = logging.getLogger(__name__)

# WebSocketメッセージタイプ (Server → ESP32)
MSG_TTS_CHUNK = 0x02
MSG_TTS_END = 0x03
MSG_SLEEP = 0x04
MSG_WAKE = 0x05

HEADER_SIZE = 7  # [type:1][seq:2][payload_len:4]
MAX_TOOL_ROUNDS = 5
MAX_CHUNK = 4096


def make_header(msg_type: int, seq: int, payload_len: int) -> bytes:
    return struct.pack(">BHI", msg_type, seq & 0xFFFF, payload_len)


class AudioPipeline:
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
        self._history: list[dict] = self._load_history()
        self._interrupted: bool = False
        self._current_task: Optional[asyncio.Task] = None
        self._pipeline_lock = asyncio.Lock()
        self._device_sleeping: bool = False
        self._sleep_after_tts: bool = False
        self._ws_closed: bool = False

    @staticmethod
    def _history_path() -> Path:
        path = Path(settings.history_file)
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

    def _save_history(self) -> None:
        """会話履歴をJSONファイルに永続化する。"""
        path = self._history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(
                json.dumps(self._history, ensure_ascii=False, indent=2),
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
            chunks = await loop.run_in_executor(
                None,
                lambda s=segment: list(self.tts.synthesize_chunks(s)),
            )
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
        system_prompt = character.persona.system_prompt or settings.system_prompt

        # コンテキスト変数を展開
        now = datetime.now()
        system_prompt = system_prompt.replace(
            "{{DATETIME}}", now.strftime("%Y年%m月%d日 %H:%M")
        )

        # ツール有効時はツール使用ガイドを追加
        if (
            self.tool_registry
            and not self.tool_registry.is_empty
            and settings.tools_enabled
        ):
            system_prompt += (
                "\n\n## ツール使用ガイド\n"
                "あなたは以下のツールを使えます。積極的に活用してください。\n"
                "- save_memory: ユーザーの好み・名前・重要な事実・約束事など、"
                "後で役立ちそうな情報は自主的にメモしてください。"
                "明示的に「覚えて」と言われなくても構いません。\n"
                "- search_memory: ユーザーの発言に関連する記憶がありそうなとき、"
                "まず検索して過去の情報を活用してください。"
                "日付(after/before)で絞り込みもできます。\n"
                "- delete_memory: 不要な記憶を削除します。"
                "先にsearch_memoryでキーを確認してください。\n"
                "- set_volume: 音量調整を頼まれたときに使います。\n"
                "- set_notification: ユーザーが通知やリマインダーを頼んだときに使います。"
                "定期通知はrepeatで指定: 'every:30m'(30分毎), 'every:2h'(2時間毎), "
                "'cron:08:00'(毎日8時), 'cron:07:30:weekdays'(平日), 'cron:09:00:mon,fri'(曜日指定)。\n"
                "- list_notifications: 予約済みの通知一覧を確認します。\n"
                "- delete_notification: 通知をキャンセルしたいときに使います。"
                "先にlist_notificationsでIDを確認してください。\n"
                "- set_sleep: デバイスをスリープさせます。「おやすみ」等の"
                "就寝の挨拶を受けたら積極的に使ってください。\n"
                "- run_subagent_research: 時間がかかる調査をバックグラウンドで開始します。"
                "Web検索やメモリ検索を複数回行う詳細調査に使います。\n"
                "- list_subagent_jobs: サブエージェントジョブの進捗を確認します。\n"
                "- get_subagent_job: 完了ジョブの結果を取得します。"
                "結果をユーザーに伝えるかどうかは、会話文脈を見てあなたが判断してください。\n"
            )

        if extra_instruction:
            system_prompt += "\n\n" + extra_instruction

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
                self._history.append(
                    {"role": "user", "content": user_text, "created_at": now_str}
                )
                self._history.append(
                    {"role": "assistant", "content": full_response, "created_at": now_str}
                )
                if len(self._history) > 20:
                    self._history = self._history[-20:]
                self._save_history()

                # メモリストアにも会話を自動記録
                if self.memory_store:
                    key = f"conv_{now_str.replace(' ', '_').replace(':', '')}"
                    content = f"ユーザー: {user_text}\nアシスタント: {full_response}"
                    self.memory_store.save(key, content, auto=True)

    async def _run_pipeline(self, audio_data: bytes) -> None:
        tts_end_sent = False
        try:
            # --- ASR ---
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, self.asr.transcribe, audio_data)

            if not text:
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
                    {"role": h["role"], "content": h["content"]}
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

    async def process_button_press(self) -> None:
        """ボタン押下をLLMに伝えて応答を生成する。"""
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
                instruction = f"[通知] {text} — この内容をユーザーにキャラクターらしく伝えてください。"
                system_prompt = self._build_system_prompt()
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(
                    {"role": h["role"], "content": h["content"]}
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
