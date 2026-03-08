"""PCのマイクとスピーカーを使った音声対話CLIモード。

マイクは常時バックグラウンドで録音する。
スピーカー再生中はVADを抑制してエコー回り込みを防止し、
ASR/LLM/TTS合成の処理待ち中はマイクを聞いて割り込みを検出する。
割り込み時はASR結果を先に検証し、有効な発話のみ前の処理を中断する。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Optional

import numpy as np
import sounddevice as sd

from ai_chatter import config
from ai_chatter.config import prompt_config, settings
from ai_chatter.local_asr import LocalASR
from ai_chatter.local_llm import LocalLLM, TextChunk, ToolCallRequest
from ai_chatter.local_tts import LocalTTS
from ai_chatter.skills import SkillProvider
from ai_chatter.speaker_id import SpeakerIdentifier
from ai_chatter.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 480  # 30ms @ 16kHz

# VADパラメータ
SILENCE_THRESHOLD = 300  # RMSエネルギー閾値
SILENCE_DURATION = 1.5  # 発話終了と判定する無音秒数
MIN_SPEECH_DURATION = 0.3  # 最短発話秒数
PLAYBACK_COOLDOWN = 0.3  # 再生停止後のクールダウン秒数（残響回避）
MAX_TOOL_ROUNDS = 5

# TTS再生時のPC向け音量スケール（synthesize_chunksのESP32向けスケールを補正）
PC_VOLUME_AMPLIFY = 10.0


class VoiceCLI:
    """PCマイク/スピーカーで音声対話するCLIランナー。

    マイクを常時バックグラウンドで録音し、VADで発話を検出する。
    スピーカー再生中はVADを抑制してエコーの回り込みを防止する。
    割り込み時はASR結果を先に検証し、有効な発話のみ前の処理を中断する。
    """

    def __init__(
        self,
        asr: LocalASR,
        llm: LocalLLM,
        tts: LocalTTS,
        tool_registry: Optional[ToolRegistry] = None,
        skill_provider: Optional[SkillProvider] = None,
        speaker_id: Optional[SpeakerIdentifier] = None,
    ) -> None:
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.tool_registry = tool_registry
        self.skill_provider = skill_provider
        self.speaker_id = speaker_id
        self._history: list[dict] = []
        self._gpu_lock = asyncio.Lock()

        # 割り込み制御
        self._interrupted = False

        # 再生状態（エコー抑制用）
        self._is_playing = False
        self._playback_ended_at: float = 0.0

        # マイクリスナー→メインループ間のキュー
        self._utterance_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _build_system_prompt(self, skill_context: str = "") -> str:
        system_prompt = config.character.persona.system_prompt

        if prompt_config.output_rules:
            system_prompt += "\n\n" + prompt_config.output_rules.strip()

        if (
            self.tool_registry
            and not self.tool_registry.is_empty
            and settings.tools_enabled
            and prompt_config.tool_guide_base
        ):
            system_prompt += "\n\n" + prompt_config.tool_guide_base.strip()

        if (
            settings.conversation_mode == "group"
            and prompt_config.group_rules
        ):
            system_prompt += "\n\n" + prompt_config.group_rules.strip()

        if skill_context:
            system_prompt += "\n\n" + skill_context

        now = datetime.now()
        system_prompt = system_prompt.replace(
            "{{DATETIME}}", now.strftime("%Y年%m月%d日 %H:%M")
        )
        return system_prompt

    def _is_vad_suppressed(self) -> bool:
        """スピーカー再生中またはクールダウン中はVADを抑制する。"""
        if self._is_playing:
            return True
        if self._playback_ended_at > 0:
            elapsed = time.monotonic() - self._playback_ended_at
            if elapsed < PLAYBACK_COOLDOWN:
                return True
        return False

    async def _start_mic_listener(self) -> None:
        """マイクを常時録音し、VADで発話を検出してキューに送信する。

        このコールバックはキューに発話データを送るのみで、
        パイプラインの中断判断はメインループ側で行う。
        """
        silence_blocks_needed = int(SILENCE_DURATION / (BLOCK_SIZE / SAMPLE_RATE))
        min_speech_blocks = int(MIN_SPEECH_DURATION / (BLOCK_SIZE / SAMPLE_RATE))

        audio_chunks: list[np.ndarray] = []
        speech_started = False
        silence_blocks = 0
        speech_blocks = 0
        loop = self._loop

        def callback(indata, frames, time_info, status):
            nonlocal speech_started, silence_blocks, speech_blocks

            if status:
                logger.debug(f"録音ステータス: {status}")

            # スピーカー再生中/クールダウン中はVADを完全に抑制
            if self._is_vad_suppressed():
                if speech_started:
                    audio_chunks.clear()
                    speech_started = False
                    silence_blocks = 0
                    speech_blocks = 0
                return

            chunk = indata[:, 0].copy()
            rms = np.sqrt(np.mean(chunk ** 2)) * 32768

            if rms > SILENCE_THRESHOLD:
                if not speech_started:
                    speech_started = True
                    logger.info("発話検出")
                silence_blocks = 0
                speech_blocks += 1
                audio_chunks.append(chunk)
            elif speech_started:
                silence_blocks += 1
                audio_chunks.append(chunk)
                if silence_blocks >= silence_blocks_needed:
                    if speech_blocks >= min_speech_blocks:
                        trim = min(silence_blocks, len(audio_chunks))
                        trimmed = audio_chunks[:-trim] if trim > 0 else list(audio_chunks)
                        if trimmed:
                            audio = np.concatenate(trimmed)
                            pcm = (audio * 32768).astype(np.int16).tobytes()
                            loop.call_soon_threadsafe(self._utterance_queue.put_nowait, pcm)
                    audio_chunks.clear()
                    speech_started = False
                    silence_blocks = 0
                    speech_blocks = 0

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=callback,
        )

        with stream:
            try:
                while True:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass

    async def _run_asr(self, pcm: bytes) -> str:
        """PCMデータをASRで音声認識する。"""
        loop = asyncio.get_event_loop()
        async with self._gpu_lock:
            done_event = threading.Event()
            def _asr_work():
                try:
                    return self.asr.transcribe(pcm)
                finally:
                    done_event.set()
            try:
                text = await loop.run_in_executor(None, _asr_work)
            except asyncio.CancelledError:
                done_event.wait(timeout=30)
                raise
        return text or ""

    async def _play_audio(self, pcm_data: bytes) -> None:
        """PCM 16bit 16kHzの音声をスピーカーで再生する。"""
        audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        audio = np.clip(audio * PC_VOLUME_AMPLIFY, -1.0, 1.0)

        loop = asyncio.get_event_loop()
        done_event = threading.Event()

        def _play():
            try:
                sd.play(audio, samplerate=SAMPLE_RATE)
                sd.wait()
            finally:
                done_event.set()

        self._is_playing = True
        try:
            await loop.run_in_executor(None, _play)
        except asyncio.CancelledError:
            sd.stop()
            done_event.wait(timeout=5)
            raise
        finally:
            self._is_playing = False
            self._playback_ended_at = time.monotonic()

    async def _process_with_text(
        self, text: str, pcm: bytes,
    ) -> None:
        """ASR済みテキストからLLM→TTS処理を行う。"""
        name = config.character.persona.name or "AI"
        loop = asyncio.get_event_loop()

        # グループモード: 話者識別
        speaker_name = None
        if self.speaker_id and settings.conversation_mode == "group":
            identified, score = await loop.run_in_executor(
                None, lambda: self.speaker_id.identify(pcm)
            )
            speaker_name = identified
            logger.info(f"話者識別: {identified} (score={score:.3f})")

        if speaker_name:
            user_text = f"[{speaker_name}] {text}"
            print(f"\r{speaker_name}> {text}")
        else:
            user_text = text
            print(f"\ryou> {text}")
        print(f"{name}> ", end="", flush=True)

        await self._process_response(user_text)

    async def _process_response(self, user_text: str) -> None:
        """LLMストリーミング → TTS合成 → スピーカー再生。"""
        skill_context = ""
        if self.skill_provider:
            skill_context = await self.skill_provider.retrieve(user_text)
        system_prompt = self._build_system_prompt(skill_context)

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        messages.extend(
            {"role": h["role"], "content": h["content"]}
            for h in self._history
        )
        messages.append({"role": "user", "content": user_text})

        tools = None
        if (
            self.tool_registry
            and not self.tool_registry.is_empty
            and settings.tools_enabled
        ):
            tools = self.tool_registry.to_openai_tools()

        full_response = ""
        loop = asyncio.get_event_loop()

        for round_num in range(MAX_TOOL_ROUNDS):
            if self._interrupted:
                break

            tool_call_requests: list[ToolCallRequest] = []

            async for event in self.llm.generate_stream(messages, tools):
                if self._interrupted:
                    break

                if isinstance(event, TextChunk):
                    text = event.text
                    full_response += text
                    print(text, end="", flush=True)

                    # TTS合成して再生
                    async with self._gpu_lock:
                        done_event = threading.Event()
                        def _tts_work(s=text):
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
                        if self._interrupted:
                            sd.stop()
                            break
                        await self._play_audio(chunk)

                elif isinstance(event, ToolCallRequest):
                    tool_call_requests.append(event)

            if self._interrupted:
                break

            if not tool_call_requests:
                if not full_response.strip():
                    messages.append({"role": "user", "content": "返答をお願いします。"})
                    tools = None
                    continue
                break

            for tc in tool_call_requests:
                messages.append(
                    {
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                )

            for tc in tool_call_requests:
                if self._interrupted:
                    break
                logger.info(f"ツール実行: {tc.name}")
                print(f"\n[ツール: {tc.name}]", flush=True)
                result = await self.tool_registry.execute(tc.name, tc.arguments)
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tc.id,
                        "output": result.content,
                    }
                )

            full_response = ""

        print()

        # 会話履歴を更新
        if full_response:
            self._history.append({"role": "user", "content": user_text})
            self._history.append({"role": "assistant", "content": full_response})
            if len(self._history) > 20:
                self._history = self._history[-20:]

    async def run(self) -> None:
        """音声対話メインループ。

        マイクリスナーが発話を検出するとキューに送信する。
        メインループはキューから発話を取得し、ASRで検証してから
        パイプラインの中断/開始を判断する。
        """
        self._loop = asyncio.get_event_loop()
        name = config.character.persona.name or "AI"
        print(f"\n音声対話モードを開始します (キャラクター: {name})")
        print("マイクに向かって話しかけてください。Ctrl+Cで終了します。\n")

        mic_task = asyncio.create_task(self._start_mic_listener())
        pipeline_task: Optional[asyncio.Task] = None
        utterance_waiter: Optional[asyncio.Task] = None

        try:
            print("🎤 聞いています...", end="", flush=True)
            while True:
                # 常にキューからの発話を待つタスクを用意
                if utterance_waiter is None or utterance_waiter.done():
                    utterance_waiter = asyncio.create_task(self._utterance_queue.get())

                wait_set: set[asyncio.Task] = {utterance_waiter}
                if pipeline_task and not pipeline_task.done():
                    wait_set.add(pipeline_task)

                done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

                # 新しい発話が到着した場合
                if utterance_waiter in done:
                    pcm = utterance_waiter.result()
                    utterance_waiter = None

                    # ASRで検証してから割り込みを判断
                    print("\r💭 認識中...        ", end="", flush=True)
                    text = await self._run_asr(pcm)

                    if not text or text == "はい。":
                        # 無効な音声 → 割り込みせずそのまま継続
                        if pipeline_task and not pipeline_task.done():
                            print("\r", end="", flush=True)
                        else:
                            print("\r🎤 聞いています...", end="", flush=True)
                        continue

                    # 有効な発話 → 進行中のパイプラインを中断
                    if pipeline_task and not pipeline_task.done():
                        self._interrupted = True
                        pipeline_task.cancel()
                        try:
                            await pipeline_task
                        except asyncio.CancelledError:
                            pass
                        self._is_playing = False
                        sd.stop()
                        print("\n[割り込み]")

                    self._interrupted = False
                    pipeline_task = asyncio.create_task(
                        self._process_with_text(text, pcm)
                    )

                # パイプラインが完了した場合
                if pipeline_task and pipeline_task in done:
                    try:
                        pipeline_task.result()
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"パイプラインエラー: {e}", exc_info=True)
                    pipeline_task = None
                    print("🎤 聞いています...", end="", flush=True)

        except KeyboardInterrupt:
            print("\n\n音声対話を終了しました。")
        finally:
            if utterance_waiter and not utterance_waiter.done():
                utterance_waiter.cancel()
            if pipeline_task and not pipeline_task.done():
                pipeline_task.cancel()
            mic_task.cancel()
            try:
                await mic_task
            except asyncio.CancelledError:
                pass
