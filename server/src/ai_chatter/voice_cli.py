"""PCのマイクとスピーカーを使った音声対話CLIモード。"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime
from typing import Optional

import numpy as np

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
MAX_TOOL_ROUNDS = 5

# TTS再生時のPC向け音量スケール（synthesize_chunksのESP32向けスケールを補正）
PC_VOLUME_AMPLIFY = 10.0


class VoiceCLI:
    """PCマイク/スピーカーで音声対話するCLIランナー。"""

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

    async def _record_utterance(self) -> Optional[bytes]:
        """マイクから音声を録音し、VADで発話終了を検出して返す。"""
        import sounddevice as sd

        loop = asyncio.get_event_loop()
        audio_chunks: list[np.ndarray] = []
        speech_started = False
        silence_blocks = 0
        silence_blocks_needed = int(SILENCE_DURATION / (BLOCK_SIZE / SAMPLE_RATE))
        min_speech_blocks = int(MIN_SPEECH_DURATION / (BLOCK_SIZE / SAMPLE_RATE))
        speech_blocks = 0
        stop_event = threading.Event()

        def callback(indata, frames, time_info, status):
            nonlocal speech_started, silence_blocks, speech_blocks
            if status:
                logger.debug(f"録音ステータス: {status}")

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
                    stop_event.set()
            else:
                # 発話前の無音は録音しない
                pass

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=callback,
        )

        with stream:
            await loop.run_in_executor(None, lambda: stop_event.wait(timeout=30))

        if not audio_chunks or speech_blocks < min_speech_blocks:
            return None

        # 末尾の無音部分をトリミング
        trim_count = min(silence_blocks, len(audio_chunks))
        if trim_count > 0:
            audio_chunks = audio_chunks[:-trim_count]

        if not audio_chunks:
            return None

        audio = np.concatenate(audio_chunks)
        pcm = (audio * 32768).astype(np.int16).tobytes()
        return pcm

    async def _play_audio(self, pcm_data: bytes) -> None:
        """PCM 16bit 16kHzの音声をスピーカーで再生する。"""
        import sounddevice as sd

        audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        # ESP32向けのボリュームスケールを補正して増幅
        audio = np.clip(audio * PC_VOLUME_AMPLIFY, -1.0, 1.0)

        loop = asyncio.get_event_loop()
        done_event = threading.Event()

        def _play():
            try:
                sd.play(audio, samplerate=SAMPLE_RATE)
                sd.wait()
            finally:
                done_event.set()

        try:
            await loop.run_in_executor(None, _play)
        except asyncio.CancelledError:
            sd.stop()
            done_event.wait(timeout=5)
            raise

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
            tool_call_requests: list[ToolCallRequest] = []

            async for event in self.llm.generate_stream(messages, tools):
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
                        await self._play_audio(chunk)

                elif isinstance(event, ToolCallRequest):
                    tool_call_requests.append(event)

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
        """音声対話メインループ。"""
        name = config.character.persona.name or "AI"
        print(f"\n音声対話モードを開始します (キャラクター: {name})")
        print("マイクに向かって話しかけてください。Ctrl+Cで終了します。\n")

        try:
            while True:
                print("🎤 聞いています...", end="", flush=True)
                pcm = await self._record_utterance()
                if pcm is None:
                    print("\r                    \r", end="", flush=True)
                    continue

                # ASR
                print("\r💭 認識中...        ", end="", flush=True)
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

                if not text or text == "はい。":
                    print("\r                    \r", end="", flush=True)
                    continue

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

        except KeyboardInterrupt:
            print("\n\n音声対話を終了しました。")
