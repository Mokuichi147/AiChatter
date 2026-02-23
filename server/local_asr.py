import logging

import numpy as np
from faster_whisper import WhisperModel

from config import settings

logger = logging.getLogger(__name__)


class LocalASR:
    def __init__(self) -> None:
        logger.info(f"Whisperモデル読み込み中: {settings.whisper_model_size}")
        # Apple M3: device="cpu", compute_type="int8" で高速動作
        self.model = WhisperModel(
            settings.whisper_model_size,
            device="cpu",
            compute_type="int8",
        )
        logger.info("Whisperモデル読み込み完了")

    def transcribe(self, pcm_bytes: bytes, language: str = "ja") -> str:
        if not pcm_bytes:
            return ""

        # PCMバイト列 (16bit signed) → numpy float32 → Whisper推論
        audio = (
            np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )

        if len(audio) < 1600:  # 0.1秒未満は無視
            return ""

        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=1,
            vad_filter=True,  # Whisper内蔵VADでノイズ除去
            vad_parameters={"min_silence_duration_ms": 300},
        )

        text = "".join(s.text for s in segments).strip()
        logger.info(f"ASR結果: '{text}' (言語: {info.language}, 確率: {info.language_probability:.2f})")
        return text
