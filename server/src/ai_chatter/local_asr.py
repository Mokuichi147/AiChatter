import logging

import numpy as np

from ai_chatter.config import settings

logger = logging.getLogger(__name__)


class LocalASR:
    def __init__(self) -> None:
        logger.info(f"ASRモデル読み込み中: {settings.asr_model}")
        from mlx_audio.stt import load

        self.model = load(settings.asr_model)
        logger.info("ASRモデル読み込み完了")

    def transcribe(self, pcm_bytes: bytes, language: str = "Japanese") -> str:
        if not pcm_bytes:
            return ""

        # PCMバイト列 (16bit signed) → float32 numpy配列
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio) < 1600:  # 0.1秒未満は無視
            return ""

        result = self.model.generate(audio, language=language)

        text = result.text.strip() if result and result.text else ""
        # 句読点のみの結果は無視
        if text in ("。", "、", ".", ","):
            return ""
        logger.info(f"ASR結果: '{text}'")
        return text
