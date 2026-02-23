import logging
from typing import Iterator

import numpy as np
from scipy import signal as scipy_signal

from config import settings

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000  # ESP32側の受信サンプルレート


class LocalTTS:
    def __init__(self) -> None:
        logger.info(f"TTSモデル読み込み中: {settings.piper_model_path}")
        try:
            from piper.voice import PiperVoice

            self.voice = PiperVoice.load(settings.piper_model_path)
            self.model_sample_rate = self.voice.config.sample_rate
            logger.info(
                f"TTSモデル読み込み完了 (サンプルレート: {self.model_sample_rate}Hz)"
            )
        except Exception as e:
            logger.error(f"TTSモデル読み込み失敗: {e}")
            logger.warning("TTS無効: モデルファイルを確認してください")
            self.voice = None
            self.model_sample_rate = TARGET_SAMPLE_RATE

    def synthesize_chunks(self, text: str) -> Iterator[bytes]:
        if not text.strip() or self.voice is None:
            return

        try:
            for audio_bytes in self.voice.synthesize_stream_raw(text):
                if not audio_bytes:
                    continue

                # モデルのネイティブサンプルレートが16kHzと異なる場合はリサンプリング
                if self.model_sample_rate != TARGET_SAMPLE_RATE:
                    audio_bytes = self._resample(
                        audio_bytes,
                        self.model_sample_rate,
                        TARGET_SAMPLE_RATE,
                    )

                yield audio_bytes
        except Exception as e:
            logger.error(f"TTS合成エラー: {e}")

    def _resample(
        self, audio_bytes: bytes, from_rate: int, to_rate: int
    ) -> bytes:
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        resampled = scipy_signal.resample_poly(audio_int16, to_rate, from_rate)
        return resampled.astype(np.int16).tobytes()
