import logging
from typing import Iterator

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000  # ESP32側の受信サンプルレート
VOLUME_SCALE = 8192  # 音量スケール (25% = PCM精度と音割れのバランス)


class LocalTTS:
    def __init__(self) -> None:
        logger.info(f"TTSモデル読み込み中: {settings.tts_model}")
        from mlx_audio.tts.utils import load_model

        self.model = load_model(settings.tts_model)
        self.model_sample_rate = getattr(self.model, "sample_rate", 24000)

        # scipy resampling用にインポートしておく
        from scipy import signal as _signal

        self._resample = _signal.resample_poly

        # スピーカー確認
        speakers = self.model.get_supported_speakers()
        logger.info(f"利用可能スピーカー: {speakers}")
        logger.info(
            f"TTSモデル読み込み完了 (スピーカー: {settings.tts_voice}, "
            f"サンプルレート: {self.model_sample_rate}Hz)"
        )

    def synthesize_chunks(self, text: str) -> Iterator[bytes]:
        """バッチTTS: 全音声を結合→一括リサンプル→PCM変換"""
        if not text.strip():
            return

        try:
            # モデルから全音声セグメントを収集
            segments = []
            for result in self.model.generate(
                text=text,
                voice=settings.tts_voice,
                lang_code="Japanese",
            ):
                if result.audio is None:
                    continue
                segments.append(np.array(result.audio, dtype=np.float32))

            if not segments:
                return

            # 全セグメントを結合して一括リサンプル (境界アーティファクト防止)
            audio_np = np.concatenate(segments)

            if self.model_sample_rate != TARGET_SAMPLE_RATE:
                audio_np = self._resample(
                    audio_np, TARGET_SAMPLE_RATE, self.model_sample_rate
                ).astype(np.float32)

            audio_np = np.clip(audio_np, -1.0, 1.0)
            pcm = (audio_np * VOLUME_SCALE).astype(np.int16).tobytes()

            if not pcm:
                return

            yield pcm

        except Exception as e:
            logger.error(f"TTS合成エラー: {e}", exc_info=True)
