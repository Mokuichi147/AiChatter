import logging
import wave
from typing import Iterator

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000  # ESP32側の受信サンプルレート
VOLUME_SCALE = 8192  # 音量スケール (25% = PCM精度と音割れのバランス)
_DEBUG_DUMP = True  # TTS出力をファイルに保存してデバッグ
_debug_counter = 0


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

    def _to_pcm(self, audio_mx) -> bytes:
        """mx.array音声をリサンプリング+PCM変換"""
        audio_np = np.array(audio_mx, dtype=np.float32)

        # リサンプリング (モデル出力 → 16kHz)
        if self.model_sample_rate != TARGET_SAMPLE_RATE:
            audio_np = self._resample(
                audio_np, TARGET_SAMPLE_RATE, self.model_sample_rate
            ).astype(np.float32)

        # クリッピング
        audio_np = np.clip(audio_np, -1.0, 1.0)
        return (audio_np * VOLUME_SCALE).astype(np.int16).tobytes()

    def synthesize_chunks(self, text: str) -> Iterator[bytes]:
        """バッチTTS: 一括生成してチャンクを返す"""
        if not text.strip():
            return

        try:
            all_pcm = bytearray()
            for result in self.model.generate(
                text=text,
                voice=settings.tts_voice,
                lang_code="Japanese",
            ):
                if result.audio is None:
                    continue
                pcm = self._to_pcm(result.audio)
                if len(pcm) > 0:
                    all_pcm.extend(pcm)
                    yield pcm

            # デバッグ: PCMデータをWAVファイルに保存
            if _DEBUG_DUMP and all_pcm:
                global _debug_counter
                _debug_counter += 1
                path = f"/tmp/tts_debug_{_debug_counter}.wav"
                with wave.open(path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(TARGET_SAMPLE_RATE)
                    wf.writeframes(bytes(all_pcm))
                logger.info(f"デバッグ: TTS出力保存 → {path} ({len(all_pcm)} bytes)")
        except Exception as e:
            logger.error(f"TTS合成エラー: {e}", exc_info=True)
