import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Iterator

import numpy as np
from sudachipy import Dictionary, SplitMode

from ai_chatter import config
from ai_chatter._paths import SERVER_ROOT

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000  # ESP32側の受信サンプルレート
DEFAULT_VOLUME_SCALE = 3275  # 音量スケール (level 5相当)


def _level_to_scale(level: int) -> int:
    """音量レベル(1-10)をスケール値に変換する。"""
    return int(655 * max(1, min(10, level)))


VOICES_DIR = SERVER_ROOT / "voices"

# TTS合成可能な文字のみ残す (日本語・英数字・句読点・記号)
_SPEAKABLE_RE = re.compile(
    r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF"
    r"a-zA-Zａ-ｚＡ-Ｚ0-9０-９"
    r"、。！？,.!?ー〜…―\-\s]"
)


_sudachi_tokenizer = Dictionary().create()
_KANJI_RE = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")


def _to_reading(text: str) -> str:
    """漢字を含むトークンをカタカナの読みに変換する。"""
    morphemes = _sudachi_tokenizer.tokenize(text, SplitMode.C)
    parts: list[str] = []
    for m in morphemes:
        surface = m.surface()
        if _KANJI_RE.search(surface):
            reading = m.reading_form()
            parts.append(reading if reading else surface)
        else:
            parts.append(surface)
    return "".join(parts)


class LocalTTS:
    def __init__(self) -> None:
        from scipy import signal as _signal

        self._resample = _signal.resample_poly
        self.volume_scale: int = self._load_volume_scale()

        voice_config = config.character.voice
        self.voice_config = voice_config

        # 参照音声を準備（description→生成、reference→ユーザー提供）
        ref_wav, ref_text = self._prepare_reference(voice_config)
        self.ref_audio = ref_wav
        self.ref_text = ref_text

        # 音声合成用Baseモデルを読み込み
        tts_model = voice_config.get_tts_model()
        logger.info(f"TTSモデル読み込み中: {tts_model}")
        from mlx_audio.tts.utils import load_model

        self.model = load_model(tts_model)
        self.model_sample_rate = getattr(self.model, "sample_rate", 24000)

        logger.info(
            f"TTSモデル読み込み完了 (参照音声: {self.ref_audio}, "
            f"サンプルレート: {self.model_sample_rate}Hz)"
        )

    @staticmethod
    def _load_volume_scale() -> int:
        """settings.jsonから音量レベルを読み込みスケール値を返す。"""
        try:
            settings_path = Path(config.character_data_path("settings.json"))
            if not settings_path.is_absolute():
                settings_path = SERVER_ROOT / settings_path
            if settings_path.exists():
                data = json.loads(settings_path.read_text(encoding="utf-8"))
                level = data.get("volume_level")
                if isinstance(level, int):
                    scale = _level_to_scale(level)
                    logger.info(f"音量設定を復元: level={level}, scale={scale}")
                    return scale
        except Exception as e:
            logger.warning(f"音量設定の読み込みに失敗: {e}")
        return DEFAULT_VOLUME_SCALE

    def _prepare_reference(self, voice_config) -> tuple[str, str]:
        """声タイプに応じて参照音声ファイルのパスとトランスクリプトを返す。"""
        if voice_config.type == "reference":
            wav_path = Path(voice_config.wav_file)
            if not wav_path.is_absolute():
                wav_path = SERVER_ROOT / wav_path
            if not wav_path.exists():
                raise FileNotFoundError(f"参照音声ファイルが見つかりません: {wav_path}")
            return str(wav_path), voice_config.transcript

        if voice_config.type == "description":
            return self._generate_reference_voice(voice_config)

        raise ValueError(f"未対応の声タイプ: {voice_config.type}")

    def _generate_reference_voice(self, voice_config) -> tuple[str, str]:
        """VoiceDesignモデルで参照音声を生成してWAVファイルに保存する。"""
        import soundfile as sf

        description = voice_config.description
        sample_text = voice_config.sample_text

        # 説明文のハッシュでキャッシュ（同じ設定なら再生成しない）
        cache_key = hashlib.md5(
            f"{description}:{sample_text}".encode()
        ).hexdigest()[:12]
        VOICES_DIR.mkdir(exist_ok=True)
        cache_path = VOICES_DIR / f"generated_{cache_key}.wav"

        if cache_path.exists():
            logger.info(f"キャッシュ済み参照音声を使用: {cache_path}")
            return str(cache_path), sample_text

        # VoiceDesignモデルで参照音声を生成
        design_model_name = voice_config.get_voice_design_model()
        logger.info(
            f"VoiceDesignモデルで参照音声を生成中: {design_model_name} "
            f"(説明: {description[:50]}...)"
        )
        from mlx_audio.tts.utils import load_model

        design_model = load_model(design_model_name)
        design_sr = getattr(design_model, "sample_rate", 24000)

        segments = []
        for result in design_model.generate(
            text=sample_text,
            instruct=description,
            lang_code="Japanese",
        ):
            if result.audio is not None:
                segments.append(np.array(result.audio, dtype=np.float32))

        if not segments:
            raise RuntimeError("VoiceDesignモデルが音声を生成できませんでした")

        audio = np.concatenate(segments)
        audio = np.clip(audio, -1.0, 1.0)

        sf.write(str(cache_path), audio, design_sr)
        logger.info(f"参照音声を保存しました: {cache_path}")

        # VoiceDesignモデルを解放
        del design_model

        return str(cache_path), sample_text

    def synthesize_chunks(self, text: str) -> Iterator[bytes]:
        """バッチTTS: 全音声を結合→一括リサンプル→PCM変換"""
        # 絵文字等TTS不可文字を除去
        text = _SPEAKABLE_RE.sub("", text).strip()
        if not text:
            return
        # 漢字を読みに変換（読み間違い防止）
        text = _to_reading(text)
        if not text:
            return
        # 句読点・記号のみで発音可能文字がない場合はスキップ
        if not re.search(
            r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF"
            r"a-zA-Zａ-ｚＡ-Ｚ0-9０-９]",
            text,
        ):
            return

        try:
            # Baseモデルで声クローンTTS
            segments = []
            for result in self.model.generate(
                text=text,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
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

            # ピーク正規化 (モデル/キャラクターによる音量差を吸収)
            peak = np.max(np.abs(audio_np))
            if peak > 0:
                audio_np = audio_np / peak

            pcm = (audio_np * self.volume_scale).astype(np.int16).tobytes()

            if not pcm:
                return

            yield pcm

        except Exception as e:
            logger.error(f"TTS合成エラー: {e}", exc_info=True)
