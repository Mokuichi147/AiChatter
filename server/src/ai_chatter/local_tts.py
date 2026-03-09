import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
from sudachipy import Dictionary, SplitMode

from ai_chatter import config
from ai_chatter._paths import SERVER_ROOT
from ai_chatter.config import tts_config

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
    # 漢字が含まれなければ形態素解析をスキップ
    if not _KANJI_RE.search(text):
        return text
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

        if sys.platform == "darwin":
            self._init_mlx()
        else:
            self._init_qwen_tts()

    # ------------------------------------------------------------------ macOS

    def _init_mlx(self) -> None:
        voice_config = config.character.voice
        self.voice_config = voice_config

        ref_wav, ref_text = self._prepare_reference_mlx(voice_config)
        self.ref_audio = ref_wav
        self.ref_text = ref_text

        tts_model = tts_config.get_model()
        logger.info(f"TTSモデル読み込み中 (mlx-audio): {tts_model}")
        from mlx_audio.tts.utils import load_model

        self._model = load_model(tts_model)
        self._model_sample_rate = getattr(self._model, "sample_rate", 24000)
        self._backend = "mlx"

        logger.info(
            f"TTSモデル読み込み完了 (参照音声: {self.ref_audio}, "
            f"サンプルレート: {self._model_sample_rate}Hz)"
        )

    def _prepare_reference_mlx(self, voice_config) -> tuple[str, str]:
        """macOS: 声タイプに応じて参照音声を準備する。"""
        if voice_config.type == "reference":
            return self._resolve_reference_wav(voice_config)
        if voice_config.type == "description":
            return self._generate_reference_voice_mlx(voice_config)
        raise ValueError(f"未対応の声タイプ: {voice_config.type}")

    def _generate_reference_voice_mlx(self, voice_config) -> tuple[str, str]:
        """mlx-audioのVoiceDesignモデルで参照音声を生成してWAVに保存する。"""
        import soundfile as sf

        description = voice_config.description
        sample_text = voice_config.sample_text

        cache_key = hashlib.md5(
            f"{description}:{sample_text}".encode()
        ).hexdigest()[:12]
        VOICES_DIR.mkdir(exist_ok=True)
        cache_path = VOICES_DIR / f"generated_{cache_key}.wav"

        if cache_path.exists():
            logger.info(f"キャッシュ済み参照音声を使用: {cache_path}")
            return str(cache_path), sample_text

        design_model_name = tts_config.get_voice_design_model()
        logger.info(
            f"VoiceDesignモデルで参照音声を生成中 (mlx-audio): {design_model_name} "
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
        del design_model

        return str(cache_path), sample_text

    # -------------------------------------------------------------- 非macOS

    def _init_qwen_tts(self) -> None:
        import torch
        from transformers import logging as transformers_logging
        from qwen_tts import Qwen3TTSModel

        transformers_logging.set_verbosity_error()

        voice_config = config.character.voice
        self.voice_config = voice_config

        ref_wav, ref_text = self._prepare_reference_cpu(voice_config)
        self.ref_audio = ref_wav
        self.ref_text = ref_text

        tts_model = tts_config.get_model()
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            device_info = f"CUDA ({gpu_name})"
        else:
            device_info = "CPU"
        logger.info(f"TTSモデル読み込み中 (qwen-tts / {device_info}): {tts_model}")

        dtype = torch.bfloat16 if cuda_available else torch.float32
        device_map = "cuda:0" if cuda_available else "cpu"

        try:
            import flash_attn  # noqa: F401
            attn_implementation = "flash_attention_2"
            logger.info("Flash Attention 2 を使用")
        except ImportError:
            attn_implementation = "sdpa"

        self._model = Qwen3TTSModel.from_pretrained(
            tts_model,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        self._model_sample_rate = 24000  # Qwen3-TTSの出力サンプルレート
        self._backend = "qwen_tts"

        # 参照音声プロンプトを事前計算してキャッシュ（毎回の再エンコードを回避）
        logger.info("声クローン用プロンプトをキャッシュ中...")
        self._voice_prompt = self._model.create_voice_clone_prompt(
            ref_audio=self.ref_audio,
            ref_text=self.ref_text,
        )

        logger.info(
            f"TTSモデル読み込み完了 ({device_info}, 参照音声: {self.ref_audio}, "
            f"サンプルレート: {self._model_sample_rate}Hz)"
        )

    def _prepare_reference_cpu(self, voice_config) -> tuple[str, str]:
        """非macOS: 声タイプに応じて参照音声を準備する。"""
        if voice_config.type == "reference":
            return self._resolve_reference_wav(voice_config)
        if voice_config.type == "description":
            return self._generate_reference_voice_cpu(voice_config)
        raise ValueError(f"未対応の声タイプ: {voice_config.type}")

    def _generate_reference_voice_cpu(self, voice_config) -> tuple[str, str]:
        """qwen-ttsのVoiceDesignモデルで参照音声を生成してWAVに保存する。"""
        import soundfile as sf
        import torch
        from qwen_tts import Qwen3TTSModel

        description = voice_config.description
        sample_text = voice_config.sample_text

        cache_key = hashlib.md5(
            f"{description}:{sample_text}".encode()
        ).hexdigest()[:12]
        VOICES_DIR.mkdir(exist_ok=True)
        cache_path = VOICES_DIR / f"generated_cpu_{cache_key}.wav"

        if cache_path.exists():
            logger.info(f"キャッシュ済み参照音声を使用: {cache_path}")
            return str(cache_path), sample_text

        design_model_name = tts_config.get_voice_design_model()
        logger.info(
            f"VoiceDesignモデルで参照音声を生成中 (qwen-tts): {design_model_name} "
            f"(説明: {description[:50]}...)"
        )

        from transformers import logging as transformers_logging
        transformers_logging.set_verbosity_error()

        cuda_available = torch.cuda.is_available()
        dtype = torch.bfloat16 if cuda_available else torch.float32
        device_map = "cuda:0" if cuda_available else "cpu"

        try:
            import flash_attn  # noqa: F401
            attn_implementation = "flash_attention_2"
        except ImportError:
            attn_implementation = "sdpa"

        design_model = Qwen3TTSModel.from_pretrained(
            design_model_name,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        design_sr = 24000

        wavs, design_sr = design_model.generate_voice_design(
            text=sample_text,
            instruct=description,
            language="Japanese",
        )

        if not wavs:
            raise RuntimeError("VoiceDesignモデルが音声を生成できませんでした")

        audio = np.array(wavs[0], dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.flatten()
        audio = np.clip(audio, -1.0, 1.0)

        sf.write(str(cache_path), audio, design_sr)
        logger.info(f"参照音声を保存しました: {cache_path}")
        del design_model

        return str(cache_path), sample_text

    # ------------------------------------------------------------------ 共通

    @staticmethod
    def _resolve_reference_wav(voice_config) -> tuple[str, str]:
        """referenceタイプのWAVファイルパスとトランスクリプトを返す。"""
        wav_path = Path(voice_config.wav_file)
        if not wav_path.is_absolute():
            wav_path = SERVER_ROOT / wav_path
        if not wav_path.exists():
            raise FileNotFoundError(f"参照音声ファイルが見つかりません: {wav_path}")
        return str(wav_path), voice_config.transcript

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

    def prepare_text(self, text: str) -> str | None:
        """TTS前処理: 不合成文字除去・漢字読み変換（CPU処理、GPUロック不要）。"""
        text = _SPEAKABLE_RE.sub("", text).strip()
        if not text:
            return None
        text = _to_reading(text)
        if not text:
            return None
        if not re.search(
            r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF"
            r"a-zA-Zａ-ｚＡ-Ｚ0-9０-９]",
            text,
        ):
            return None
        return text

    def synthesize_raw(self, text: str) -> list[np.ndarray] | None:
        """TTS推論: 前処理済みテキストから生オーディオセグメントを生成（GPU処理）。"""
        if self._backend == "mlx":
            return self._generate_raw_mlx(text)
        else:
            return self._generate_raw_qwen_tts(text)

    def postprocess_audio(self, segments: list[np.ndarray]) -> bytes | None:
        """後処理: リサンプル・正規化・PCM変換（CPU処理、GPUロック不要）。"""
        try:
            audio_np = np.concatenate(segments)

            if self._model_sample_rate != TARGET_SAMPLE_RATE:
                audio_np = self._resample(
                    audio_np, TARGET_SAMPLE_RATE, self._model_sample_rate
                ).astype(np.float32)

            peak = np.max(np.abs(audio_np))
            if peak > 0:
                audio_np = audio_np / peak

            pcm = (audio_np * self.volume_scale).astype(np.int16).tobytes()
            return pcm if pcm else None
        except Exception as e:
            logger.error(f"TTS後処理エラー: {e}", exc_info=True)
            return None

    def synthesize_chunks(self, text: str) -> Iterator[bytes]:
        """バッチTTS: 前処理→推論→後処理を一括実行（後方互換用）。"""
        prepared = self.prepare_text(text)
        if not prepared:
            return
        raw = self.synthesize_raw(prepared)
        if not raw:
            return
        pcm = self.postprocess_audio(raw)
        if pcm:
            yield pcm

    def _generate_raw_mlx(self, text: str) -> list[np.ndarray] | None:
        try:
            segments = []
            for result in self._model.generate(
                text=text,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
                lang_code="Japanese",
            ):
                if result.audio is None:
                    continue
                segments.append(np.array(result.audio, dtype=np.float32))
            return segments if segments else None
        except Exception as e:
            logger.error(f"TTS合成エラー (mlx): {e}", exc_info=True)
            return None

    def _generate_raw_qwen_tts(self, text: str) -> list[np.ndarray] | None:
        try:
            wavs, sr = self._model.generate_voice_clone(
                text=text,
                language="Japanese",
                voice_clone_prompt=self._voice_prompt,
            )
            if not wavs:
                return None
            audio_np = np.array(wavs[0], dtype=np.float32)
            if audio_np.ndim > 1:
                audio_np = audio_np.flatten()
            return [audio_np] if audio_np.size > 0 else None
        except Exception as e:
            logger.error(f"TTS合成エラー (qwen-tts): {e}", exc_info=True)
            return None
