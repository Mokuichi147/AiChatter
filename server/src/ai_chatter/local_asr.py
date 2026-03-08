import logging
import os
import sys
import tempfile

import numpy as np

from ai_chatter.config import asr_config

logger = logging.getLogger(__name__)


class LocalASR:
    def __init__(self) -> None:
        if sys.platform == "darwin":
            self._init_mlx()
        else:
            self._init_qwen_asr()

    def _init_mlx(self) -> None:
        model_name = asr_config.get_model()
        logger.info(f"ASRモデル読み込み中 (mlx-audio): {model_name}")
        from mlx_audio.stt import load

        self._model = load(model_name)
        self._backend = "mlx"
        logger.info("ASRモデル読み込み完了")

    def _init_qwen_asr(self) -> None:
        import torch
        from transformers import logging as transformers_logging
        from qwen_asr import Qwen3ASRModel

        transformers_logging.set_verbosity_error()

        model_name = asr_config.get_model()
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            device_info = f"CUDA ({gpu_name})"
        else:
            device_info = "CPU"
        logger.info(f"ASRモデル読み込み中 (qwen-asr / {device_info}): {model_name}")

        dtype = torch.bfloat16 if cuda_available else torch.float32
        device_map = "cuda:0" if cuda_available else "cpu"
        self._model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            max_new_tokens=256,
        )
        self._backend = "qwen_asr"
        logger.info(f"ASRモデル読み込み完了 ({device_info})")

    def transcribe(self, pcm_bytes: bytes, language: str = "Japanese") -> str:
        if not pcm_bytes:
            return ""

        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio) < 1600:  # 0.1秒未満は無視
            return ""

        if self._backend == "mlx":
            text = self._transcribe_mlx(audio, language)
        else:
            text = self._transcribe_qwen_asr(audio, language)

        if text in ("。", "、", ".", ",", "ええ。", "ええ。.", "日本語", "日本語."):
            return ""
        logger.info(f"ASR結果: '{text}'")
        return text

    def _transcribe_mlx(self, audio: np.ndarray, language: str) -> str:
        result = self._model.generate(audio, language=language)
        return result.text.strip() if result and result.text else ""

    def _transcribe_qwen_asr(self, audio: np.ndarray, language: str) -> str:
        import soundfile as sf

        # qwen-asr はファイルパスを受け取るため一時ファイルに書き出す
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, audio, 16000)
            results = self._model.transcribe(audio=tmp_path, language=language)
            if not results or not results[0].text:
                return ""
            return results[0].text.strip()
        finally:
            os.unlink(tmp_path)
