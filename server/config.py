from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM (Ollama)
    llm_model: str = "granite4:micro-h"
    ollama_base_url: str = "http://localhost:11434/v1"

    # ASR (faster-whisper)
    whisper_model_size: str = "base"

    # TTS (piper-tts)
    piper_model_path: str = "models/tts.onnx"

    # システムプロンプト
    system_prompt: str = (
        "あなたは役立つ日本語アシスタントです。簡潔に、2〜3文で答えてください。"
    )

    # サーバー設定
    host: str = "0.0.0.0"
    port: int = 8765


settings = Settings()
