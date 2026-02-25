from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM (Ollama)
    llm_model: str = "glm-5:cloud"
    ollama_base_url: str = "http://localhost:11434/v1"

    # ASR (mlx-audio Qwen3-ASR)
    asr_model: str = "mlx-community/Qwen3-ASR-0.6B-8bit"

    # TTS (mlx-audio Qwen3-TTS CustomVoice)
    tts_model: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
    tts_voice: str = "ono_anna"

    # システムプロンプト
    system_prompt: str = (
        "あなたは役立つ日本語アシスタントです。簡潔に、2〜3文で答えてください。"
    )

    # サーバー設定
    host: str = "0.0.0.0"
    port: int = 8765


settings = Settings()
