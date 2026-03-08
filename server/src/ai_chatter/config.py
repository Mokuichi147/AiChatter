import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_chatter._paths import SERVER_ROOT

logger = logging.getLogger(__name__)

# デフォルトモデル (macOS / mlx-audio)
DEFAULT_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit"
DEFAULT_VOICE_DESIGN_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"
DEFAULT_ASR_MODEL = "mlx-community/Qwen3-ASR-0.6B-8bit"

# デフォルトモデル (非macOS / qwen-tts, qwen-asr)
DEFAULT_TTS_MODEL_CPU = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_VOICE_DESIGN_MODEL_CPU = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_ASR_MODEL_CPU = "Qwen/Qwen3-ASR-0.6B"


@dataclass
class VoiceConfig:
    type: str = "description"
    # description用
    description: str = "可愛らしい女性の声。高めのトーンで、明るく弾むような話し方。"
    sample_text: str = "こんにちは、今日はいい天気ですね。"
    # reference用
    wav_file: str = ""
    transcript: str = ""


@dataclass
class PersonaConfig:
    name: str = ""
    system_prompt: str = "あなたは役立つ日本語アシスタントです。簡潔に、2〜3文で答えてください。"


@dataclass
class CharacterConfig:
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)


def load_character(yaml_path: str) -> CharacterConfig:
    """YAMLファイルからキャラクター設定を読み込む。"""
    path = Path(yaml_path)
    if not path.is_absolute():
        path = SERVER_ROOT / path

    if not path.exists():
        logger.warning(f"キャラクター設定ファイルが見つかりません: {path}")
        return CharacterConfig()

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    persona_data = data.get("persona", {})
    voice_data = data.get("voice", {})

    persona = PersonaConfig(
        name=persona_data.get("name", ""),
        system_prompt=persona_data.get("system_prompt", PersonaConfig.system_prompt),
    )

    voice = VoiceConfig(
        type=voice_data.get("type", "description"),
        description=voice_data.get("description", VoiceConfig.description),
        sample_text=voice_data.get("sample_text", VoiceConfig.sample_text),
        wav_file=voice_data.get("wav_file", ""),
        transcript=voice_data.get("transcript", ""),
    )

    config = CharacterConfig(persona=persona, voice=voice)
    logger.info(f"キャラクター設定読み込み完了: {persona.name} (声タイプ: {voice.type})")
    return config


@dataclass
class LlmSubConfig:
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    reasoning: str = ""


@dataclass
class LlmEmbeddingsConfig:
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    dimensions: int = 0
    bm25_weight: float = 0.4
    embedding_weight: float = 0.3
    rerank_weight: float = 0.3


@dataclass
class LlmRerankConfig:
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    top_n: int = 20


@dataclass
class LlmConfig:
    model: str = "gpt-4o"
    base_url: str = ""
    api_key: str = ""
    reasoning: str = ""
    sub: LlmSubConfig = field(default_factory=LlmSubConfig)
    embeddings: LlmEmbeddingsConfig = field(default_factory=LlmEmbeddingsConfig)
    rerank: LlmRerankConfig = field(default_factory=LlmRerankConfig)


@dataclass
class TtsConfig:
    model: str = ""
    voice_design_model: str = ""

    def get_model(self) -> str:
        import sys
        default = DEFAULT_TTS_MODEL if sys.platform == "darwin" else DEFAULT_TTS_MODEL_CPU
        return self.model or default

    def get_voice_design_model(self) -> str:
        import sys
        default = DEFAULT_VOICE_DESIGN_MODEL if sys.platform == "darwin" else DEFAULT_VOICE_DESIGN_MODEL_CPU
        return self.voice_design_model or default


@dataclass
class AsrConfig:
    model: str = ""

    def get_model(self) -> str:
        import sys
        default = DEFAULT_ASR_MODEL if sys.platform == "darwin" else DEFAULT_ASR_MODEL_CPU
        return self.model or default


@dataclass
class ModelConfig:
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)


def load_model(yaml_path: str) -> ModelConfig:
    """YAMLファイルからLLM・TTS・ASR設定を読み込む。"""
    path = Path(yaml_path)
    if not path.is_absolute():
        path = SERVER_ROOT / path

    if not path.exists():
        logger.warning(f"モデル設定ファイルが見つかりません: {path}")
        return ModelConfig()

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    def _to_int(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _to_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # LLM
    sub_data = data.get("sub", {})
    sub = LlmSubConfig(
        model=sub_data.get("model", ""),
        base_url=sub_data.get("base_url", ""),
        api_key=sub_data.get("api_key", ""),
        reasoning=sub_data.get("reasoning", ""),
    )
    embeddings_data = data.get("embeddings", {})
    embeddings = LlmEmbeddingsConfig(
        model=embeddings_data.get("model", ""),
        base_url=embeddings_data.get("base_url", ""),
        api_key=embeddings_data.get("api_key", ""),
        dimensions=_to_int(embeddings_data.get("dimensions", 0), 0),
        bm25_weight=_to_float(embeddings_data.get("bm25_weight", 0.4), 0.4),
        embedding_weight=_to_float(embeddings_data.get("embedding_weight", 0.3), 0.3),
        rerank_weight=_to_float(embeddings_data.get("rerank_weight", 0.3), 0.3),
    )
    rerank_data = data.get("rerank", {})
    rerank = LlmRerankConfig(
        model=rerank_data.get("model", ""),
        base_url=rerank_data.get("base_url", ""),
        api_key=rerank_data.get("api_key", ""),
        top_n=_to_int(rerank_data.get("top_n", 20), 20),
    )
    llm = LlmConfig(
        model=data.get("model", LlmConfig.model),
        base_url=data.get("base_url", ""),
        api_key=data.get("api_key", ""),
        reasoning=data.get("reasoning", ""),
        sub=sub,
        embeddings=embeddings,
        rerank=rerank,
    )
    logger.info(f"LLM設定読み込み完了: {llm.model} (base_url: {llm.base_url or '(default)'})")

    # TTS
    tts_data = data.get("tts", {})
    tts = TtsConfig(
        model=tts_data.get("model", ""),
        voice_design_model=tts_data.get("voice_design_model", ""),
    )
    logger.info(f"TTS設定読み込み完了: {tts.get_model()}")

    # ASR
    asr_data = data.get("asr", {})
    asr = AsrConfig(
        model=asr_data.get("model", ""),
    )
    logger.info(f"ASR設定読み込み完了: {asr.get_model()}")

    return ModelConfig(llm=llm, tts=tts, asr=asr)


@dataclass
class SkillEntry:
    match: str = ""
    guide: str = ""


@dataclass
class SkillsConfig:
    memory_top_k: int = 3
    tool_skill_top_k: int = 5
    tools: list[SkillEntry] = field(default_factory=list)


@dataclass
class PromptConfig:
    output_rules: str = ""
    tool_guide_base: str = ""
    group_rules: str = ""
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    subagent_system_prompt: str = ""


def load_prompt(yaml_path: str) -> PromptConfig:
    """YAMLファイルからプロンプト設定を読み込む。"""
    path = Path(yaml_path)
    if not path.is_absolute():
        path = SERVER_ROOT / path

    if not path.exists():
        logger.warning(f"プロンプト設定ファイルが見つかりません: {path}")
        return PromptConfig()

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    skills_data = data.get("skills", {})
    tool_entries = []
    for entry in skills_data.get("tools", []):
        if isinstance(entry, dict):
            tool_entries.append(SkillEntry(
                match=entry.get("match", ""),
                guide=entry.get("guide", ""),
            ))
    try:
        memory_top_k = int(skills_data.get("memory_top_k", 3))
    except (TypeError, ValueError):
        memory_top_k = 3
    try:
        tool_skill_top_k = int(skills_data.get("tool_skill_top_k", 5))
    except (TypeError, ValueError):
        tool_skill_top_k = 5
    skills = SkillsConfig(
        memory_top_k=memory_top_k,
        tool_skill_top_k=tool_skill_top_k,
        tools=tool_entries,
    )

    config = PromptConfig(
        output_rules=data.get("output_rules", ""),
        tool_guide_base=data.get("tool_guide_base", ""),
        group_rules=data.get("group_rules", ""),
        skills=skills,
        subagent_system_prompt=data.get("subagent_system_prompt", ""),
    )
    logger.info(f"プロンプト設定読み込み完了: {path}")
    return config


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(SERVER_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # キャラクター設定ファイル
    character_file: str = "configs/character.yaml"
    # REST/CLI向けキャラクターカタログ設定
    character_dir: str = "configs"
    character_glob: str = "character*.yaml"

    # プロンプト設定ファイル
    prompt_file: str = "configs/prompt.yaml"

    # ツール設定
    tools_enabled: bool = True
    memory_file: str = "data/memory.json"
    tavily_api_key: str = ""
    subagent_num_ctx: int = 128000
    subagent_enabled: bool = True
    subagent_max_rounds: int = 8
    subagent_timeout_sec: int = 45
    subagent_mcp_tool_denylist: str = ""
    subagent_result_max_chars: int = 4000

    # 会話履歴の永続化
    history_file: str = "data/history.json"
    history_restore_count: int = 3
    default_history_mode: str = "shared"
    chat_max_history_messages: int = 20

    # 通知の永続化
    notification_file: str = "data/notifications.json"

    # 会話モード
    conversation_mode: str = "solo"  # "solo" or "group"
    speaker_similarity_threshold: float = 0.65

    # サーバー設定
    host: str = "0.0.0.0"
    port: int = 8765


settings = Settings()
character = load_character(settings.character_file)
prompt_config = load_prompt(settings.prompt_file)
_model_config = load_model("configs/model.yaml")

# 後方互換エイリアス
llm_config = _model_config.llm
tts_config = _model_config.tts
asr_config = _model_config.asr


def character_data_path(filename: str) -> str:
    """キャラクター名ベースのデータファイルパスを返す。"""
    name = character.persona.name
    if name:
        return f"data/{name}/{filename}"
    return f"data/{filename}"
