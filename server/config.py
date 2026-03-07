import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# デフォルトモデル
DEFAULT_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit"
DEFAULT_VOICE_DESIGN_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"


@dataclass
class VoiceConfig:
    type: str = "description"
    # description用
    description: str = "可愛らしい女性の声。高めのトーンで、明るく弾むような話し方。"
    sample_text: str = "こんにちは、今日はいい天気ですね。"
    voice_design_model: str = ""
    # reference用
    wav_file: str = ""
    transcript: str = ""
    # 音声合成用Baseモデル
    tts_model: str = ""

    def get_tts_model(self) -> str:
        """音声合成用Baseモデルを返す。"""
        return self.tts_model or DEFAULT_TTS_MODEL

    def get_voice_design_model(self) -> str:
        """参照音声生成用VoiceDesignモデルを返す。"""
        return self.voice_design_model or DEFAULT_VOICE_DESIGN_MODEL


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
        path = Path(__file__).parent / path

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
        voice_design_model=voice_data.get("voice_design_model", ""),
        wav_file=voice_data.get("wav_file", ""),
        transcript=voice_data.get("transcript", ""),
        tts_model=voice_data.get("tts_model", ""),
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


def load_llm(yaml_path: str) -> LlmConfig:
    """YAMLファイルからLLM設定を読み込む。"""
    path = Path(yaml_path)
    if not path.is_absolute():
        path = Path(__file__).parent / path

    if not path.exists():
        logger.warning(f"LLM設定ファイルが見つかりません: {path}")
        return LlmConfig()

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
        embedding_weight=_to_float(
            embeddings_data.get("embedding_weight", 0.3), 0.3
        ),
        rerank_weight=_to_float(
            embeddings_data.get("rerank_weight", 0.3), 0.3
        ),
    )
    rerank_data = data.get("rerank", {})
    rerank = LlmRerankConfig(
        model=rerank_data.get("model", ""),
        base_url=rerank_data.get("base_url", ""),
        api_key=rerank_data.get("api_key", ""),
        top_n=_to_int(rerank_data.get("top_n", 20), 20),
    )

    config = LlmConfig(
        model=data.get("model", LlmConfig.model),
        base_url=data.get("base_url", ""),
        api_key=data.get("api_key", ""),
        reasoning=data.get("reasoning", ""),
        sub=sub,
        embeddings=embeddings,
        rerank=rerank,
    )
    logger.info(f"LLM設定読み込み完了: {config.model} (base_url: {config.base_url or '(default)'})")
    return config


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
        path = Path(__file__).parent / path

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
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ASR (mlx-audio Qwen3-ASR)
    asr_model: str = "mlx-community/Qwen3-ASR-0.6B-8bit"

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
llm_config = load_llm("configs/llm.yaml")


def character_data_path(filename: str) -> str:
    """キャラクター名ベースのデータファイルパスを返す。"""
    name = character.persona.name
    if name:
        return f"data/{name}/{filename}"
    return f"data/{filename}"
