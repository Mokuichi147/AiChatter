import asyncio
import logging
import signal
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from ai_chatter.character_catalog import CharacterCatalog
from ai_chatter.config import settings
from ai_chatter.session_manager import ALLOWED_HISTORY_MODES, SessionManager
from ai_chatter.tool_factory import ToolFactory

logger = logging.getLogger(__name__)

app = typer.Typer(help="AiChatter CLI", invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})

# 共通オプション型
OptCharacter = Annotated[str, typer.Option("-c", "--character", help="キャラクター設定ファイル")]
OptModel = Annotated[str, typer.Option("-m", "--model", help="モデル設定ファイル")]
OptPrompt = Annotated[str, typer.Option("-p", "--prompt", help="プロンプト設定ファイル")]


@app.callback()
def _app_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        raise typer.Exit()


def _build_tools():
    factory = ToolFactory(tts=None, get_pipelines=lambda: [])
    return factory.create_registry(set()), factory.skill_provider


def _setup_logging(*, debug: bool = False, server: bool = False) -> None:
    if server:
        log_level = logging.INFO
    elif debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _apply_cli_overrides(character: str, model: str, prompt: str) -> None:
    """-c / -m / -p の値を解釈して設定を再読み込みする。"""
    from ai_chatter import config as _config
    from ai_chatter.config import (
        _resolve_config_path,
        load_character,
        load_model,
        load_prompt,
    )

    def _resolve(file_path: str) -> str:
        """cwd基準で絶対パスに変換し、example fallbackを適用する。"""
        p = Path(file_path).expanduser()
        if not p.is_absolute():
            p = Path.cwd() / p
        return _resolve_config_path(str(p))

    # -- character --
    resolved = _resolve(character)
    settings.character_file = resolved
    settings.character_dir = str(Path(resolved).parent)
    _config.character = load_character(resolved)
    logger.info("キャラクター設定: %s", resolved)

    # -- model --
    resolved = _resolve(model)
    settings.model_file = resolved
    mc = load_model(resolved)
    _config._model_config = mc
    _config.llm_config = mc.llm
    _config.tts_config = mc.tts
    _config.asr_config = mc.asr
    logger.info("モデル設定: %s", resolved)

    # -- prompt --
    resolved = _resolve(prompt)
    settings.prompt_file = resolved
    _config.prompt_config = load_prompt(resolved)
    logger.info("プロンプト設定: %s", resolved)


def _play_tts(tts, text: str) -> None:
    """TTS合成してスピーカーで再生する。"""
    import numpy as np
    import sounddevice as sd

    _TTS_SAMPLE_RATE = 16000
    _PC_VOLUME_AMPLIFY = 10.0

    for pcm in tts.synthesize_chunks(text):
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        audio = np.clip(audio * _PC_VOLUME_AMPLIFY, -1.0, 1.0)
        sd.play(audio, samplerate=_TTS_SAMPLE_RATE)
        sd.wait()


@app.command()
def chat(
    character: OptCharacter = "configs/character.yaml",
    model: OptModel = "configs/model.yaml",
    prompt: OptPrompt = "configs/prompt.yaml",
    history_mode: Annotated[str, typer.Option("--history-mode", help="会話履歴モード")] = settings.default_history_mode,
    session_id: Annotated[str, typer.Option("--session-id", help="セッションID")] = "cli",
    stream: Annotated[bool, typer.Option("--stream", help="ストリーミング表示")] = False,
    voice: Annotated[bool, typer.Option("--voice", help="返答をTTS音声で再生")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="デバッグログを表示")] = False,
) -> None:
    """対話CLIモード"""
    _setup_logging(debug=debug)
    _apply_cli_overrides(character, model, prompt)
    asyncio.run(_run_chat_async(history_mode, session_id, stream, voice))


async def _run_chat_async(
    history_mode: str,
    session_id: str,
    stream: bool,
    use_voice: bool,
) -> None:
    from ai_chatter.chat_engine import ChatEngine
    from ai_chatter.local_llm import LocalLLM

    catalog = CharacterCatalog(settings.character_dir, settings.character_glob)
    catalog.reload()
    default_name = Path(settings.character_file).name
    if not catalog.has(default_name):
        catalog.register_file(settings.character_file)
    character_id = default_name

    history_mode = (history_mode or settings.default_history_mode).strip().lower()
    if history_mode not in ALLOWED_HISTORY_MODES:
        raise typer.BadParameter(
            f"history-mode は {sorted(ALLOWED_HISTORY_MODES)} のいずれかを指定してください"
        )

    tts = None
    if use_voice:
        from ai_chatter.local_tts import LocalTTS

        tts = LocalTTS()

    llm = LocalLLM()
    tool_registry, skill_provider = _build_tools()
    session_manager = SessionManager(
        default_character_id=character_id,
        default_history_mode=history_mode,
        max_messages=settings.chat_max_history_messages,
    )
    engine = ChatEngine(
        llm=llm,
        session_manager=session_manager,
        character_catalog=catalog,
        tool_registry=tool_registry,
        skill_provider=skill_provider,
    )

    await engine.ensure_session(
        session_id=session_id,
        history_mode=history_mode,
        character_id=character_id,
    )

    loop = asyncio.get_event_loop()
    chosen = catalog.get(character_id)
    chosen_name = chosen.config.persona.name or chosen.file_name
    print(f"\nCLI会話を開始します。session_id={session_id}, character={chosen_name} ({character_id})")
    print("終了するには 'exit' または 'quit' を入力してください。\n")

    signal.signal(signal.SIGINT, signal.default_int_handler)

    from prompt_toolkit import PromptSession
    session = PromptSession()

    try:
        while True:
            try:
                user_text = (await session.prompt_async("you> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                break

            if stream:
                print(f"{chosen_name}> ", end="", flush=True)
                full_response = ""
                async for event in engine.stream_chat(session_id=session_id, text=user_text):
                    if event.get("type") == "chunk":
                        chunk_text = event.get("text", "")
                        full_response += chunk_text
                        print(chunk_text, end="", flush=True)
                    elif event.get("type") == "done":
                        print()
                if tts and full_response:
                    await loop.run_in_executor(None, _play_tts, tts, full_response)
                reply = full_response
            else:
                result = await engine.chat(session_id=session_id, text=user_text)
                reply = result.get("text", "")
                print(f"{chosen_name}> {reply}")
                if tts and reply:
                    await loop.run_in_executor(None, _play_tts, tts, reply)

            if reply:
                from ai_chatter._paths import save_history

                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                save_history([
                    {"role": "user", "content": user_text, "created_at": now_str},
                    {"role": "assistant", "content": reply, "created_at": now_str},
                ])
    except KeyboardInterrupt:
        print()


@app.command()
def voice(
    character: OptCharacter = "configs/character.yaml",
    model: OptModel = "configs/model.yaml",
    prompt: OptPrompt = "configs/prompt.yaml",
    group: Annotated[bool, typer.Option("--group", help="グループモードで起動する")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="デバッグログを表示")] = False,
) -> None:
    """音声対話CLIモード (PCマイク/スピーカー)"""
    _setup_logging(debug=debug)
    _apply_cli_overrides(character, model, prompt)
    asyncio.run(_run_voice_async(group))


async def _run_voice_async(group: bool) -> None:
    from ai_chatter.local_asr import LocalASR
    from ai_chatter.local_llm import LocalLLM
    from ai_chatter.local_tts import LocalTTS
    from ai_chatter.voice_cli import VoiceCLI

    if group:
        settings.conversation_mode = "group"

    logger.info("AIモデルをプリロード中...")
    asr = LocalASR()
    llm = LocalLLM()
    tts = LocalTTS()
    logger.info("AIモデルプリロード完了")

    speaker_id = None
    if settings.conversation_mode == "group":
        from ai_chatter.config import character_data_path
        from ai_chatter.speaker_id import SpeakerIdentifier

        speakers_path = character_data_path("speakers.json")
        speaker_id = SpeakerIdentifier(
            speakers_path,
            similarity_threshold=settings.speaker_similarity_threshold,
        )
        logger.info("話者識別モジュール初期化完了 (グループモード)")

    tool_registry, skill_provider = _build_tools()

    voice_cli = VoiceCLI(
        asr=asr,
        llm=llm,
        tts=tts,
        tool_registry=tool_registry,
        skill_provider=skill_provider,
        speaker_id=speaker_id,
    )
    await voice_cli.run()


@app.command()
def server(
    character: OptCharacter = "configs/character.yaml",
    model: OptModel = "configs/model.yaml",
    prompt: OptPrompt = "configs/prompt.yaml",
    group: Annotated[bool, typer.Option("--group", help="グループモードで起動する")] = False,
) -> None:
    """サーバー起動モード"""
    _setup_logging(server=True)
    _apply_cli_overrides(character, model, prompt)

    if group:
        settings.conversation_mode = "group"

    from ai_chatter.main import ECHO_MODE, app as fastapi_app
    import uvicorn

    mode = "エコー" if ECHO_MODE else "AI"
    logger.info(f"サーバー起動: {settings.host}:{settings.port} ({mode}モード)")
    if not ECHO_MODE:
        from ai_chatter.config import llm_config
        logger.info(f"LLMモデル: {llm_config.model}")

    uvicorn.run(
        fastapi_app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        ws_ping_interval=10,
        ws_ping_timeout=10,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
