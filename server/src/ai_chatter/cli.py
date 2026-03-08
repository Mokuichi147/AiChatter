from __future__ import annotations

import argparse
import asyncio
import glob
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

from ai_chatter.character_catalog import CharacterCatalog
from ai_chatter.config import settings
from ai_chatter.session_manager import ALLOWED_HISTORY_MODES, SessionManager
from ai_chatter.tool_factory import ToolFactory

logger = logging.getLogger(__name__)



def _build_tools():
    factory = ToolFactory(tts=None, get_pipelines=lambda: [])
    return factory.create_registry(set()), factory.skill_provider


def _apply_server_character_selection(values: list[str] | None) -> None:
    """server起動時の-c/--character引数を解釈して設定へ反映する。"""
    if not values:
        return

    from ai_chatter import config as _config
    from ai_chatter.config import load_character

    expanded: list[str] = []
    original_patterns: list[str] = []

    for raw in values:
        token = (raw or "").strip()
        if not token:
            continue

        if any(ch in token for ch in "*?[]"):
            original_patterns.append(token)
            matched = sorted(glob.glob(token))
            if matched:
                expanded.extend(matched)
            continue

        expanded.append(token)

    # サンプル定義はデフォルト候補から除外
    expanded = [p for p in expanded if not str(p).endswith(".example.yaml")]

    if not expanded:
        raise ValueError(f"character指定に一致するファイルがありません: {values}")

    # 先頭をデフォルトキャラクターとして採用
    default_path = Path(expanded[0]).expanduser()
    if not default_path.is_absolute():
        default_path = Path.cwd() / default_path
    if not default_path.exists() or not default_path.is_file():
        raise ValueError(f"デフォルトキャラクターファイルが見つかりません: {default_path}")

    default_str = str(default_path)
    settings.character_file = default_str
    _config.character = load_character(default_str)

    # ワイルドカード指定なら、カタログ探索条件も更新する
    if original_patterns:
        pat = original_patterns[0]
        p = Path(pat).expanduser()
        if p.is_absolute():
            settings.character_dir = str(p.parent)
            settings.character_glob = p.name
        else:
            # 相対パターンは現在ディレクトリ基準で絶対化
            abs_pat = (Path.cwd() / p).resolve()
            settings.character_dir = str(abs_pat.parent)
            settings.character_glob = abs_pat.name
    else:
        # 明示ファイル指定のみの場合は、デフォルトファイルのディレクトリを探索対象に含める
        settings.character_dir = str(default_path.parent)
        settings.character_glob = "character*.yaml"

    logger.info(
        "server起動キャラクター設定: default=%s, dir=%s, glob=%s",
        settings.character_file,
        settings.character_dir,
        settings.character_glob,
    )


async def _run_voice(args: argparse.Namespace) -> None:
    from ai_chatter.local_asr import LocalASR
    from ai_chatter.local_llm import LocalLLM
    from ai_chatter.local_tts import LocalTTS
    from ai_chatter.voice_cli import VoiceCLI

    _apply_server_character_selection(args.character)

    if args.group:
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

    voice = VoiceCLI(
        asr=asr,
        llm=llm,
        tts=tts,
        tool_registry=tool_registry,
        skill_provider=skill_provider,
        speaker_id=speaker_id,
    )
    await voice.run()


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


async def _run_chat(args: argparse.Namespace) -> None:
    from ai_chatter.chat_engine import ChatEngine
    from ai_chatter.local_llm import LocalLLM

    _apply_server_character_selection(args.character)

    catalog = CharacterCatalog(settings.character_dir, settings.character_glob)
    catalog.reload()
    default_name = Path(settings.character_file).name
    if not catalog.has(default_name):
        catalog.register_file(settings.character_file)
    character_id = default_name

    history_mode = (args.history_mode or settings.default_history_mode).strip().lower()
    if history_mode not in ALLOWED_HISTORY_MODES:
        raise ValueError(
            f"history-mode は {sorted(ALLOWED_HISTORY_MODES)} のいずれかを指定してください"
        )

    tts = None
    if args.voice:
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

    session_id = args.session_id or "cli"
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

    # asyncio の SIGINT ハンドラを無効化し、input() が Ctrl+C で
    # 即座に KeyboardInterrupt を受け取れるようにする
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

            if args.stream:
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


def _run_server(args: argparse.Namespace) -> None:
    _apply_server_character_selection(args.character)

    if args.group:
        settings.conversation_mode = "group"

    from ai_chatter.main import ECHO_MODE, app
    import uvicorn

    mode = "エコー" if ECHO_MODE else "AI"
    logger.info(f"サーバー起動: {settings.host}:{settings.port} ({mode}モード)")
    if not ECHO_MODE:
        from ai_chatter.config import llm_config
        logger.info(f"LLMモデル: {llm_config.model}")

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        ws_ping_interval=10,
        ws_ping_timeout=10,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AiChatter CLI")
    subparsers = parser.add_subparsers(dest="command")

    chat_parser = subparsers.add_parser("chat", help="対話CLIモード")
    chat_parser.add_argument(
        "-c",
        "--character",
        nargs="+",
        help=(
            "デフォルトキャラクターファイルまたはglobパターン。"
            "例: -c character.yaml / -c 'character*.yaml'"
        ),
    )
    chat_parser.add_argument(
        "--history-mode",
        default=settings.default_history_mode,
        help=f"会話履歴モード ({'/'.join(sorted(ALLOWED_HISTORY_MODES))})",
    )
    chat_parser.add_argument("--session-id", default="cli")
    chat_parser.add_argument("--stream", action="store_true", help="ストリーミング表示")
    chat_parser.add_argument("--voice", action="store_true", help="返答をTTS音声で再生")
    chat_parser.add_argument("--debug", action="store_true", help="デバッグログを表示")

    voice_parser = subparsers.add_parser("voice", help="音声対話CLIモード (PCマイク/スピーカー)")
    voice_parser.add_argument(
        "-c",
        "--character",
        nargs="+",
        help=(
            "デフォルトキャラクターファイルまたはglobパターン。"
            "例: -c character.yaml / -c 'character*.yaml'"
        ),
    )
    voice_parser.add_argument(
        "--group",
        action="store_true",
        help="グループモードで起動する（複数人会話・話者識別を有効化）",
    )
    voice_parser.add_argument("--debug", action="store_true", help="デバッグログを表示")

    server_parser = subparsers.add_parser("server", help="サーバー起動モード")
    server_parser.add_argument(
        "-c",
        "--character",
        nargs="+",
        help=(
            "デフォルトキャラクターファイルまたはglobパターン。"
            "例: -c character.yaml / -c 'character*.yaml'"
        ),
    )
    server_parser.add_argument(
        "--group",
        action="store_true",
        help="グループモードで起動する（複数人会話・話者識別を有効化）",
    )

    # 互換性: サブコマンド未指定時は chat 扱い
    if len(sys.argv) > 1 and sys.argv[1] in {"chat", "voice", "server", "-h", "--help"}:
        ns = parser.parse_args()
    else:
        ns = chat_parser.parse_args(sys.argv[1:])
        ns.command = "chat"

    # chat/voice は --debug 指定時のみログ表示、server は常に表示
    if ns.command == "server":
        log_level = logging.INFO
    elif getattr(ns, "debug", False):
        log_level = logging.DEBUG
    else:
        log_level = logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        if ns.command == "server":
            _run_server(ns)
        elif ns.command == "voice":
            asyncio.run(_run_voice(ns))
        else:
            asyncio.run(_run_chat(ns))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
