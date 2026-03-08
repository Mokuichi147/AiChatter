from __future__ import annotations

import argparse
import asyncio
import glob
import logging
import signal
import sys
from pathlib import Path

from ai_chatter.character_catalog import CharacterCatalog
from ai_chatter.config import settings
from ai_chatter.session_manager import ALLOWED_HISTORY_MODES, SessionManager
from ai_chatter.tool_factory import ToolFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve_character(catalog: CharacterCatalog, value: str | None) -> str:
    if value:
        candidate = value.strip()
        if catalog.has(candidate):
            return candidate

        # CLIはローカルファイルパス指定も許可
        path = Path(candidate).expanduser()
        if path.exists() and path.is_file():
            entry = catalog.register_file(str(path))
            logger.info(f"キャラクターを追加登録: {entry.character_id} ({entry.file_path})")
            return entry.character_id

        raise ValueError(f"character指定が不正です: {value}")

    entries = catalog.list_entries()
    if not entries:
        raise RuntimeError("利用可能なキャラクターがありません")

    print("利用可能なキャラクター:")
    for idx, entry in enumerate(entries, 1):
        name = entry.config.persona.name or entry.file_name
        print(f"  {idx}. {name} [{entry.character_id}]")
    print("  p. YAMLファイルパスを入力")

    while True:
        raw = input("選択してください (番号 or p): ").strip().lower()
        if raw == "p":
            file_path = input("YAMLファイルパス: ").strip()
            entry = catalog.register_file(file_path)
            logger.info(f"キャラクターを追加登録: {entry.character_id} ({entry.file_path})")
            return entry.character_id
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(entries):
                return entries[n - 1].character_id
        print("入力が不正です。もう一度入力してください。")


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


async def _run_chat(args: argparse.Namespace) -> None:
    from ai_chatter.chat_engine import ChatEngine
    from ai_chatter.local_llm import LocalLLM

    catalog = CharacterCatalog(settings.character_dir, settings.character_glob)
    catalog.reload()
    if not catalog.list_entries():
        catalog.register_file(settings.character_file)

    character_id = _resolve_character(catalog, args.character)

    history_mode = (args.history_mode or settings.default_history_mode).strip().lower()
    if history_mode not in ALLOWED_HISTORY_MODES:
        raise ValueError(
            f"history-mode は {sorted(ALLOWED_HISTORY_MODES)} のいずれかを指定してください"
        )

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

    chosen = catalog.get(character_id)
    chosen_name = chosen.config.persona.name or chosen.file_name
    print(f"\nCLI会話を開始します。session_id={session_id}, character={chosen_name} ({character_id})")
    print("終了するには 'exit' または 'quit' を入力してください。\n")

    # asyncio の SIGINT ハンドラを無効化し、input() が Ctrl+C で
    # 即座に KeyboardInterrupt を受け取れるようにする
    signal.signal(signal.SIGINT, signal.default_int_handler)

    try:
        while True:
            try:
                user_text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                break

            if args.stream:
                print("ai> ", end="", flush=True)
                async for event in engine.stream_chat(session_id=session_id, text=user_text):
                    if event.get("type") == "chunk":
                        print(event.get("text", ""), end="", flush=True)
                    elif event.get("type") == "done":
                        print()
            else:
                result = await engine.chat(session_id=session_id, text=user_text)
                print(f"ai> {result.get('text', '')}")
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
        "--character",
        help="キャラクターIDまたはYAMLファイルパス。省略時は一覧選択",
    )
    chat_parser.add_argument(
        "--history-mode",
        default=settings.default_history_mode,
        help=f"会話履歴モード ({'/'.join(sorted(ALLOWED_HISTORY_MODES))})",
    )
    chat_parser.add_argument("--session-id", default="cli")
    chat_parser.add_argument("--stream", action="store_true", help="ストリーミング表示")

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
