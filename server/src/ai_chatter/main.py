import asyncio
import gc
import glob
import inspect
import logging
import os
import struct
import uuid
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from ai_chatter.character_catalog import CharacterCatalog
from ai_chatter.chat_engine import ChatEngine
from ai_chatter.config import settings
from ai_chatter.session_manager import ALLOWED_HISTORY_MODES, HISTORY_MODE_SHARED, SessionManager
from ai_chatter.tool_factory import CAP_M5_DEVICE, ToolFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# WebSocketメッセージタイプ (ESP32 → Server)
MSG_AUDIO_CHUNK = 0x01
MSG_EOS = 0x11
MSG_INTERRUPT = 0x12
MSG_BUTTON = 0x13

# WebSocketメッセージタイプ (Server → ESP32)
MSG_TTS_CHUNK = 0x02
MSG_TTS_END = 0x03

HEADER_SIZE = 7  # [type:1][seq:2][payload_len:4]

# エコーモード: 環境変数 ECHO_MODE=1 またはAIモデル未インストール時
ECHO_MODE = os.environ.get("ECHO_MODE", "0") == "1"

app = FastAPI(title="AiChatter Server")

# プリロード済みAIモデル（サーバー起動時に初期化）
_asr = None
_llm = None
_tts = None
_speaker_id = None
_tool_factory: ToolFactory | None = None
_tool_registry_ws = None
_tool_registry_api = None
_notification_store = None
_memory_store = None
_skill_provider = None
_subagent_job_manager = None
_scheduler_tasks: list[asyncio.Task] = []

# REST/CLI向けコンポーネント
_character_catalog: CharacterCatalog | None = None
_session_manager: SessionManager | None = None
_chat_engine: ChatEngine | None = None

# アクティブなパイプライン管理
_active_pipelines: list = []


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateSessionRequest(StrictModel):
    session_id: str | None = None
    history_mode: str | None = None
    character_id: str | None = None


class SetSessionCharacterRequest(StrictModel):
    character_id: str


class ChatRequest(StrictModel):
    session_id: str
    text: str


class ChatStreamRequest(StrictModel):
    session_id: str
    text: str


def make_header(msg_type: int, seq: int, payload_len: int) -> bytes:
    return struct.pack(">BHI", msg_type, seq & 0xFFFF, payload_len)


def _resolve_character_cli_values(values: list[str]) -> tuple[str, str, str]:
    """-c/--character引数を解決し、default_file/dir/globを返す。"""
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

    # サンプル定義は除外
    expanded = [p for p in expanded if not str(p).endswith(".example.yaml")]
    if not expanded:
        raise ValueError(f"character指定に一致するファイルがありません: {values}")

    default_path = Path(expanded[0]).expanduser()
    if not default_path.is_absolute():
        default_path = (Path.cwd() / default_path).resolve()
    if not default_path.exists() or not default_path.is_file():
        raise ValueError(f"デフォルトキャラクターファイルが見つかりません: {default_path}")

    if original_patterns:
        p = Path(original_patterns[0]).expanduser()
        if p.is_absolute():
            character_dir = str(p.parent)
            character_glob = p.name
        else:
            abs_pat = (Path.cwd() / p).resolve()
            character_dir = str(abs_pat.parent)
            character_glob = abs_pat.name
    else:
        character_dir = str(default_path.parent)
        character_glob = "character*.yaml"

    return str(default_path), character_dir, character_glob


def _validate_history_mode(history_mode: str | None) -> str:
    mode = (history_mode or settings.default_history_mode or HISTORY_MODE_SHARED).strip().lower()
    if mode not in ALLOWED_HISTORY_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"history_modeは {sorted(ALLOWED_HISTORY_MODES)} のいずれかを指定してください。",
        )
    return mode


def _require_character_catalog() -> CharacterCatalog:
    if _character_catalog is None:
        raise HTTPException(status_code=503, detail="キャラクターカタログが初期化されていません。")
    return _character_catalog


def _require_session_manager() -> SessionManager:
    if _session_manager is None:
        raise HTTPException(status_code=503, detail="セッション管理が初期化されていません。")
    return _session_manager


def _require_chat_engine() -> ChatEngine:
    if _chat_engine is None:
        raise HTTPException(status_code=503, detail="チャットエンジンが初期化されていません。")
    return _chat_engine


async def _close_component(name: str, component: object | None) -> None:
    """close/shutdownを持つコンポーネントを安全に終了させる。"""
    if component is None:
        return

    close_fn = getattr(component, "close", None) or getattr(component, "shutdown", None)
    if not callable(close_fn):
        return

    try:
        result = close_fn()
        if inspect.isawaitable(result):
            await result
    except Exception as e:
        logger.warning(f"{name} の終了処理でエラー: {e}", exc_info=True)


async def _notification_scheduler() -> None:
    """30秒間隔で期限到来の通知をチェックし、全アクティブpipelineに送信する。"""
    while True:
        await asyncio.sleep(30)
        if _notification_store is None:
            continue
        due = _notification_store.pop_due()
        if not due and not _active_pipelines:
            continue
        for notification in due:
            message = notification.get("message", "")
            logger.info(f"通知発火: id={notification.get('id')} msg={message}")
            for pipeline in list(_active_pipelines):
                try:
                    await pipeline.generate_from_text(message)
                except Exception as e:
                    logger.error(f"通知送信エラー: {e}", exc_info=True)


async def _subagent_result_scheduler() -> None:
    """完了したサブエージェント結果をメインエージェントへ通知として渡す。"""
    while True:
        await asyncio.sleep(5)
        if _subagent_job_manager is None:
            continue
        if not _active_pipelines:
            # 送信先がいない間はキューを保持する
            continue

        messages = await _subagent_job_manager.pop_completed_messages(limit=20)
        if not messages:
            continue

        undelivered: list[str] = []
        for message in messages:
            logger.info("サブエージェント結果通知を送信")
            delivered = False
            for pipeline in list(_active_pipelines):
                try:
                    await pipeline.generate_from_text(message)
                    delivered = True
                except Exception as e:
                    logger.error(f"サブエージェント結果通知エラー: {e}", exc_info=True)
            if not delivered:
                undelivered.append(message)

        if undelivered:
            await _subagent_job_manager.requeue_completed_messages(undelivered)


@app.on_event("startup")
async def startup_event() -> None:
    global _asr, _llm, _tts, _speaker_id, _tool_factory, _tool_registry_ws, _tool_registry_api
    global _notification_store, _memory_store, _skill_provider
    global _subagent_job_manager, _scheduler_tasks, _character_catalog, _session_manager, _chat_engine

    # キャラクターカタログ初期化（REST/CLI共通）
    _character_catalog = CharacterCatalog(settings.character_dir, settings.character_glob)
    _character_catalog.reload()
    preferred_id = Path(settings.character_file).name if settings.character_file else ""
    if preferred_id and not _character_catalog.has(preferred_id):
        try:
            _character_catalog.register_file(settings.character_file, character_id=preferred_id)
            logger.info(f"起動時キャラクターを追加登録: {preferred_id}")
        except Exception:
            logger.warning(
                f"起動時キャラクターを追加できませんでした: {settings.character_file}",
                exc_info=True,
            )
    if not _character_catalog.list_entries():
        try:
            _character_catalog.register_file(settings.character_file)
        except Exception as e:
            raise RuntimeError(f"キャラクター読み込みに失敗: {e}") from e

    default_character_id = _character_catalog.default_character_id(settings.character_file)

    default_mode = settings.default_history_mode.strip().lower()
    if default_mode not in ALLOWED_HISTORY_MODES:
        logger.warning(
            f"default_history_modeが不正です: {settings.default_history_mode} -> '{HISTORY_MODE_SHARED}' を使用"
        )
        default_mode = HISTORY_MODE_SHARED

    _session_manager = SessionManager(
        default_character_id=default_character_id,
        default_history_mode=default_mode,
        max_messages=settings.chat_max_history_messages,
    )

    if not ECHO_MODE:
        logger.info("AIモデルをプリロード中...")
        from ai_chatter.local_asr import LocalASR
        from ai_chatter.local_llm import LocalLLM
        from ai_chatter.local_tts import LocalTTS

        _asr = LocalASR()
        _llm = LocalLLM()
        _tts = LocalTTS()
        logger.info("AIモデルプリロード完了")

        # グループモード: 話者識別初期化
        if settings.conversation_mode == "group":
            from ai_chatter.config import character_data_path
            from ai_chatter.speaker_id import SpeakerIdentifier

            speakers_path = character_data_path("speakers.json")
            _speaker_id = SpeakerIdentifier(
                speakers_path,
                similarity_threshold=settings.speaker_similarity_threshold,
            )
            logger.info("話者識別モジュール初期化完了 (グループモード)")

        # ツールレジストリ初期化
        if settings.tools_enabled:
            _tool_factory = ToolFactory(
                tts=_tts,
                get_pipelines=lambda: _active_pipelines,
                speaker_id=_speaker_id,
            )
            _tool_registry_ws = _tool_factory.create_registry({CAP_M5_DEVICE})
            _tool_registry_api = _tool_factory.create_registry(set())
            _memory_store = _tool_factory.memory_store
            _notification_store = _tool_factory.notification_store
            _skill_provider = _tool_factory.skill_provider

            if settings.subagent_enabled:
                from ai_chatter.subagent.job_manager import SubAgentJobManager
                from ai_chatter.subagent.runner import SubAgentRunner
                from ai_chatter.subagent.tool_adapter import SubAgentToolAdapter
                from ai_chatter.subagent_llm import SubAgentLLM

                subagent_llm = SubAgentLLM()
                tool_adapter = SubAgentToolAdapter(
                    _tool_registry_api,
                    settings.subagent_mcp_tool_denylist,
                )
                runner = SubAgentRunner(
                    llm=subagent_llm,
                    tool_adapter=tool_adapter,
                    max_rounds=settings.subagent_max_rounds,
                    result_max_chars=settings.subagent_result_max_chars,
                )
                _subagent_job_manager = SubAgentJobManager(
                    runner,
                    timeout_sec=settings.subagent_timeout_sec,
                )

                ToolFactory.register_subagent_tools(
                    _tool_registry_ws, _subagent_job_manager
                )
                ToolFactory.register_subagent_tools(
                    _tool_registry_api, _subagent_job_manager
                )
                logger.info("サブエージェント機能初期化完了")

            logger.info("ツールレジストリ初期化完了")

        # RESTチャットエンジン初期化
        _chat_engine = ChatEngine(
            llm=_llm,
            session_manager=_session_manager,
            character_catalog=_character_catalog,
            tool_registry=_tool_registry_api,
            skill_provider=_skill_provider,
        )

        # 通知スケジューラー起動
        _scheduler_tasks = [
            asyncio.create_task(_notification_scheduler(), name="notification_scheduler"),
            asyncio.create_task(
                _subagent_result_scheduler(), name="subagent_result_scheduler"
            ),
        ]


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global _asr, _llm, _tts, _speaker_id, _tool_factory, _tool_registry_ws, _tool_registry_api
    global _notification_store, _memory_store, _skill_provider
    global _subagent_job_manager, _scheduler_tasks, _character_catalog, _session_manager, _chat_engine

    logger.info("シャットダウン処理開始")

    if _scheduler_tasks:
        for task in _scheduler_tasks:
            task.cancel()
        results = await asyncio.gather(*_scheduler_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception) and not isinstance(
                result, asyncio.CancelledError
            ):
                logger.warning(f"スケジューラー停止中にエラー: {result}")
        _scheduler_tasks.clear()

    if _active_pipelines:
        pipelines = list(_active_pipelines)
        _active_pipelines.clear()
        for pipeline in pipelines:
            try:
                await pipeline.close()
            except Exception as e:
                logger.warning(f"パイプライン終了処理でエラー: {e}", exc_info=True)

    await _close_component("subagent_job_manager", _subagent_job_manager)
    await _close_component("tts", _tts)
    await _close_component("llm", _llm)
    await _close_component("asr", _asr)

    _subagent_job_manager = None
    _speaker_id = None
    _notification_store = None
    _memory_store = None
    _skill_provider = None
    _tool_registry_ws = None
    _tool_registry_api = None
    _tool_factory = None
    _tts = None
    _llm = None
    _asr = None

    _chat_engine = None
    _session_manager = None
    _character_catalog = None

    gc.collect()
    logger.info("シャットダウン処理完了")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "mode": "echo" if ECHO_MODE else "ai",
        "chat_api": _chat_engine is not None,
    }


@app.get("/api/v1/characters")
async def list_characters() -> dict:
    catalog = _require_character_catalog()
    items = []
    for entry in catalog.list_entries():
        persona = entry.config.persona
        summary = (persona.system_prompt or "").strip().replace("\n", " ")
        if len(summary) > 120:
            summary = summary[:117] + "..."
        items.append(
            {
                "character_id": entry.character_id,
                "name": persona.name or entry.file_name,
                "file_name": entry.file_name,
                "summary": summary,
            }
        )
    return {"items": items}


@app.get("/api/v1/characters/{character_id}")
async def get_character(character_id: str) -> dict:
    catalog = _require_character_catalog()
    try:
        entry = catalog.get(character_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {
        "character_id": entry.character_id,
        "file_name": entry.file_name,
        "name": entry.config.persona.name,
        "system_prompt": entry.config.persona.system_prompt,
        "voice": {
            "type": entry.config.voice.type,
            "description": entry.config.voice.description,
            "sample_text": entry.config.voice.sample_text,
            "voice_design_model": entry.config.voice.voice_design_model,
            "wav_file": entry.config.voice.wav_file,
            "transcript": entry.config.voice.transcript,
            "tts_model": entry.config.voice.tts_model,
        },
    }


@app.post("/api/v1/sessions")
async def create_session(req: CreateSessionRequest) -> dict:
    manager = _require_session_manager()
    catalog = _require_character_catalog()

    session_id = (req.session_id or "").strip() or uuid.uuid4().hex
    history_mode = _validate_history_mode(req.history_mode)

    character_id = (req.character_id or "").strip() or None
    if character_id and not catalog.has(character_id):
        raise HTTPException(status_code=422, detail=f"不明なcharacter_idです: {character_id}")

    try:
        state = await manager.ensure_session(
            session_id=session_id,
            history_mode=history_mode,
            character_id=character_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return {
        "session_id": state.session_id,
        "character_id": state.character_id,
        "history_mode": state.history_mode,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
    }


@app.get("/api/v1/sessions")
async def list_sessions() -> dict:
    manager = _require_session_manager()
    sessions = await manager.list_sessions()
    return {
        "items": [
            {
                "session_id": s.session_id,
                "character_id": s.character_id,
                "history_mode": s.history_mode,
                "created_at": s.created_at,
                "updated_at": s.updated_at,
            }
            for s in sessions
        ]
    }


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    manager = _require_session_manager()
    deleted = await manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"session_idが見つかりません: {session_id}")
    return {"deleted": True, "session_id": session_id}


@app.patch("/api/v1/sessions/{session_id}/character")
async def set_session_character(session_id: str, req: SetSessionCharacterRequest) -> dict:
    manager = _require_session_manager()
    catalog = _require_character_catalog()

    character_id = req.character_id.strip()
    if not character_id:
        raise HTTPException(status_code=422, detail="character_idは必須です")
    if not catalog.has(character_id):
        raise HTTPException(status_code=422, detail=f"不明なcharacter_idです: {character_id}")

    try:
        state = await manager.set_character(session_id, character_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return {
        "session_id": state.session_id,
        "character_id": state.character_id,
        "history_mode": state.history_mode,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
    }


@app.post("/api/v1/chat")
async def chat(req: ChatRequest) -> dict:
    engine = _require_chat_engine()
    session_id = req.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=422, detail="session_idは必須です")

    try:
        result = await engine.chat(session_id=session_id, text=req.text)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post("/api/v1/chat/stream")
async def chat_stream(req: ChatStreamRequest) -> StreamingResponse:
    engine = _require_chat_engine()
    session_id = req.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=422, detail="session_idは必須です")

    async def _event_generator() -> AsyncIterator[str]:
        try:
            async for event in engine.stream_chat(session_id=session_id, text=req.text):
                yield ChatEngine.event_to_sse(event)
        except Exception as e:
            error_event = {
                "type": "error",
                "session_id": session_id,
                "error": str(e),
            }
            yield ChatEngine.event_to_sse(error_event)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info(f"WebSocket接続: {client_host}")

    pipeline = None
    if not ECHO_MODE and _asr and _llm and _tts:
        try:
            from ai_chatter.audio_pipeline import AudioPipeline

            async def send_fn(data: bytes) -> None:
                await websocket.send_bytes(data)

            pipeline = AudioPipeline(
                send_fn, _asr, _llm, _tts, _tool_registry_ws, _skill_provider,
                speaker_id=_speaker_id,
            )
            _active_pipelines.append(pipeline)
            await pipeline.send_wake()
            logger.info("AIパイプラインモード")
        except Exception as e:
            logger.warning(f"AIパイプライン初期化失敗 (エコーモードにフォールバック): {e}")

    if pipeline is None:
        logger.info("エコーモード (受信音声をそのまま返送)")

    audio_buffer = bytearray()
    seq_counter = 0

    try:
        while True:
            data = await websocket.receive_bytes()

            if len(data) < HEADER_SIZE:
                logger.warning(f"短すぎるメッセージ: {len(data)}バイト")
                continue

            msg_type, seq, payload_len = struct.unpack(">BHI", data[:HEADER_SIZE])
            payload = data[HEADER_SIZE : HEADER_SIZE + payload_len]

            if pipeline:
                # AIパイプラインモード
                if msg_type == MSG_AUDIO_CHUNK:
                    await pipeline.process_audio_chunk(payload)
                elif msg_type == MSG_EOS:
                    logger.info(f"EOS受信 (seq={seq})")
                    await pipeline.process_end_of_speech()
                elif msg_type == MSG_INTERRUPT:
                    logger.info(f"バージイン受信 (seq={seq})")
                    await pipeline.process_interrupt()
                elif msg_type == MSG_BUTTON:
                    logger.info(f"ボタン押下受信 (seq={seq})")
                    asyncio.create_task(pipeline.process_button_press())
            else:
                # エコーモード
                if msg_type == MSG_AUDIO_CHUNK:
                    audio_buffer.extend(payload)
                elif msg_type == MSG_EOS:
                    logger.info(
                        f"EOS受信 (seq={seq}, バッファ: {len(audio_buffer)}バイト)"
                    )
                    # 蓄積した音声をそのままTTSチャンクとして返送
                    chunk_size = 1024
                    for i in range(0, len(audio_buffer), chunk_size):
                        chunk = bytes(audio_buffer[i : i + chunk_size])
                        seq_counter = (seq_counter + 1) & 0xFFFF
                        header = make_header(MSG_TTS_CHUNK, seq_counter, len(chunk))
                        await websocket.send_bytes(header + chunk)
                    # TTS終了通知
                    seq_counter = (seq_counter + 1) & 0xFFFF
                    header = make_header(MSG_TTS_END, seq_counter, 0)
                    await websocket.send_bytes(header)
                    logger.info(
                        f"エコーバック送信完了 ({len(audio_buffer)}バイト)"
                    )
                    audio_buffer.clear()
                elif msg_type == MSG_INTERRUPT:
                    logger.info(f"バージイン受信 (seq={seq})")
                    audio_buffer.clear()

    except WebSocketDisconnect:
        logger.info(f"WebSocket切断: {client_host}")
    except Exception as e:
        logger.error(f"WebSocketエラー: {e}", exc_info=True)
    finally:
        if pipeline and pipeline in _active_pipelines:
            _active_pipelines.remove(pipeline)
        if pipeline:
            try:
                await pipeline.close()
            except Exception as e:
                logger.warning(f"WebSocket終了時のパイプライン解放失敗: {e}", exc_info=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AiChatter Server")
    parser.add_argument(
        "-c",
        "--character",
        nargs="+",
        help=(
            "デフォルトキャラクターファイルまたはglobパターン。"
            "例: -c character.yaml / -c 'character*.yaml'"
        ),
    )
    parser.add_argument(
        "--group",
        action="store_true",
        help="グループモードで起動する（複数人会話・話者識別を有効化）",
    )
    args = parser.parse_args()

    if args.group:
        settings.conversation_mode = "group"

    if args.character:
        from ai_chatter import config as _config
        from ai_chatter.config import load_character

        try:
            default_file, character_dir, character_glob = _resolve_character_cli_values(args.character)
        except ValueError as e:
            parser.error(str(e))

        settings.character_file = default_file
        settings.character_dir = character_dir
        settings.character_glob = character_glob
        _config.character = load_character(default_file)
        logger.info(
            "起動キャラクター設定: default=%s, dir=%s, glob=%s",
            settings.character_file,
            settings.character_dir,
            settings.character_glob,
        )

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
