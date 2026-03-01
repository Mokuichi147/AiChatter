import asyncio
import logging
import os
import struct

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from config import settings, character_data_path

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
_tool_registry = None
_notification_store = None
_memory_store = None
_subagent_job_manager = None

# アクティブなパイプライン管理
_active_pipelines: list = []


def make_header(msg_type: int, seq: int, payload_len: int) -> bytes:
    return struct.pack(">BHI", msg_type, seq & 0xFFFF, payload_len)


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
    global _asr, _llm, _tts, _tool_registry, _notification_store, _memory_store, _subagent_job_manager
    if not ECHO_MODE:
        logger.info("AIモデルをプリロード中...")
        from local_asr import LocalASR
        from local_llm import LocalLLM
        from local_tts import LocalTTS
        _asr = LocalASR()
        _llm = LocalLLM()
        _tts = LocalTTS()
        logger.info("AIモデルプリロード完了")

        # ツールレジストリ初期化
        if settings.tools_enabled:
            from tools import ToolRegistry
            from tools.conversation_memory import (
                MemoryStore,
                SaveMemoryTool,
                SearchMemoryTool,
                DeleteMemoryTool,
            )
            from tools.voice_control import SetVolumeTool
            from tools.search import SearchTool
            from tools.notification import (
                NotificationStore,
                SetNotificationTool,
                ListNotificationsTool,
                DeleteNotificationTool,
            )
            from tools.sleep_control import SetSleepTool
            from tools.display_control import DisplayTextTool, DisplayImageTool

            _tool_registry = ToolRegistry()
            _memory_store = MemoryStore(character_data_path("memory.json"))
            memory_store = _memory_store
            _tool_registry.register(SaveMemoryTool(memory_store))
            _tool_registry.register(SearchMemoryTool(memory_store))
            _tool_registry.register(DeleteMemoryTool(memory_store))
            _tool_registry.register(SetVolumeTool(_tts))
            _tool_registry.register(SearchTool())

            _notification_store = NotificationStore(settings.notification_file)
            _tool_registry.register(SetNotificationTool(_notification_store))
            _tool_registry.register(ListNotificationsTool(_notification_store))
            _tool_registry.register(DeleteNotificationTool(_notification_store))
            _tool_registry.register(SetSleepTool(lambda: _active_pipelines))
            _tool_registry.register(DisplayTextTool(lambda: _active_pipelines))
            _tool_registry.register(DisplayImageTool(lambda: _active_pipelines))

            if settings.subagent_enabled:
                from subagent.job_manager import SubAgentJobManager
                from subagent.runner import SubAgentRunner
                from subagent.tool_adapter import SubAgentToolAdapter
                from subagent_llm import SubAgentLLM
                from tools.subagent_research import (
                    GetSubAgentJobTool,
                    ListSubAgentJobsTool,
                    RunSubAgentResearchTool,
                )

                subagent_llm = SubAgentLLM()
                tool_adapter = SubAgentToolAdapter(
                    _tool_registry,
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

                _tool_registry.register(
                    RunSubAgentResearchTool(_subagent_job_manager)
                )
                _tool_registry.register(ListSubAgentJobsTool(_subagent_job_manager))
                _tool_registry.register(GetSubAgentJobTool(_subagent_job_manager))
                logger.info("サブエージェント機能初期化完了")

            logger.info("ツールレジストリ初期化完了")

        # 通知スケジューラー起動
        asyncio.create_task(_notification_scheduler())
        asyncio.create_task(_subagent_result_scheduler())


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "mode": "echo" if ECHO_MODE else "ai"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info(f"WebSocket接続: {client_host}")

    pipeline = None
    if not ECHO_MODE and _asr and _llm and _tts:
        try:
            from audio_pipeline import AudioPipeline

            async def send_fn(data: bytes) -> None:
                await websocket.send_bytes(data)

            pipeline = AudioPipeline(send_fn, _asr, _llm, _tts, _tool_registry, _memory_store)
            _active_pipelines.append(pipeline)
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


if __name__ == "__main__":
    mode = "エコー" if ECHO_MODE else "AI"
    logger.info(f"サーバー起動: {settings.host}:{settings.port} ({mode}モード)")
    if not ECHO_MODE:
        logger.info(f"LLMモデル: {settings.llm_model}")
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        ws_ping_interval=10,
        ws_ping_timeout=10,
    )
