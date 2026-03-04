from __future__ import annotations

from dataclasses import dataclass

from character_catalog import CharacterCatalog
from chat_engine import ChatEngine
from config import settings
from local_llm import LocalLLM
from session_manager import SessionManager
from tool_factory import ToolFactory


@dataclass
class AiChatterOptions:
    character_dir: str = settings.character_dir
    character_glob: str = settings.character_glob
    default_character_file: str = settings.character_file
    default_history_mode: str = settings.default_history_mode
    max_history_messages: int = settings.chat_max_history_messages
    enable_tools: bool = settings.tools_enabled


class AiChatterRuntime:
    """同一プロセスで会話エンジンを使うための軽量SDK。"""

    def __init__(self, engine: ChatEngine, catalog: CharacterCatalog, session_manager: SessionManager) -> None:
        self.engine = engine
        self.catalog = catalog
        self.session_manager = session_manager

    @classmethod
    async def create(cls, options: AiChatterOptions | None = None) -> "AiChatterRuntime":
        opts = options or AiChatterOptions()

        catalog = CharacterCatalog(opts.character_dir, opts.character_glob)
        catalog.reload()
        if not catalog.list_entries():
            catalog.register_file(opts.default_character_file)

        default_character_id = catalog.default_character_id(opts.default_character_file)
        session_manager = SessionManager(
            default_character_id=default_character_id,
            default_history_mode=opts.default_history_mode,
            max_messages=opts.max_history_messages,
        )

        tool_registry = _build_tool_registry() if opts.enable_tools else None
        llm = LocalLLM()
        engine = ChatEngine(
            llm=llm,
            session_manager=session_manager,
            character_catalog=catalog,
            tool_registry=tool_registry,
        )
        return cls(engine=engine, catalog=catalog, session_manager=session_manager)

    async def create_session(
        self,
        session_id: str,
        history_mode: str | None = None,
        character_id: str | None = None,
    ) -> dict:
        return await self.engine.ensure_session(session_id, history_mode, character_id)

    async def set_session_character(self, session_id: str, character_id: str) -> dict:
        return await self.engine.set_session_character(session_id, character_id)

    async def list_sessions(self) -> list[dict]:
        return await self.engine.list_sessions()

    async def delete_session(self, session_id: str) -> bool:
        return await self.engine.delete_session(session_id)

    def list_characters(self) -> list[dict]:
        return self.engine.list_characters()

    def get_character(self, character_id: str) -> dict:
        return self.engine.get_character(character_id)

    async def chat(self, session_id: str, text: str) -> dict:
        return await self.engine.chat(session_id=session_id, text=text)

    async def stream_chat(self, session_id: str, text: str):
        async for event in self.engine.stream_chat(session_id=session_id, text=text):
            yield event


async def create_runtime(options: AiChatterOptions | None = None) -> AiChatterRuntime:
    return await AiChatterRuntime.create(options)


def _build_tool_registry():
    factory = ToolFactory(tts=None, get_pipelines=lambda: [])
    return factory.create_registry(set())
