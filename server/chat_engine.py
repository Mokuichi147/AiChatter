from __future__ import annotations

import json
from collections import OrderedDict
from datetime import datetime
from typing import AsyncIterator

from character_catalog import CharacterCatalog
from config import prompt_config, settings
from local_llm import LocalLLM, TextChunk, ToolCallRequest
from session_manager import SessionManager
from skills import SkillProvider
from tools.base import ToolResult
from tools.registry import ToolRegistry

MAX_TOOL_ROUNDS = 5


class ChatEngine:
    """テキスト会話のコア実行器（REST/CLI/SDK共通）。"""

    def __init__(
        self,
        llm: LocalLLM,
        session_manager: SessionManager,
        character_catalog: CharacterCatalog,
        tool_registry: ToolRegistry | None = None,
        skill_provider: SkillProvider | None = None,
    ) -> None:
        self._llm = llm
        self._session_manager = session_manager
        self._character_catalog = character_catalog
        self._tool_registry = tool_registry
        self._skill_provider = skill_provider

    def _resolve_openai_tools(self) -> list[dict]:
        if (
            self._tool_registry is None
            or self._tool_registry.is_empty
            or not settings.tools_enabled
        ):
            return []
        return self._tool_registry.to_openai_tools()

    def _build_system_prompt(
        self, character_id: str, skill_context: str = "",
    ) -> str:
        entry = self._character_catalog.get(character_id)
        system_prompt = entry.config.persona.system_prompt

        if prompt_config.output_rules:
            system_prompt += "\n\n" + prompt_config.output_rules.strip()

        if (
            self._skill_provider
            and self._tool_registry
            and not self._tool_registry.is_empty
            and settings.tools_enabled
        ):
            base = self._skill_provider.tool_guide_base
            if base:
                system_prompt += "\n\n" + base

        if skill_context:
            system_prompt += "\n\n" + skill_context

        now = datetime.now()
        return system_prompt.replace("{{DATETIME}}", now.strftime("%Y年%m月%d日 %H:%M"))

    async def ensure_session(
        self,
        session_id: str,
        history_mode: str | None = None,
        character_id: str | None = None,
    ) -> dict:
        if character_id and not self._character_catalog.has(character_id):
            raise ValueError(f"不明なcharacter_idです: {character_id}")
        state = await self._session_manager.ensure_session(
            session_id=session_id,
            history_mode=history_mode,
            character_id=character_id,
        )
        return {
            "session_id": state.session_id,
            "character_id": state.character_id,
            "history_mode": state.history_mode,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
        }

    async def set_session_character(self, session_id: str, character_id: str) -> dict:
        if not self._character_catalog.has(character_id):
            raise ValueError(f"不明なcharacter_idです: {character_id}")
        state = await self._session_manager.set_character(session_id, character_id)
        return {
            "session_id": state.session_id,
            "character_id": state.character_id,
            "history_mode": state.history_mode,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
        }

    async def list_sessions(self) -> list[dict]:
        sessions = await self._session_manager.list_sessions()
        return [
            {
                "session_id": s.session_id,
                "character_id": s.character_id,
                "history_mode": s.history_mode,
                "created_at": s.created_at,
                "updated_at": s.updated_at,
            }
            for s in sessions
        ]

    async def delete_session(self, session_id: str) -> bool:
        return await self._session_manager.delete_session(session_id)

    async def chat(self, session_id: str, text: str) -> dict:
        full_text = ""
        used_tools: list[str] = []
        done_data: dict | None = None

        async for event in self.stream_chat(session_id=session_id, text=text):
            etype = event.get("type")
            if etype == "chunk":
                full_text += event.get("text", "")
            elif etype == "tool_call":
                name = event.get("name")
                if name:
                    used_tools.append(name)
            elif etype == "done":
                done_data = event

        uniq_tools = list(OrderedDict.fromkeys(t for t in used_tools if t))
        if done_data is None:
            now = datetime.now().isoformat()
            done_data = {
                "type": "done",
                "session_id": session_id,
                "text": full_text.strip(),
                "used_tools": uniq_tools,
                "created_at": now,
            }
        else:
            done_data["used_tools"] = uniq_tools

        return done_data

    async def stream_chat(self, session_id: str, text: str) -> AsyncIterator[dict]:
        text = (text or "").strip()
        if not text:
            raise ValueError("textは必須です")

        await self._session_manager.ensure_session(session_id)
        lock = await self._session_manager.get_session_lock(session_id)

        async with lock:
            character_id = await self._session_manager.resolve_character_id(session_id)
            tools = self._resolve_openai_tools()
            tool_names = {
                t.get("name", "")
                for t in tools
                if t.get("name")
            }

            skill_context = ""
            if self._skill_provider:
                skill_context = await self._skill_provider.retrieve(
                    text, available_tools=tool_names or None,
                )

            system_prompt = self._build_system_prompt(character_id, skill_context)
            history = await self._session_manager.get_history(session_id)

            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(
                {
                    "role": h["role"],
                    "content": (
                        f"[{h['created_at']}] {h['content']}"
                        if h["role"] == "user" and h.get("created_at")
                        else h["content"]
                    ),
                }
                for h in history
            )
            messages.append({"role": "user", "content": text})

            full_response = ""
            used_tools: list[str] = []
            allow_empty_retry = True

            for _ in range(MAX_TOOL_ROUNDS):
                tool_call_requests: list[ToolCallRequest] = []
                round_text = ""

                async for event in self._llm.generate_stream(messages, tools):
                    if isinstance(event, TextChunk):
                        round_text += event.text
                        yield {
                            "type": "chunk",
                            "session_id": session_id,
                            "text": event.text,
                        }
                    elif isinstance(event, ToolCallRequest):
                        tool_call_requests.append(event)

                # ツール呼び出しがなければ応答完了
                if not tool_call_requests:
                    if not round_text.strip() and allow_empty_retry:
                        allow_empty_retry = False
                        messages.append({"role": "user", "content": "返答をお願いします。"})
                        tools = None
                        continue
                    full_response = round_text.strip()
                    break

                # function_callアイテムをinputに追加
                for tc in tool_call_requests:
                    messages.append(
                        {
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    )

                for tc in tool_call_requests:
                    used_tools.append(tc.name)
                    result: ToolResult
                    if self._tool_registry is None:
                        result = ToolResult(
                            content=f"ツールは無効です: {tc.name}", is_error=True
                        )
                    else:
                        result = await self._tool_registry.execute(tc.name, tc.arguments)

                    messages.append(
                        {
                            "type": "function_call_output",
                            "call_id": tc.id,
                            "output": result.content,
                        }
                    )
                    yield {
                        "type": "tool_call",
                        "session_id": session_id,
                        "name": tc.name,
                        "is_error": result.is_error,
                        "result": result.content,
                    }

            if not full_response:
                full_response = "すみません。うまく応答を生成できませんでした。"

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            await self._session_manager.append_history(
                session_id,
                [
                    {"role": "user", "content": text, "created_at": now_str},
                    {
                        "role": "assistant",
                        "content": full_response,
                        "created_at": now_str,
                    },
                ],
            )

            uniq_tools = list(OrderedDict.fromkeys(t for t in used_tools if t))
            yield {
                "type": "done",
                "session_id": session_id,
                "text": full_response,
                "used_tools": uniq_tools,
                "created_at": datetime.now().isoformat(),
                "character_id": character_id,
            }

    def list_characters(self) -> list[dict]:
        results: list[dict] = []
        for entry in self._character_catalog.list_entries():
            persona = entry.config.persona
            summary = (persona.system_prompt or "").strip().replace("\n", " ")
            if len(summary) > 120:
                summary = summary[:117] + "..."
            results.append(
                {
                    "character_id": entry.character_id,
                    "name": persona.name or entry.file_name,
                    "file_name": entry.file_name,
                    "summary": summary,
                }
            )
        return results

    def get_character(self, character_id: str) -> dict:
        entry = self._character_catalog.get(character_id)
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

    @staticmethod
    def event_to_sse(event: dict) -> str:
        """SSEフォーマットへ整形する。"""
        etype = event.get("type", "message")
        payload = json.dumps(event, ensure_ascii=False)
        return f"event: {etype}\ndata: {payload}\n\n"
