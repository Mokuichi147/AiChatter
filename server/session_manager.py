from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

HISTORY_MODE_SHARED = "shared"
HISTORY_MODE_ISOLATED = "isolated"
ALLOWED_HISTORY_MODES = {HISTORY_MODE_SHARED, HISTORY_MODE_ISOLATED}


@dataclass
class SessionState:
    session_id: str
    character_id: str
    history_mode: str = HISTORY_MODE_SHARED
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SessionManager:
    """REST/CLI向けのセッション状態と会話履歴を管理する。"""

    def __init__(self, default_character_id: str, default_history_mode: str = HISTORY_MODE_SHARED, max_messages: int = 20) -> None:
        if default_history_mode not in ALLOWED_HISTORY_MODES:
            raise ValueError(f"history_modeが不正です: {default_history_mode}")
        self._default_character_id = default_character_id
        self._default_history_mode = default_history_mode
        self._max_messages = max(2, max_messages)

        self._sessions: dict[str, SessionState] = {}
        self._histories: dict[str, list[dict]] = {}
        self._shared_history: list[dict] = []

        self._lock = asyncio.Lock()
        self._session_locks: dict[str, asyncio.Lock] = {}

    @property
    def default_history_mode(self) -> str:
        return self._default_history_mode

    async def ensure_session(
        self,
        session_id: str,
        history_mode: str | None = None,
        character_id: str | None = None,
    ) -> SessionState:
        mode = history_mode or self._default_history_mode
        if mode not in ALLOWED_HISTORY_MODES:
            raise ValueError(f"history_modeが不正です: {mode}")

        async with self._lock:
            current = self._sessions.get(session_id)
            now = datetime.now().isoformat()
            if current is None:
                current = SessionState(
                    session_id=session_id,
                    history_mode=mode,
                    character_id=character_id or self._default_character_id,
                    created_at=now,
                    updated_at=now,
                )
                self._sessions[session_id] = current
                if current.history_mode == HISTORY_MODE_ISOLATED:
                    self._histories.setdefault(session_id, [])
            else:
                if history_mode is not None:
                    current.history_mode = mode
                    if mode == HISTORY_MODE_ISOLATED:
                        self._histories.setdefault(session_id, [])
                if character_id:
                    current.character_id = character_id
                current.updated_at = now

            self._session_locks.setdefault(session_id, asyncio.Lock())
            return SessionState(**current.__dict__)

    async def set_character(self, session_id: str, character_id: str) -> SessionState:
        async with self._lock:
            current = self._sessions.get(session_id)
            if current is None:
                raise KeyError(f"session_idが存在しません: {session_id}")
            current.character_id = character_id
            current.updated_at = datetime.now().isoformat()
            return SessionState(**current.__dict__)

    async def list_sessions(self) -> list[SessionState]:
        async with self._lock:
            return [SessionState(**s.__dict__) for s in self._sessions.values()]

    async def delete_session(self, session_id: str) -> bool:
        async with self._lock:
            existed = session_id in self._sessions
            self._sessions.pop(session_id, None)
            self._histories.pop(session_id, None)
            self._session_locks.pop(session_id, None)
            return existed

    async def get_history(self, session_id: str) -> list[dict]:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(f"session_idが存在しません: {session_id}")
            if state.history_mode == HISTORY_MODE_SHARED:
                return list(self._shared_history)
            return list(self._histories.get(session_id, []))

    async def append_history(self, session_id: str, entries: list[dict]) -> None:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(f"session_idが存在しません: {session_id}")
            state.updated_at = datetime.now().isoformat()

            if state.history_mode == HISTORY_MODE_SHARED:
                self._shared_history.extend(entries)
                if len(self._shared_history) > self._max_messages:
                    self._shared_history = self._shared_history[-self._max_messages :]
                return

            history = self._histories.setdefault(session_id, [])
            history.extend(entries)
            if len(history) > self._max_messages:
                self._histories[session_id] = history[-self._max_messages :]

    async def resolve_character_id(self, session_id: str) -> str:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(f"session_idが存在しません: {session_id}")
            return state.character_id

    async def get_session_lock(self, session_id: str) -> asyncio.Lock:
        async with self._lock:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[session_id] = lock
            return lock
