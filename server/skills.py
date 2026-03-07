from __future__ import annotations

import logging

from tools.conversation_memory import MemoryStore

logger = logging.getLogger(__name__)


class SkillProvider:
    """ユーザーの発言に基づいて関連スキル(ツールガイド+メモリ)を動的に取得する。"""

    def __init__(
        self,
        memory_store: MemoryStore,
        tool_guide: str = "",
        memory_top_k: int = 3,
        tool_skill_top_k: int = 5,
    ) -> None:
        self._memory_store = memory_store
        self._memory_top_k = memory_top_k
        self._tool_skill_top_k = tool_skill_top_k
        self._tool_guide_base, self._tool_skills = self._parse_tool_guide(tool_guide)

    @staticmethod
    def _parse_tool_guide(tool_guide: str) -> tuple[str, list[dict]]:
        base_lines: list[str] = []
        tool_entries: list[dict] = []
        for line in tool_guide.splitlines():
            stripped = line.strip()
            if stripped.startswith("- ") and ":" in stripped:
                rest = stripped[2:]
                name, desc = rest.split(":", 1)
                tool_entries.append({
                    "name": name.strip(),
                    "text": stripped,
                    "description": desc.strip(),
                })
            else:
                base_lines.append(line)
        return "\n".join(base_lines).strip(), tool_entries

    @property
    def tool_guide_base(self) -> str:
        return self._tool_guide_base

    async def retrieve(
        self,
        query: str,
        available_tools: set[str] | None = None,
    ) -> str:
        sections: list[str] = []

        tool_ctx = await self._retrieve_tool_skills(query, available_tools)
        if tool_ctx:
            sections.append(tool_ctx)

        memory_ctx = await self._retrieve_memories(query)
        if memory_ctx:
            sections.append(memory_ctx)

        return "\n\n".join(sections)

    async def _retrieve_tool_skills(
        self,
        query: str,
        available_tools: set[str] | None,
    ) -> str:
        candidates = self._tool_skills
        if available_tools:
            candidates = [s for s in candidates if s["name"] in available_tools]
        if not candidates:
            return ""

        # 候補数がtop_k以下ならそのまま全て返す
        if len(candidates) <= self._tool_skill_top_k:
            return "\n".join(s["text"] for s in candidates)

        # Embeddingで関連度の高いツールを選択
        docs = [s["description"] for s in candidates]
        scores = await self._memory_store.embedding_similarity(query, docs)

        ranked = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )
        selected = [s for s, score in ranked[: self._tool_skill_top_k] if score > 0]

        if not selected:
            return "\n".join(s["text"] for s in candidates)

        return "\n".join(s["text"] for s in selected)

    async def _retrieve_memories(self, query: str) -> str:
        results = await self._memory_store.search(query, include_auto=False)
        if not results:
            return ""

        top = results[: self._memory_top_k]
        lines = [f"- {r['key']}: {r['content']}" for r in top]
        return "## 関連する記憶\n" + "\n".join(lines)
