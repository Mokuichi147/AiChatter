from __future__ import annotations

import logging

from ai_chatter.config import SkillEntry, SkillsConfig
from ai_chatter.tools.conversation_memory import MemoryStore

logger = logging.getLogger(__name__)


class SkillProvider:
    """ユーザーの発言に基づいて関連スキル(ツールガイド+メモリ)を動的に取得する。"""

    def __init__(
        self,
        memory_store: MemoryStore,
        skills_config: SkillsConfig,
    ) -> None:
        self._memory_store = memory_store
        self._memory_top_k = skills_config.memory_top_k
        self._tool_skill_top_k = skills_config.tool_skill_top_k
        self._tool_skills: list[SkillEntry] = [
            s for s in skills_config.tools if s.match and s.guide
        ]

    async def retrieve(
        self,
        query: str,
        available_tools: set[str] | None = None,
    ) -> str:
        sections: list[str] = []

        tool_ctx = await self._retrieve_tool_skills(query)
        if tool_ctx:
            sections.append(tool_ctx)

        memory_ctx = await self._retrieve_memories(query)
        if memory_ctx:
            sections.append(memory_ctx)

        return "\n\n".join(sections)

    async def _retrieve_tool_skills(self, query: str) -> str:
        candidates = self._tool_skills
        if not candidates:
            return ""

        # 候補数がtop_k以下ならそのまま全て返す
        if len(candidates) <= self._tool_skill_top_k:
            return "\n".join(s.guide.strip() for s in candidates)

        # Embeddingで関連度の高いスキルを選択
        docs = [s.match for s in candidates]
        scores = await self._memory_store.embedding_similarity(query, docs)

        ranked = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )
        selected = [s for s, score in ranked[: self._tool_skill_top_k] if score > 0]

        if not selected:
            return "\n".join(s.guide.strip() for s in candidates)

        return "\n".join(s.guide.strip() for s in selected)

    async def _retrieve_memories(self, query: str) -> str:
        results = await self._memory_store.search(query, include_auto=False)
        if not results:
            return ""

        top = results[: self._memory_top_k]
        lines = [f"- {r['key']}: {r['content']}" for r in top]
        return "## 関連する記憶\n" + "\n".join(lines)
