from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from config import character_data_path, llm_config, prompt_config, settings
from skills import SkillProvider
from tools import ToolRegistry
from tools.conversation_memory import DeleteMemoryTool, MemoryStore, SaveMemoryTool, SearchMemoryTool
from tools.display_control import DisplayImageTool, DisplayTextTool
from tools.notification import DeleteNotificationTool, ListNotificationsTool, NotificationStore, SetNotificationTool
from tools.search import SearchTool
from tools.sleep_control import SetSleepTool

CAP_M5_DEVICE = "m5_device"


@dataclass
class ToolFactory:
    """環境capabilityに応じて利用可能なツール集合を構築する。"""

    tts: object | None
    get_pipelines: Callable[[], list]

    def __post_init__(self) -> None:
        embeddings = llm_config.embeddings
        rerank = llm_config.rerank
        self.memory_store = MemoryStore(
            character_data_path("memory.json"),
            history_file=character_data_path("history.json"),
            embedding_model=embeddings.model,
            embedding_base_url=embeddings.base_url or llm_config.base_url,
            embedding_api_key=embeddings.api_key or llm_config.api_key,
            embedding_dimensions=embeddings.dimensions,
            embedding_cache_file="data/.cache/embeddings.json",
            bm25_weight=embeddings.bm25_weight,
            embedding_weight=embeddings.embedding_weight,
            rerank_weight=embeddings.rerank_weight,
            rerank_model=rerank.model,
            rerank_base_url=rerank.base_url or llm_config.base_url,
            rerank_api_key=rerank.api_key or llm_config.api_key,
            rerank_top_n=rerank.top_n,
        )
        self.notification_store = NotificationStore(settings.notification_file)
        self.skill_provider = SkillProvider(
            memory_store=self.memory_store,
            tool_guide=prompt_config.tool_guide,
        )

    def create_registry(self, capabilities: set[str] | None = None) -> ToolRegistry:
        caps = capabilities or set()
        registry = ToolRegistry()

        # 共通ツール
        registry.register(SaveMemoryTool(self.memory_store))
        registry.register(SearchMemoryTool(self.memory_store))
        registry.register(DeleteMemoryTool(self.memory_store))
        registry.register(SearchTool())

        registry.register(SetNotificationTool(self.notification_store))
        registry.register(ListNotificationsTool(self.notification_store))
        registry.register(DeleteNotificationTool(self.notification_store))

        if self.tts is not None:
            from tools.voice_control import SetVolumeTool

            registry.register(SetVolumeTool(self.tts))

        # M5StickS3依存ツール
        if CAP_M5_DEVICE in caps:
            registry.register(SetSleepTool(self.get_pipelines))
            registry.register(DisplayTextTool(self.get_pipelines))
            registry.register(DisplayImageTool(self.get_pipelines))

        return registry

    @staticmethod
    def register_subagent_tools(registry: ToolRegistry, job_manager: object) -> None:
        from tools.subagent_research import (
            GetSubAgentJobTool,
            ListSubAgentJobsTool,
            RunSubAgentResearchTool,
        )

        registry.register(RunSubAgentResearchTool(job_manager))
        registry.register(ListSubAgentJobsTool(job_manager))
        registry.register(GetSubAgentJobTool(job_manager))
