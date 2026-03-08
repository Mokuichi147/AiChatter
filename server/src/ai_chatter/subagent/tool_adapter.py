import logging

from ai_chatter.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_FORCED_DENYLIST = {
    "set_sleep",
    "set_volume",
    "set_notification",
    "list_notifications",
    "delete_notification",
    "run_subagent_research",
    "list_subagent_jobs",
    "get_subagent_job",
}


class SubAgentToolAdapter:
    def __init__(self, registry: ToolRegistry, denylist: str = "") -> None:
        self._registry = registry
        self._denylist_from_env = {
            name.strip() for name in denylist.split(",") if name.strip()
        }

    def _is_allowed(self, name: str) -> bool:
        # 再帰呼び出し防止: サブエージェント管理ツールは設定値に関係なく常時除外
        return (
            name not in _FORCED_DENYLIST
            and name not in self._denylist_from_env
        )

    def to_openai_tools(self) -> list[dict]:
        tools: list[dict] = []
        for tool in self._registry.to_openai_tools():
            name = tool.get("name", "")
            if name and self._is_allowed(name):
                tools.append(tool)
        return tools

    async def execute(self, name: str, arguments: str | dict) -> str:
        if not self._is_allowed(name):
            return f"ツール '{name}' はサブエージェントでは使用できません。"

        result = await self._registry.execute(name, arguments)
        if result.is_error:
            logger.warning(f"サブエージェントツールエラー ({name}): {result.content}")
            return f"[ERROR] {result.content}"
        return result.content

    @property
    def denied_tools(self) -> list[str]:
        return sorted(_FORCED_DENYLIST | self._denylist_from_env)
