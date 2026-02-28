import json
import logging

from tools.base import ToolBase, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolBase] = {}

    def register(self, tool: ToolBase) -> None:
        self._tools[tool.name] = tool
        logger.info(f"ツール登録: {tool.name}")

    def get(self, name: str) -> ToolBase | None:
        return self._tools.get(name)

    async def execute(self, name: str, arguments: str | dict) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(content=f"不明なツール: {name}", is_error=True)

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return ToolResult(
                    content=f"引数のJSONパースに失敗: {arguments}", is_error=True
                )

        try:
            return await tool.execute(**arguments)
        except Exception as e:
            logger.error(f"ツール実行エラー ({name}): {e}", exc_info=True)
            return ToolResult(content=f"ツール実行エラー: {e}", is_error=True)

    def to_openai_tools(self) -> list[dict]:
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def to_mcp_tools(self) -> list[dict]:
        return [tool.to_mcp_tool() for tool in self._tools.values()]

    @property
    def is_empty(self) -> bool:
        return len(self._tools) == 0
