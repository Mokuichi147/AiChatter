from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


class ToolBase(ABC):
    name: str = ""
    description: str = ""
    input_schema: dict = field(default_factory=dict)

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        ...

    def to_openai_tool(self) -> dict:
        """OpenAI Responses API形式に変換する。"""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }

    def to_mcp_tool(self) -> dict:
        """MCP形式で出力する。"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
