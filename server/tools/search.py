from tools.base import ToolBase, ToolResult


class SearchTool(ToolBase):
    name = "web_search"
    description = "Webで情報を検索します。"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "検索クエリ",
            },
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(content="検索機能は現在未実装です。")
