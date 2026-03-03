import logging

import httpx

from config import settings
from tools.base import ToolBase, ToolResult

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class SearchTool(ToolBase):
    name = "web_search"
    description = "Webで最新情報を調べます。ニュース・天気・時事・価格など、変化しやすい情報が必要なときに使います。"
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
        query = kwargs.get("query", "")
        if not query:
            return ToolResult(content="queryは必須です", is_error=True)

        if not settings.tavily_api_key:
            return ToolResult(
                content="TAVILY_API_KEY が設定されていません。", is_error=True,
            )

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    TAVILY_SEARCH_URL,
                    json={
                        "api_key": settings.tavily_api_key,
                        "query": query,
                        "max_results": 5,
                        "include_answer": True,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Tavily APIエラー: {e}")
            return ToolResult(content=f"検索APIエラー: {e}", is_error=True)
        except Exception as e:
            logger.error(f"検索エラー: {e}", exc_info=True)
            return ToolResult(content=f"検索エラー: {e}", is_error=True)

        # 結果を整形
        parts = []
        answer = data.get("answer")
        if answer:
            parts.append(f"回答: {answer}")

        results = data.get("results", [])
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            parts.append(f"\n[{i}] {title}\n{content}\nURL: {url}")

        if not parts:
            return ToolResult(content="検索結果が見つかりませんでした。")

        return ToolResult(content="\n".join(parts))
