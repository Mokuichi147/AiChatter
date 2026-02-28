import json
import logging
from pathlib import Path

from config import settings
from tools.base import ToolBase, ToolResult

logger = logging.getLogger(__name__)


class MemoryStore:
    def __init__(self, file_path: str) -> None:
        self._path = Path(file_path)
        if not self._path.is_absolute():
            self._path = Path(__file__).parent.parent / self._path
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
                logger.info(f"メモリ読み込み: {len(self._data)}件 ({self._path})")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"メモリファイル読み込み失敗: {e}")
                self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save(self, key: str, content: str) -> None:
        self._data[key] = content
        self._save()

    def search(self, query: str) -> list[dict[str, str]]:
        query_lower = query.lower()
        results = []
        for key, content in self._data.items():
            if query_lower in key.lower() or query_lower in content.lower():
                results.append({"key": key, "content": content})
        return results


class SaveMemoryTool(ToolBase):
    name = "save_memory"
    description = "情報を記憶に保存します。ユーザーの名前・好み・重要な事実・約束事など、後で役立ちそうな情報を自主的に記録してください。"
    input_schema = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "記憶のキー（短い識別名）",
            },
            "content": {
                "type": "string",
                "description": "記憶する内容",
            },
        },
        "required": ["key", "content"],
    }

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def execute(self, **kwargs) -> ToolResult:
        key = kwargs.get("key", "")
        content = kwargs.get("content", "")
        if not key or not content:
            return ToolResult(content="keyとcontentは必須です", is_error=True)
        self._store.save(key, content)
        logger.info(f"メモリ保存: {key}")
        return ToolResult(content=f"'{key}'を記憶しました。")


class SearchMemoryTool(ToolBase):
    name = "search_memory"
    description = "保存された記憶を検索します。以前覚えた内容を思い出すときに使います。"
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

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        if not query:
            return ToolResult(content="queryは必須です", is_error=True)
        results = self._store.search(query)
        if not results:
            return ToolResult(content="該当する記憶が見つかりませんでした。")
        formatted = json.dumps(results, ensure_ascii=False, indent=2)
        return ToolResult(content=f"検索結果:\n{formatted}")
