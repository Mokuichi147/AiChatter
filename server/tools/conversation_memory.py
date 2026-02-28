import json
import logging
from datetime import datetime
from pathlib import Path

from tools.base import ToolBase, ToolResult

logger = logging.getLogger(__name__)


class MemoryStore:
    def __init__(self, file_path: str) -> None:
        self._path = Path(file_path)
        if not self._path.is_absolute():
            self._path = Path(__file__).parent.parent / self._path
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
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
        self._data[key] = {
            "content": content,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        self._save()

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            self._save()
            return True
        return False

    def search(self, query: str, after: str = "", before: str = "") -> list[dict]:
        query_lower = query.lower()
        after_dt = None
        before_dt = None
        if after:
            try:
                after_dt = datetime.strptime(after, "%Y-%m-%d")
            except ValueError:
                pass
        if before:
            try:
                before_dt = datetime.strptime(before, "%Y-%m-%d")
            except ValueError:
                pass

        results = []
        for key, entry in self._data.items():
            content = entry.get("content", "")
            created_at = entry.get("created_at", "")

            if query_lower not in key.lower() and query_lower not in content.lower():
                continue

            if (after_dt or before_dt) and created_at:
                try:
                    entry_dt = datetime.strptime(created_at[:10], "%Y-%m-%d")
                except ValueError:
                    continue
                if after_dt and entry_dt < after_dt:
                    continue
                if before_dt and entry_dt > before_dt:
                    continue

            results.append({
                "key": key,
                "content": content,
                "created_at": created_at,
            })
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
    description = "保存された記憶を検索します。日付で絞り込むこともできます。"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "検索クエリ",
            },
            "after": {
                "type": "string",
                "description": "この日付以降の記憶を検索 (YYYY-MM-DD形式、省略可)",
            },
            "before": {
                "type": "string",
                "description": "この日付以前の記憶を検索 (YYYY-MM-DD形式、省略可)",
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
        results = self._store.search(
            query, after=kwargs.get("after", ""), before=kwargs.get("before", ""),
        )
        if not results:
            return ToolResult(content="該当する記憶が見つかりませんでした。")
        formatted = json.dumps(results, ensure_ascii=False, indent=2)
        return ToolResult(content=f"検索結果:\n{formatted}")


class DeleteMemoryTool(ToolBase):
    name = "delete_memory"
    description = "保存された記憶を削除します。事前にsearch_memoryでキーを確認してください。"
    input_schema = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "削除する記憶のキー",
            },
        },
        "required": ["key"],
    }

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def execute(self, **kwargs) -> ToolResult:
        key = kwargs.get("key", "")
        if not key:
            return ToolResult(content="keyは必須です", is_error=True)
        if self._store.delete(key):
            logger.info(f"メモリ削除: {key}")
            return ToolResult(content=f"'{key}'を削除しました。")
        return ToolResult(
            content=f"キー '{key}' の記憶は見つかりませんでした。",
            is_error=True,
        )
