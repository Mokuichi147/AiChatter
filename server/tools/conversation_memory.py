import json
import logging
from datetime import datetime
from pathlib import Path

from rank_bm25 import BM25Okapi
from sudachipy import Dictionary, SplitMode
from tools.base import ToolBase, ToolResult

logger = logging.getLogger(__name__)

_sudachi_tokenizer = Dictionary().create()
_STOP_POS = {"助詞", "助動詞", "記号", "空白", "補助記号"}


class MemoryStore:
    def __init__(self, file_path: str, history_file: str = "") -> None:
        self._path = Path(file_path)
        if not self._path.is_absolute():
            self._path = Path(__file__).parent.parent / self._path
        self._history_path: Path | None = None
        if history_file:
            hp = Path(history_file)
            if not hp.is_absolute():
                hp = Path(__file__).parent.parent / hp
            self._history_path = hp
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

    def save(self, key: str, content: str, *, auto: bool = False) -> None:
        self._data[key] = {
            "content": content,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "auto": auto,
        }
        self._save()

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            self._save()
            return True
        return False

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        morphemes = _sudachi_tokenizer.tokenize(text, SplitMode.C)
        return [
            m.normalized_form()
            for m in morphemes
            if m.part_of_speech()[0] not in _STOP_POS
        ]

    def _load_history_entries(
        self, after_dt: datetime | None, before_dt: datetime | None,
    ) -> list[tuple[str, dict]]:
        """history.jsonからuser/assistantペアを読み込み、候補として返す。"""
        if not self._history_path or not self._history_path.exists():
            return []
        try:
            entries: list[dict] = json.loads(
                self._history_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"history.json読み込み失敗: {e}")
            return []

        candidates: list[tuple[str, dict]] = []
        i = 0
        while i < len(entries):
            entry = entries[i]
            # user + assistant のペアを結合
            if entry.get("role") == "user" and i + 1 < len(entries) and entries[i + 1].get("role") == "assistant":
                user_entry = entry
                asst_entry = entries[i + 1]
                created_at = user_entry.get("created_at", "")
                content = f"ユーザー: {user_entry.get('content', '')}\nアシスタント: {asst_entry.get('content', '')}"
                i += 2
            else:
                created_at = entry.get("created_at", "")
                content = f"{entry.get('role', '')}: {entry.get('content', '')}"
                i += 1

            # 日付フィルタ
            if (after_dt or before_dt) and created_at:
                try:
                    entry_dt = datetime.strptime(created_at[:10], "%Y-%m-%d")
                except ValueError:
                    continue
                if after_dt and entry_dt < after_dt:
                    continue
                if before_dt and entry_dt > before_dt:
                    continue

            key = f"history_{created_at.replace(' ', '_').replace(':', '')}"
            candidates.append((key, {"content": content, "created_at": created_at}))
        return candidates

    def search(
        self, query: str, after: str = "", before: str = "", include_auto: bool = True,
    ) -> list[dict]:
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

        # memory.jsonから候補を絞る（auto=Trueのエントリは除外）
        candidates: list[tuple[str, dict]] = []
        for key, entry in self._data.items():
            if entry.get("auto", False):
                continue
            created_at = entry.get("created_at", "")
            if (after_dt or before_dt) and created_at:
                try:
                    entry_dt = datetime.strptime(created_at[:10], "%Y-%m-%d")
                except ValueError:
                    continue
                if after_dt and entry_dt < after_dt:
                    continue
                if before_dt and entry_dt > before_dt:
                    continue
            candidates.append((key, entry))

        # include_autoの場合、history.jsonの会話履歴も検索対象に追加
        if include_auto:
            candidates.extend(self._load_history_entries(after_dt, before_dt))

        if not candidates:
            return []

        # BM25コーパス構築
        corpus_tokens = []
        for key, entry in candidates:
            text = key + " " + entry.get("content", "")
            corpus_tokens.append(self._tokenize(text))

        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = self._tokenize(query)
        scores = bm25.get_scores(query_tokens) if query_tokens else [0.0] * len(candidates)

        # 部分文字列一致ボーナス
        query_lower = query.lower()
        SUBSTRING_BONUS = 5.0

        results = []
        for i, (key, entry) in enumerate(candidates):
            score = float(scores[i])
            content = entry.get("content", "")
            if query_lower in key.lower() or query_lower in content.lower():
                score += SUBSTRING_BONUS
            if score > 0:
                results.append({
                    "key": key,
                    "content": content,
                    "created_at": entry.get("created_at", ""),
                    "score": round(score, 3),
                })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:10]


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
    description = "保存された記憶や過去の会話履歴を検索します。日付で絞り込んだり、include_autoで会話履歴を除外できます。"
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
            "include_auto": {
                "type": "boolean",
                "description": "自動記録（会話履歴など）を含めるか (デフォルト: true)",
                "default": True,
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
        include_auto = kwargs.get("include_auto", True)
        results = self._store.search(
            query,
            after=kwargs.get("after", ""),
            before=kwargs.get("before", ""),
            include_auto=include_auto,
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
