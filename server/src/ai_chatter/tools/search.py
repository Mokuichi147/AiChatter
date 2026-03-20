import json
import logging
from datetime import datetime, date
from pathlib import Path
from threading import Lock

import httpx

from ai_chatter.config import settings
from ai_chatter.tools.base import ToolBase, ToolResult
from ai_chatter._paths import SERVER_ROOT

logger = logging.getLogger(__name__)

# --- エンジン定義 ---

TAVILY_URL = "https://api.tavily.com/search"
EXA_URL = "https://api.exa.ai/search"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

# 無料枠 (period_type, limit)
FREE_TIERS: dict[str, tuple[str, int]] = {
    "tavily": ("monthly", 1000),
    "exa": ("monthly", 1000),
    "brave": ("monthly", 2000),
}


# --- 使用量トラッカー ---

class SearchUsageTracker:
    """検索エンジンごとの使用量を追跡し、無料枠の残量を管理する。"""

    def __init__(self, file_path: str) -> None:
        self._path = Path(file_path)
        if not self._path.is_absolute():
            self._path = SERVER_ROOT / file_path
        self._data: dict[str, dict] = {}
        self._lock = Lock()
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            self._data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning(f"検索使用量の保存失敗: {e}")

    @staticmethod
    def _current_period(period_type: str) -> str:
        now = datetime.now()
        if period_type == "daily":
            return now.strftime("%Y-%m-%d")
        return now.strftime("%Y-%m")

    def get_count(self, engine: str) -> int:
        """現在の期間内の使用回数を返す。期間が変わっていたら0にリセット。"""
        period_type = FREE_TIERS.get(engine, ("monthly", 0))[0]
        current = self._current_period(period_type)
        with self._lock:
            entry = self._data.get(engine, {})
            if entry.get("period") != current:
                return 0
            return entry.get("count", 0)

    def remaining_ratio(self, engine: str) -> float:
        """無料枠の残り割合 (0.0〜1.0) を返す。"""
        _, limit = FREE_TIERS.get(engine, ("monthly", 0))
        if limit <= 0:
            return 0.0
        count = self.get_count(engine)
        return max(0.0, (limit - count) / limit)

    def increment(self, engine: str) -> None:
        """使用回数を1増やす。"""
        period_type = FREE_TIERS.get(engine, ("monthly", 0))[0]
        current = self._current_period(period_type)
        with self._lock:
            entry = self._data.get(engine, {})
            if entry.get("period") != current:
                entry = {"period": current, "count": 0}
            entry["count"] = entry.get("count", 0) + 1
            self._data[engine] = entry
            self._save()

    def is_within_free_tier(self, engine: str) -> bool:
        """無料枠内かどうかを返す。"""
        _, limit = FREE_TIERS.get(engine, ("monthly", 0))
        return self.get_count(engine) < limit


# --- 各エンジンの検索実装 ---

async def _search_tavily(query: str, max_results: int = 5) -> list[dict]:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            TAVILY_URL,
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": True,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    answer = data.get("answer")
    if answer:
        results.append({"title": "AI要約", "snippet": answer, "url": ""})
    for r in data.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "snippet": r.get("content", ""),
            "url": r.get("url", ""),
        })
    return results


async def _search_exa(query: str, max_results: int = 5) -> list[dict]:
    headers = {
        "Content-Type": "application/json",
        "x-api-key": settings.exa_api_key,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            EXA_URL,
            headers=headers,
            json={
                "query": query,
                "numResults": max_results,
                "contents": {"highlights": True},
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for r in data.get("results", []):
        highlights = r.get("highlights", [])
        snippet = highlights[0] if highlights else ""
        results.append({
            "title": r.get("title", ""),
            "snippet": snippet,
            "url": r.get("url", ""),
        })
    return results


async def _search_brave(query: str, max_results: int = 5) -> list[dict]:
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": settings.brave_search_api_key,
    }
    params = {"q": query, "count": min(max_results, 20)}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(BRAVE_URL, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

    results = []
    for r in data.get("web", {}).get("results", []):
        results.append({
            "title": r.get("title", ""),
            "snippet": r.get("description", ""),
            "url": r.get("url", ""),
        })
    return results


# --- エンジンディスパッチャー ---

# エンジン名 → (APIキー存在チェック関数, 検索関数)
_ENGINE_REGISTRY: dict[str, tuple] = {
    "tavily": (lambda: bool(settings.tavily_api_key), _search_tavily),
    "exa": (lambda: bool(settings.exa_api_key), _search_exa),
    "brave": (lambda: bool(settings.brave_search_api_key), _search_brave),
}


def _select_engine(tracker: SearchUsageTracker) -> list[str]:
    """利用可能なエンジンを無料枠の残り割合が高い順に返す。"""
    candidates = []
    for name, (has_key, _) in _ENGINE_REGISTRY.items():
        if not has_key():
            continue
        ratio = tracker.remaining_ratio(name)
        candidates.append((name, ratio))

    # 無料枠の残り割合が高い順 → 同率なら名前順
    candidates.sort(key=lambda x: (-x[1], x[0]))
    return [name for name, _ in candidates]


# --- ツール本体 ---

# モジュールレベルのトラッカー（SearchToolインスタンス間で共有）
_usage_tracker: SearchUsageTracker | None = None


def _get_tracker() -> SearchUsageTracker:
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = SearchUsageTracker("data/.cache/search_usage.json")
    return _usage_tracker


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

        tracker = _get_tracker()
        engines = _select_engine(tracker)

        if not engines:
            return ToolResult(
                content="検索APIキーが設定されていません。TAVILY_API_KEY, EXA_API_KEY, BRAVE_SEARCH_API_KEY のいずれかを設定してください。",
                is_error=True,
            )

        last_error = None
        for engine_name in engines:
            _, search_fn = _ENGINE_REGISTRY[engine_name]
            try:
                logger.info(f"検索実行: engine={engine_name} query='{query}'")
                results = await search_fn(query)
                tracker.increment(engine_name)

                if not results:
                    continue

                parts = []
                for i, r in enumerate(results, 1):
                    title = r.get("title", "")
                    snippet = r.get("snippet", "")
                    url = r.get("url", "")
                    if url:
                        parts.append(f"\n[{i}] {title}\n{snippet}\nURL: {url}")
                    else:
                        parts.append(f"\n{title}: {snippet}")

                return ToolResult(content="\n".join(parts))

            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(f"検索APIエラー ({engine_name}): {e}")
                if e.response.status_code == 429:
                    # レート制限 → 次のエンジンへ
                    continue
                # その他のHTTPエラーも次のエンジンで再試行
                continue
            except Exception as e:
                last_error = e
                logger.warning(f"検索エラー ({engine_name}): {e}", exc_info=True)
                continue

        error_msg = f"すべての検索エンジンで失敗しました: {last_error}" if last_error else "検索結果が見つかりませんでした。"
        return ToolResult(content=error_msg, is_error=bool(last_error))
