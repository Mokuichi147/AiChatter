import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi
from sudachipy import Dictionary, SplitMode
from tools.base import ToolBase, ToolResult

logger = logging.getLogger(__name__)

_sudachi_tokenizer = Dictionary().create()
_STOP_POS = {"助詞", "助動詞", "記号", "空白", "補助記号"}


class MemoryStore:
    def __init__(
        self,
        file_path: str,
        history_file: str = "",
        *,
        embedding_model: str = "",
        embedding_base_url: str = "",
        embedding_api_key: str = "",
        embedding_dimensions: int = 0,
        embedding_cache_file: str = "",
        bm25_weight: float = 0.4,
        embedding_weight: float = 0.3,
        rerank_weight: float = 0.3,
        rerank_model: str = "",
        rerank_base_url: str = "",
        rerank_api_key: str = "",
        rerank_top_n: int = 20,
    ) -> None:
        self._path = Path(file_path)
        if not self._path.is_absolute():
            self._path = Path(__file__).parent.parent / self._path
        self._history_path: Path | None = None
        if history_file:
            hp = Path(history_file)
            if not hp.is_absolute():
                hp = Path(__file__).parent.parent / hp
            self._history_path = hp
        self._embedding_cache_path: Path | None = None
        if embedding_cache_file:
            cp = Path(embedding_cache_file)
            if not cp.is_absolute():
                cp = Path(__file__).parent.parent / cp
            self._embedding_cache_path = cp
        self._embedding_model = embedding_model.strip()
        self._embedding_dimensions = max(0, int(embedding_dimensions))
        self._bm25_weight = max(0.0, float(bm25_weight))
        self._embedding_weight = max(0.0, float(embedding_weight))
        self._rerank_weight = max(0.0, float(rerank_weight))
        self._embedding_client: AsyncOpenAI | None = None
        self._embedding_cache: dict[str, list[float]] = {}
        if self._embedding_model:
            self._embedding_client = AsyncOpenAI(
                api_key=embedding_api_key or "no-key",
                base_url=embedding_base_url or None,
            )
            self._load_embedding_cache()
        self._rerank_model = rerank_model.strip()
        self._rerank_base_url = rerank_base_url.strip().rstrip("/")
        self._rerank_api_key = rerank_api_key.strip()
        self._rerank_top_n = max(1, int(rerank_top_n))
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

    def _load_embedding_cache(self) -> None:
        if not self._embedding_cache_path or not self._embedding_cache_path.exists():
            return
        try:
            raw = json.loads(self._embedding_cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Embeddingキャッシュ読み込み失敗: {e}")
            return

        vectors = raw.get("vectors", {})
        if not isinstance(vectors, dict):
            return

        loaded: dict[str, list[float]] = {}
        for key, value in vectors.items():
            if not isinstance(key, str) or not isinstance(value, list):
                continue
            try:
                loaded[key] = [float(v) for v in value]
            except (TypeError, ValueError):
                continue
        self._embedding_cache = loaded
        logger.info(
            f"Embeddingキャッシュ読み込み: {len(self._embedding_cache)}件 ({self._embedding_cache_path})"
        )

    def _save_embedding_cache(self) -> None:
        if not self._embedding_cache_path:
            return
        try:
            self._embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model": self._embedding_model,
                "dimensions": self._embedding_dimensions,
                "vectors": self._embedding_cache,
            }
            self._embedding_cache_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning(f"Embeddingキャッシュ保存失敗: {e}")

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

    @staticmethod
    def _normalize_scores(scores: list[float]) -> list[float]:
        if not scores:
            return []
        mn = min(scores)
        mx = max(scores)
        if mx == mn:
            return [1.0 if mx > 0 else 0.0 for _ in scores]
        return [(s - mn) / (mx - mn) for s in scores]

    def _embedding_cache_key(self, text: str) -> str:
        payload = f"{self._embedding_model}\n{self._embedding_dimensions}\n{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def _embedding_similarity_scores(
        self, query: str, documents: list[str],
    ) -> list[float]:
        if not self._embedding_client or not self._embedding_model:
            return [0.0] * len(documents)
        if not documents:
            return []

        texts = [query] + documents
        keys = [self._embedding_cache_key(text) for text in texts]

        missing_texts: list[str] = []
        missing_keys: list[str] = []
        for key, text in zip(keys, texts):
            if key not in self._embedding_cache:
                missing_keys.append(key)
                missing_texts.append(text)

        if missing_texts:
            kwargs: dict = {
                "model": self._embedding_model,
                "input": missing_texts,
            }
            if self._embedding_dimensions > 0:
                kwargs["dimensions"] = self._embedding_dimensions

            try:
                response = await self._embedding_client.embeddings.create(**kwargs)
            except Exception as e:
                logger.warning(f"Embedding検索をスキップします: {e}")
                return [0.0] * len(documents)

            if len(response.data) != len(missing_texts):
                logger.warning(
                    "Embedding応答件数が不正です: expected=%s got=%s",
                    len(missing_texts),
                    len(response.data),
                )
                return [0.0] * len(documents)

            for key, item in zip(missing_keys, response.data):
                self._embedding_cache[key] = [float(v) for v in item.embedding]

            self._save_embedding_cache()

        vectors = []
        for key in keys:
            vec = self._embedding_cache.get(key)
            if vec is None:
                return [0.0] * len(documents)
            vectors.append(np.asarray(vec, dtype=np.float32))

        query_vec = vectors[0]
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm == 0.0:
            return [0.0] * len(documents)

        similarities: list[float] = []
        for doc_vec in vectors[1:]:
            doc_norm = float(np.linalg.norm(doc_vec))
            if doc_norm == 0.0:
                similarities.append(0.0)
                continue
            sim = float(np.dot(query_vec, doc_vec) / (query_norm * doc_norm))
            similarities.append(sim)
        return similarities

    async def _rerank_scores(
        self, query: str, documents: list[str],
    ) -> list[float]:
        if not self._rerank_model or not self._rerank_base_url or not documents:
            return [0.0] * len(documents)

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._rerank_api_key:
            headers["Authorization"] = f"Bearer {self._rerank_api_key}"

        payload = {
            "model": self._rerank_model,
            "query": query,
            "documents": documents,
            "top_n": min(self._rerank_top_n, len(documents)),
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self._rerank_base_url}/v1/rerank",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning(f"リランキングをスキップします: {e}")
            return [0.0] * len(documents)

        scores = [0.0] * len(documents)
        for item in data.get("results", []):
            idx = item.get("index")
            score = item.get("relevance_score", 0.0)
            if isinstance(idx, int) and 0 <= idx < len(documents):
                scores[idx] = float(score)
        return scores

    def _collect_candidates(
        self, after_dt: datetime | None, before_dt: datetime | None, include_auto: bool,
    ) -> list[tuple[str, dict]]:
        candidates: list[tuple[str, dict]] = []

        # memory.jsonから候補を絞る（auto=Trueのエントリは除外）
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

        return candidates

    async def search(
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

        candidates = self._collect_candidates(after_dt, before_dt, include_auto)
        if not candidates:
            return []

        texts = [f"{key} {entry.get('content', '')}" for key, entry in candidates]
        corpus_tokens = [self._tokenize(text) for text in texts]
        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = self._tokenize(query)
        bm25_scores = (
            bm25.get_scores(query_tokens) if query_tokens else [0.0] * len(candidates)
        )

        # 部分文字列一致はBM25側へボーナスとして加算
        query_lower = query.lower()
        substring_bonus = 5.0
        lexical_raw_scores: list[float] = []
        for i, (key, entry) in enumerate(candidates):
            score = float(bm25_scores[i])
            content = entry.get("content", "")
            if query_lower in key.lower() or query_lower in content.lower():
                score += substring_bonus
            lexical_raw_scores.append(score)

        bm25_norm_scores = self._normalize_scores(lexical_raw_scores)

        embedding_raw_scores = await self._embedding_similarity_scores(query, texts)
        embedding_norm_scores = self._normalize_scores(embedding_raw_scores)

        rerank_raw_scores = await self._rerank_scores(query, texts)
        rerank_norm_scores = self._normalize_scores(rerank_raw_scores)

        use_embedding = self._embedding_client is not None and any(
            s > 0 for s in embedding_norm_scores
        )
        use_rerank = self._rerank_model and any(
            s > 0 for s in rerank_norm_scores
        )

        w_bm25 = self._bm25_weight
        w_emb = self._embedding_weight if use_embedding else 0.0
        w_rerank = self._rerank_weight if use_rerank else 0.0
        total_weight = w_bm25 + w_emb + w_rerank
        if total_weight <= 0:
            w_bm25 = 1.0
            w_emb = 0.0
            w_rerank = 0.0
        else:
            w_bm25 /= total_weight
            w_emb /= total_weight
            w_rerank /= total_weight

        results = []
        for i, (key, entry) in enumerate(candidates):
            score = (
                bm25_norm_scores[i] * w_bm25
                + embedding_norm_scores[i] * w_emb
                + rerank_norm_scores[i] * w_rerank
            )
            content = entry.get("content", "")
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
    description = "会話の継続に役立つ情報を記憶として保存します。ユーザーの名前・好み・重要な事実・約束など、後で参照する価値が高い内容に使います。"
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
    description = "保存された記憶や過去の会話履歴を検索します。日付での絞り込みや、include_autoによる会話履歴の除外が可能です。"
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
        results = await self._store.search(
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
    description = "保存済みの記憶を削除します。削除前にsearch_memoryでキーを確認してください。"
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
