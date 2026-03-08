from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ECAPA-TDNN が期待するサンプルレート
_SAMPLE_RATE = 16000

_UNKNOWN_PREFIX = "不明な人"
_LABEL_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class SpeakerIdentifier:
    """WavLM-XVectorベースの話者識別モジュール（transformers使用）。"""

    def __init__(self, data_path: str, similarity_threshold: float = 0.65) -> None:
        self._data_path = Path(data_path)
        self._similarity_threshold = similarity_threshold
        self._speakers: dict[str, list[list[float]]] = {}  # name -> [embedding, ...]
        self._encoder = None
        # 未登録者の一時クラスタ: label -> list[embedding]
        self._unknown_clusters: dict[str, list[list[float]]] = {}
        # 発話ごとのembedding: utterance_id -> embedding
        self._utterance_embeddings: dict[str, list[float]] = {}
        self._load_data()

    def _load_encoder(self):
        if self._encoder is None:
            from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

            model_name = "microsoft/wavlm-base-plus-sv"
            self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self._encoder = WavLMForXVector.from_pretrained(model_name)
            self._encoder.eval()
            logger.info("WavLM-XVector ロード完了")
        return self._encoder

    def _load_data(self) -> None:
        if not self._data_path.exists():
            self._speakers = {}
            self._utterance_embeddings = {}
            return
        try:
            data = json.loads(self._data_path.read_text(encoding="utf-8"))
            self._speakers = data.get("speakers", {})
            self._utterance_embeddings = data.get("utterance_embeddings", {})
            logger.info(f"話者データ読み込み: {len(self._speakers)}人 ({self._data_path})")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"話者データ読み込み失敗: {e}")
            self._speakers = {}
            self._utterance_embeddings = {}

    def _save_data(self) -> None:
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "speakers": self._speakers,
            "utterance_embeddings": self._utterance_embeddings,
        }
        self._data_path.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )

    def store_utterance_embedding(self, utterance_id: str, embedding: list[float]) -> None:
        """発話IDに紐づけてembeddingを保存する。"""
        self._utterance_embeddings[utterance_id] = embedding
        self._save_data()

    def get_utterance_embedding(self, utterance_id: str) -> list[float] | None:
        """発話IDからembeddingを取得する。"""
        return self._utterance_embeddings.get(utterance_id)

    def _pcm_to_float(self, pcm_bytes: bytes) -> np.ndarray:
        """16bit PCM (16kHz mono) を float32 配列に変換する。"""
        return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def compute_embedding(self, pcm_bytes: bytes) -> np.ndarray:
        """音声PCMから話者embedding (WavLM-XVector 512次元) を計算する。"""
        import torch

        model = self._load_encoder()
        wav = self._pcm_to_float(pcm_bytes)

        # 短い音声はパディング（最低1秒）
        min_samples = _SAMPLE_RATE
        if len(wav) < min_samples:
            padded = np.zeros(min_samples, dtype=np.float32)
            padded[: len(wav)] = wav
            wav = padded

        inputs = self._feature_extractor(
            wav, sampling_rate=_SAMPLE_RATE, return_tensors="pt"
        )
        with torch.no_grad():
            embeddings = model(**inputs).embeddings
        return embeddings.squeeze().cpu().numpy()

    def enroll(self, name: str, pcm_bytes: bytes) -> dict:
        """話者を登録する。同名で追加登録すると精度が向上する。"""
        embedding = self.compute_embedding(pcm_bytes)
        emb_list = embedding.tolist()
        if name in self._speakers:
            self._speakers[name].append(emb_list)
        else:
            self._speakers[name] = [emb_list]
        self._save_data()
        logger.info(f"話者登録: {name} (embedding数: {len(self._speakers[name])})")
        return {"name": name, "embedding_count": len(self._speakers[name])}

    def _next_unknown_label(self) -> str:
        """未使用の一時ラベル（不明な人A, B, ...）を返す。"""
        for ch in _LABEL_CHARS:
            label = f"{_UNKNOWN_PREFIX}{ch}"
            if label not in self._unknown_clusters:
                return label
        i = len(self._unknown_clusters) + 1
        return f"{_UNKNOWN_PREFIX}{i}"

    def _match_unknown_cluster(self, embedding: np.ndarray) -> tuple[str, float]:
        """未登録者クラスタとの照合。一致するクラスタがあればラベルを返す。"""
        best_label = ""
        best_score = 0.0
        for label, embeddings in self._unknown_clusters.items():
            centroid = np.mean(embeddings, axis=0)
            score = self._cosine_similarity(embedding, centroid)
            if score > best_score:
                best_score = score
                best_label = label
        return (best_label, best_score)

    def identify(self, pcm_bytes: bytes) -> tuple[str, float]:
        """話者を識別する。登録済み話者→未登録者クラスタ→新規クラスタ作成の順で判定。"""
        embedding = self.compute_embedding(pcm_bytes)

        # 1. 登録済み話者と照合
        best_name = ""
        best_score = 0.0
        for name, embeddings in self._speakers.items():
            centroid = np.mean(embeddings, axis=0)
            score = self._cosine_similarity(embedding, centroid)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= self._similarity_threshold:
            return (best_name, best_score)

        # 2. 未登録者クラスタと照合
        if self._unknown_clusters:
            cluster_label, cluster_score = self._match_unknown_cluster(embedding)
            if cluster_score >= self._similarity_threshold:
                self._unknown_clusters[cluster_label].append(embedding.tolist())
                return (cluster_label, cluster_score)

        # 3. 新規クラスタ作成
        label = self._next_unknown_label()
        self._unknown_clusters[label] = [embedding.tolist()]
        logger.info(f"未登録話者の新規クラスタ作成: {label}")
        return (label, 0.0)

    def identify_from_embedding(self, embedding: np.ndarray | list[float]) -> tuple[str, float]:
        """事前計算済みのembeddingから登録済み話者を識別する。"""
        if isinstance(embedding, list):
            embedding = np.array(embedding)

        best_name = ""
        best_score = 0.0

        for name, embeddings in self._speakers.items():
            centroid = np.mean(embeddings, axis=0)
            score = self._cosine_similarity(embedding, centroid)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < self._similarity_threshold:
            return ("", best_score)

        return (best_name, best_score)

    def unenroll(self, name: str) -> bool:
        """話者の登録を解除する。"""
        if name not in self._speakers:
            return False
        del self._speakers[name]
        self._save_data()
        logger.info(f"話者登録解除: {name}")
        return True

    def list_speakers(self) -> list[str]:
        """登録済み話者の一覧を返す。"""
        return list(self._speakers.keys())

    def merge_speakers(self, source: str, target: str) -> dict:
        """sourceの声紋をtargetに統合し、sourceを削除する。"""
        if source not in self._speakers:
            return {"error": f"話者 '{source}' が見つかりません"}
        if target not in self._speakers:
            return {"error": f"話者 '{target}' が見つかりません"}

        self._speakers[target].extend(self._speakers[source])
        del self._speakers[source]
        self._save_data()
        logger.info(f"話者マージ: {source} → {target} (embedding数: {len(self._speakers[target])})")
        return {
            "merged": True,
            "source": source,
            "target": target,
            "embedding_count": len(self._speakers[target]),
        }

    def retroactive_update(self, name: str, history: list[dict]) -> int:
        """新規登録した話者のembeddingで、履歴中の未登録話者を遡って更新する。"""
        if name not in self._speakers:
            return 0

        updated = 0
        updated_from: set[str] = set()
        for entry in history:
            speaker = entry.get("speaker", "")
            if not speaker or not speaker.startswith(_UNKNOWN_PREFIX):
                continue
            uid = entry.get("utterance_id")
            if not uid:
                continue
            emb = self.get_utterance_embedding(uid)
            if not emb:
                continue
            identified, score = self.identify_from_embedding(emb)
            if identified == name:
                updated_from.add(speaker)
                entry["speaker"] = name
                updated += 1

        # 更新された一時クラスタを削除
        for old_label in updated_from:
            self._unknown_clusters.pop(old_label, None)

        if updated:
            logger.info(f"履歴遡及更新: {updated}件を '{name}' に更新")
        return updated

    def retroactive_merge(self, source: str, target: str, history: list[dict]) -> int:
        """履歴中のspeakerフィールドをsource→targetに一括置換する。"""
        updated = 0
        for entry in history:
            if entry.get("speaker") == source:
                entry["speaker"] = target
                updated += 1
        if updated:
            logger.info(f"履歴マージ更新: {updated}件を '{source}' → '{target}' に更新")
        return updated
