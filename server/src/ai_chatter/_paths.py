from __future__ import annotations

import json
import logging
from pathlib import Path

# server/ ディレクトリ（configs/, data/, voices/ の親）
SERVER_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)


def history_path() -> Path:
    """現在のキャラクターに対応する history.json のパスを返す。"""
    from ai_chatter.config import character_data_path

    path = Path(character_data_path("history.json"))
    if not path.is_absolute():
        path = SERVER_ROOT / path
    return path


def save_history(new_entries: list[dict]) -> None:
    """新しい会話エントリを history.json に追記する。"""
    path = history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing: list = []
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        existing.extend(new_entries)
        path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning(f"会話履歴の保存失敗: {e}")
