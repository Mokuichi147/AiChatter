from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ai_chatter.config import CharacterConfig, load_character
from ai_chatter._paths import SERVER_ROOT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CharacterEntry:
    """利用可能なキャラクター定義。"""

    character_id: str
    file_name: str
    file_path: str
    config: CharacterConfig


class CharacterCatalog:
    """キャラクターYAMLを読み込み、IDで参照できるように管理する。"""

    def __init__(self, directory: str, pattern: str = "character*.yaml") -> None:
        base = Path(directory)
        if not base.is_absolute():
            base = SERVER_ROOT / base
        self._directory = base
        self._pattern = pattern
        self._entries: dict[str, CharacterEntry] = {}

    @property
    def directory(self) -> Path:
        return self._directory

    def reload(self) -> None:
        self._entries = {}
        self._directory.mkdir(parents=True, exist_ok=True)

        for path in sorted(self._directory.glob(self._pattern)):
            if not path.is_file():
                continue
            # サンプル定義は公開対象にしない
            if path.name.endswith(".example.yaml"):
                continue
            self._register_path(path)

        logger.info(f"キャラクターカタログ読み込み: {len(self._entries)}件")

    def _register_path(self, path: Path, character_id: str | None = None) -> CharacterEntry:
        entry_id = (character_id or path.name).strip()
        if not entry_id:
            raise ValueError("character_idが空です")
        if "/" in entry_id or "\\" in entry_id:
            raise ValueError("character_idにパス区切り文字は使えません")
        if entry_id in self._entries:
            raise ValueError(f"character_idが重複しています: {entry_id}")

        config = load_character(str(path))
        entry = CharacterEntry(
            character_id=entry_id,
            file_name=path.name,
            file_path=str(path),
            config=config,
        )
        self._entries[entry_id] = entry
        return entry

    def register_file(self, file_path: str, character_id: str | None = None) -> CharacterEntry:
        """CLI用途: 任意のローカルファイルをカタログへ追加する。"""
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            candidate = self._directory / path
            if candidate.exists():
                path = candidate
            else:
                path = Path.cwd() / path
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"キャラクターファイルが見つかりません: {path}")
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("キャラクターファイルはYAML形式のみ対応です")

        # 既定IDはファイル名（パスは隠す）
        resolved_id = character_id or path.name
        if resolved_id in self._entries:
            # 同名が存在する場合は衝突回避
            stem = Path(resolved_id).stem
            suffix = Path(resolved_id).suffix or ".yaml"
            idx = 2
            while True:
                candidate = f"{stem}-{idx}{suffix}"
                if candidate not in self._entries:
                    resolved_id = candidate
                    break
                idx += 1

        return self._register_path(path.resolve(), character_id=resolved_id)

    def has(self, character_id: str) -> bool:
        return character_id in self._entries

    def get(self, character_id: str) -> CharacterEntry:
        try:
            return self._entries[character_id]
        except KeyError as e:
            raise KeyError(f"不明なcharacter_id: {character_id}") from e

    def list_entries(self) -> list[CharacterEntry]:
        return [self._entries[k] for k in sorted(self._entries.keys())]

    def default_character_id(self, preferred_file: str) -> str:
        """既定キャラクターIDを返す。"""
        if preferred_file and preferred_file in self._entries:
            return preferred_file

        preferred_name = Path(preferred_file).name if preferred_file else ""
        if preferred_name and preferred_name in self._entries:
            return preferred_name

        if self._entries:
            return sorted(self._entries.keys())[0]

        raise RuntimeError("利用可能なキャラクターがありません")
