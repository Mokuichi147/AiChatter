import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock

from ai_chatter.tools.base import ToolBase, ToolResult
from ai_chatter._paths import SERVER_ROOT

logger = logging.getLogger(__name__)

VALID_TYPES = {"short_term", "long_term"}
VALID_STATUSES = {"active", "completed", "paused"}


class GoalStore:
    """JSONファイルベースの目標永続ストア。"""

    def __init__(self, file_path: str, seed_goals: list[str] | None = None) -> None:
        self._path = Path(file_path)
        if not self._path.is_absolute():
            self._path = SERVER_ROOT / file_path
        self._goals: dict[str, dict] = {}
        self._lock = Lock()
        self._load()

        # 初回起動時のみシード目標を投入
        if not self._goals and seed_goals:
            for desc in seed_goals:
                self.add(desc, goal_type="long_term")
            logger.info(f"シード目標を投入: {len(seed_goals)}件")

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._goals = data if isinstance(data, dict) else {}
            logger.info(f"目標を読み込み: {len(self._goals)}件 ({self._path})")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"目標の読み込み失敗: {e}")
            self._goals = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._path.write_text(
                json.dumps(self._goals, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning(f"目標の保存失敗: {e}")

    def add(
        self,
        description: str,
        goal_type: str = "short_term",
        progress: str = "",
    ) -> str:
        goal_id = uuid.uuid4().hex[:8]
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = {
            "description": description,
            "type": goal_type if goal_type in VALID_TYPES else "short_term",
            "status": "active",
            "created_at": now,
            "updated_at": now,
            "progress": progress,
        }
        with self._lock:
            self._goals[goal_id] = entry
            self._save()
        logger.info(f"目標を追加: id={goal_id} type={goal_type} desc={description}")
        return goal_id

    def update(
        self,
        goal_id: str,
        description: str | None = None,
        goal_type: str | None = None,
        status: str | None = None,
        progress: str | None = None,
    ) -> dict | None:
        with self._lock:
            entry = self._goals.get(goal_id)
            if entry is None:
                return None
            if description is not None:
                entry["description"] = description
            if goal_type is not None and goal_type in VALID_TYPES:
                entry["type"] = goal_type
            if status is not None and status in VALID_STATUSES:
                entry["status"] = status
            if progress is not None:
                entry["progress"] = progress
            entry["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            self._save()
        return entry

    def list_all(self, status_filter: str | None = None) -> dict[str, dict]:
        with self._lock:
            if status_filter and status_filter in VALID_STATUSES:
                return {
                    k: v for k, v in self._goals.items()
                    if v.get("status") == status_filter
                }
            return dict(self._goals)

    def get(self, goal_id: str) -> dict | None:
        with self._lock:
            return self._goals.get(goal_id)

    def remove(self, goal_id: str) -> bool:
        with self._lock:
            if goal_id in self._goals:
                del self._goals[goal_id]
                self._save()
                logger.info(f"目標を削除: id={goal_id}")
                return True
        return False

    def active_goals_summary(self) -> str:
        """自律思考ループ用: アクティブ目標の要約テキストを返す。"""
        with self._lock:
            active = [
                (gid, g) for gid, g in self._goals.items()
                if g.get("status") == "active"
            ]
        if not active:
            return ""
        lines = []
        for gid, g in active:
            line = f"- [{g['type']}] {g['description']}"
            if g.get("progress"):
                line += f" (進捗: {g['progress']})"
            lines.append(line)
        return "\n".join(lines)


class AddGoalTool(ToolBase):
    name = "add_goal"
    description = "新しい目標を追加します。短期・長期を指定できます。"
    input_schema = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "目標の内容",
            },
            "type": {
                "type": "string",
                "enum": ["short_term", "long_term"],
                "description": "目標の種別 (short_term: 短期, long_term: 長期)",
            },
            "progress": {
                "type": "string",
                "description": "現時点の進捗メモ (省略可)",
            },
        },
        "required": ["description"],
    }

    def __init__(self, store: GoalStore) -> None:
        self._store = store

    async def execute(self, **kwargs) -> ToolResult:
        description = kwargs.get("description", "").strip()
        if not description:
            return ToolResult(content="description は必須です。", is_error=True)
        goal_type = kwargs.get("type", "short_term")
        progress = kwargs.get("progress", "")
        goal_id = self._store.add(description, goal_type, progress)
        return ToolResult(
            content=f"目標を追加しました。ID: {goal_id}, 種別: {goal_type}, 内容: {description}"
        )


class UpdateGoalTool(ToolBase):
    name = "update_goal"
    description = "既存の目標を更新します。進捗の記録、ステータス変更、内容の修正に使います。"
    input_schema = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "更新する目標のID",
            },
            "description": {
                "type": "string",
                "description": "新しい目標内容 (変更する場合のみ)",
            },
            "type": {
                "type": "string",
                "enum": ["short_term", "long_term"],
                "description": "新しい種別 (変更する場合のみ)",
            },
            "status": {
                "type": "string",
                "enum": ["active", "completed", "paused"],
                "description": "新しいステータス",
            },
            "progress": {
                "type": "string",
                "description": "進捗メモの更新",
            },
        },
        "required": ["id"],
    }

    def __init__(self, store: GoalStore) -> None:
        self._store = store

    async def execute(self, **kwargs) -> ToolResult:
        goal_id = kwargs.get("id", "").strip()
        if not goal_id:
            return ToolResult(content="id は必須です。", is_error=True)
        result = self._store.update(
            goal_id,
            description=kwargs.get("description"),
            goal_type=kwargs.get("type"),
            status=kwargs.get("status"),
            progress=kwargs.get("progress"),
        )
        if result is None:
            return ToolResult(
                content=f"ID '{goal_id}' の目標は見つかりませんでした。",
                is_error=True,
            )
        return ToolResult(
            content=f"目標を更新しました。ID: {goal_id}, ステータス: {result['status']}, 内容: {result['description']}"
        )


class ListGoalsTool(ToolBase):
    name = "list_goals"
    description = "目標の一覧を取得します。ステータスで絞り込むこともできます。"
    input_schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["active", "completed", "paused"],
                "description": "絞り込むステータス (省略時は全件)",
            },
        },
    }

    def __init__(self, store: GoalStore) -> None:
        self._store = store

    async def execute(self, **kwargs) -> ToolResult:
        status_filter = kwargs.get("status")
        goals = self._store.list_all(status_filter)
        if not goals:
            label = f" (ステータス: {status_filter})" if status_filter else ""
            return ToolResult(content=f"目標はありません{label}。")
        lines = []
        for gid, g in goals.items():
            line = f"- ID: {gid}, [{g['type']}] {g['description']} (ステータス: {g['status']})"
            if g.get("progress"):
                line += f"\n  進捗: {g['progress']}"
            lines.append(line)
        return ToolResult(
            content=f"目標一覧 ({len(goals)}件):\n" + "\n".join(lines)
        )


class CompleteGoalTool(ToolBase):
    name = "complete_goal"
    description = "目標を完了にします。達成した目標に使ってください。"
    input_schema = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "完了にする目標のID",
            },
            "progress": {
                "type": "string",
                "description": "完了時の最終メモ (省略可)",
            },
        },
        "required": ["id"],
    }

    def __init__(self, store: GoalStore) -> None:
        self._store = store

    async def execute(self, **kwargs) -> ToolResult:
        goal_id = kwargs.get("id", "").strip()
        if not goal_id:
            return ToolResult(content="id は必須です。", is_error=True)
        progress = kwargs.get("progress")
        result = self._store.update(goal_id, status="completed", progress=progress)
        if result is None:
            return ToolResult(
                content=f"ID '{goal_id}' の目標は見つかりませんでした。",
                is_error=True,
            )
        return ToolResult(
            content=f"目標を完了にしました。ID: {goal_id}, 内容: {result['description']}"
        )


class DeleteGoalTool(ToolBase):
    name = "delete_goal"
    description = "目標を削除します。不要になった目標に使ってください。先にlist_goalsで確認してください。"
    input_schema = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "削除する目標のID",
            },
        },
        "required": ["id"],
    }

    def __init__(self, store: GoalStore) -> None:
        self._store = store

    async def execute(self, **kwargs) -> ToolResult:
        goal_id = kwargs.get("id", "").strip()
        if not goal_id:
            return ToolResult(content="id は必須です。", is_error=True)
        if self._store.remove(goal_id):
            return ToolResult(content=f"目標を削除しました。ID: {goal_id}")
        return ToolResult(
            content=f"ID '{goal_id}' の目標は見つかりませんでした。",
            is_error=True,
        )
