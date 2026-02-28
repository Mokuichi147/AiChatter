import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock

from tools.base import ToolBase, ToolResult

logger = logging.getLogger(__name__)


class NotificationStore:
    """JSONファイルベースの通知永続ストア。"""

    def __init__(self, file_path: str) -> None:
        self._path = Path(file_path)
        if not self._path.is_absolute():
            self._path = Path(__file__).parent.parent / file_path
        self._notifications: list[dict] = []
        self._lock = Lock()
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._notifications = data if isinstance(data, list) else []
            logger.info(f"通知を読み込み: {len(self._notifications)}件 ({self._path})")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"通知の読み込み失敗: {e}")
            self._notifications = []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._path.write_text(
                json.dumps(self._notifications, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning(f"通知の保存失敗: {e}")

    def add(self, dt: datetime, message: str) -> str:
        """通知を追加し、一意IDを返す。"""
        notification_id = uuid.uuid4().hex[:8]
        entry = {
            "id": notification_id,
            "datetime": dt.strftime("%Y-%m-%d %H:%M"),
            "message": message,
        }
        with self._lock:
            self._notifications.append(entry)
            self._save()
        logger.info(f"通知を追加: id={notification_id} dt={entry['datetime']} msg={message}")
        return notification_id

    def list_all(self) -> list[dict]:
        """登録済みの全通知を返す。"""
        with self._lock:
            return list(self._notifications)

    def remove(self, notification_id: str) -> bool:
        """指定IDの通知を削除する。成功時True。"""
        with self._lock:
            before = len(self._notifications)
            self._notifications = [
                n for n in self._notifications if n.get("id") != notification_id
            ]
            if len(self._notifications) < before:
                self._save()
                logger.info(f"通知を削除: id={notification_id}")
                return True
        return False

    def pop_due(self) -> list[dict]:
        """現在時刻を過ぎた通知を取り出して削除する。"""
        now = datetime.now()
        due = []
        remaining = []
        with self._lock:
            for n in self._notifications:
                try:
                    ndt = datetime.strptime(n["datetime"], "%Y-%m-%d %H:%M")
                except (ValueError, KeyError):
                    continue
                if ndt <= now:
                    due.append(n)
                else:
                    remaining.append(n)
            if due:
                self._notifications = remaining
                self._save()
        return due


class SetNotificationTool(ToolBase):
    name = "set_notification"
    description = (
        "指定した日時にユーザーへ通知を予約します。"
        "日時は 'YYYY-MM-DD HH:MM' 形式で指定してください。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "datetime": {
                "type": "string",
                "description": "通知日時 (YYYY-MM-DD HH:MM 形式)",
            },
            "message": {
                "type": "string",
                "description": "通知内容",
            },
        },
        "required": ["datetime", "message"],
    }

    def __init__(self, store: NotificationStore) -> None:
        self._store = store

    async def execute(self, **kwargs) -> ToolResult:
        dt_str = kwargs.get("datetime", "")
        message = kwargs.get("message", "")

        if not dt_str or not message:
            return ToolResult(content="datetime と message は必須です。", is_error=True)

        try:
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        except ValueError:
            return ToolResult(
                content=f"日時の形式が不正です: '{dt_str}'。YYYY-MM-DD HH:MM 形式で指定してください。",
                is_error=True,
            )

        if dt <= datetime.now():
            return ToolResult(
                content=f"過去の日時は指定できません: {dt_str}",
                is_error=True,
            )

        notification_id = self._store.add(dt, message)
        return ToolResult(
            content=f"通知を予約しました。ID: {notification_id}, 日時: {dt_str}, 内容: {message}"
        )


class ListNotificationsTool(ToolBase):
    name = "list_notifications"
    description = "予約済みの通知一覧を表示します。"
    input_schema = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, store: NotificationStore) -> None:
        self._store = store

    async def execute(self, **kwargs) -> ToolResult:
        notifications = self._store.list_all()
        if not notifications:
            return ToolResult(content="予約されている通知はありません。")
        lines = []
        for n in notifications:
            lines.append(f"- ID: {n['id']}, 日時: {n['datetime']}, 内容: {n['message']}")
        return ToolResult(content=f"通知一覧 ({len(notifications)}件):\n" + "\n".join(lines))


class DeleteNotificationTool(ToolBase):
    name = "delete_notification"
    description = "指定IDの予約済み通知を削除します。事前にlist_notificationsでIDを確認してください。"
    input_schema = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "削除する通知のID",
            },
        },
        "required": ["id"],
    }

    def __init__(self, store: NotificationStore) -> None:
        self._store = store

    async def execute(self, **kwargs) -> ToolResult:
        notification_id = kwargs.get("id", "")
        if not notification_id:
            return ToolResult(content="id は必須です。", is_error=True)
        if self._store.remove(notification_id):
            return ToolResult(content=f"通知を削除しました。ID: {notification_id}")
        return ToolResult(
            content=f"ID '{notification_id}' の通知は見つかりませんでした。",
            is_error=True,
        )
