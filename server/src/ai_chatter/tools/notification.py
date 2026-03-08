import json
import logging
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

from ai_chatter.tools.base import ToolBase, ToolResult
from ai_chatter._paths import SERVER_ROOT

logger = logging.getLogger(__name__)


class NotificationStore:
    """JSONファイルベースの通知永続ストア。"""

    def __init__(self, file_path: str) -> None:
        self._path = Path(file_path)
        if not self._path.is_absolute():
            self._path = SERVER_ROOT / file_path
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

    def add(self, dt: datetime, message: str, repeat: str | None = None) -> str:
        """通知を追加し、一意IDを返す。"""
        notification_id = uuid.uuid4().hex[:8]
        entry = {
            "id": notification_id,
            "datetime": dt.strftime("%Y-%m-%d %H:%M"),
            "message": message,
        }
        if repeat:
            entry["repeat"] = repeat
        with self._lock:
            self._notifications.append(entry)
            self._save()
        logger.info(f"通知を追加: id={notification_id} dt={entry['datetime']} msg={message} repeat={repeat}")
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
        """現在時刻を過ぎた通知を取り出す。repeat付きは次回日時に更新して残す。"""
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
                    repeat = n.get("repeat")
                    if repeat:
                        next_dt = self._calc_next(now, repeat)
                        if next_dt:
                            updated = dict(n)
                            updated["datetime"] = next_dt.strftime("%Y-%m-%d %H:%M")
                            remaining.append(updated)
                else:
                    remaining.append(n)
            if due:
                self._notifications = remaining
                self._save()
        return due

    @staticmethod
    def _calc_next(now: datetime, repeat: str) -> datetime | None:
        """repeat文字列から次回発火日時を計算する。"""
        if repeat.startswith("every:"):
            return NotificationStore._calc_next_interval(now, repeat[6:])
        if repeat.startswith("cron:"):
            return NotificationStore._calc_next_cron(now, repeat[5:])
        return None

    @staticmethod
    def _calc_next_interval(now: datetime, interval_str: str) -> datetime | None:
        """'30m', '2h', '1d' 形式からnowに加算した次回日時を返す。"""
        m = re.fullmatch(r"(\d+)([mhd])", interval_str)
        if not m:
            return None
        value, unit = int(m.group(1)), m.group(2)
        if unit == "m":
            return now + timedelta(minutes=value)
        if unit == "h":
            return now + timedelta(hours=value)
        if unit == "d":
            return now + timedelta(days=value)
        return None

    @staticmethod
    def _calc_next_cron(now: datetime, cron_str: str) -> datetime | None:
        """'HH:MM' or 'HH:MM:days' 形式から次回日時を返す。"""
        parts = cron_str.split(":")
        if len(parts) < 2:
            return None
        try:
            hour, minute = int(parts[0]), int(parts[1])
        except ValueError:
            return None
        day_filter = parts[2] if len(parts) >= 3 else None
        allowed_days: set[int] | None = None
        if day_filter:
            day_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
            if day_filter == "weekdays":
                allowed_days = {0, 1, 2, 3, 4}
            elif day_filter == "weekends":
                allowed_days = {5, 6}
            else:
                allowed_days = set()
                for d in day_filter.split(","):
                    d = d.strip().lower()
                    if d in day_map:
                        allowed_days.add(day_map[d])
                if not allowed_days:
                    return None

        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        for _ in range(8):
            if allowed_days is None or candidate.weekday() in allowed_days:
                return candidate
            candidate += timedelta(days=1)
        return None


class SetNotificationTool(ToolBase):
    name = "set_notification"
    description = (
        "指定した日時に通知を予約します。"
        "日時は 'YYYY-MM-DD HH:MM' 形式で指定します。"
        "repeatを指定すると定期通知として登録します。"
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
            "repeat": {
                "type": "string",
                "description": (
                    "定期実行パターン (省略時は1回限り)。"
                    "interval: 'every:30m', 'every:2h', 'every:1d'。"
                    "cron: 'cron:08:00'(毎日), 'cron:07:30:weekdays'(平日), 'cron:09:00:mon,fri'(指定曜日)。"
                ),
            },
        },
        "required": ["datetime", "message"],
    }

    def __init__(self, store: NotificationStore) -> None:
        self._store = store

    async def execute(self, **kwargs) -> ToolResult:
        dt_str = kwargs.get("datetime", "")
        message = kwargs.get("message", "")
        repeat = kwargs.get("repeat")

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

        if repeat:
            if not (repeat.startswith("every:") or repeat.startswith("cron:")):
                return ToolResult(
                    content=f"repeat形式が不正です: '{repeat}'。'every:30m' や 'cron:08:00' 形式で指定してください。",
                    is_error=True,
                )
            test_now = datetime.now()
            if repeat.startswith("every:"):
                if NotificationStore._calc_next_interval(test_now, repeat[6:]) is None:
                    return ToolResult(
                        content=f"repeatのinterval形式が不正です: '{repeat}'。'every:30m', 'every:2h', 'every:1d' 等を指定してください。",
                        is_error=True,
                    )
            elif repeat.startswith("cron:"):
                if NotificationStore._calc_next_cron(test_now, repeat[5:]) is None:
                    return ToolResult(
                        content=f"repeatのcron形式が不正です: '{repeat}'。'cron:08:00', 'cron:07:30:weekdays' 等を指定してください。",
                        is_error=True,
                    )

        notification_id = self._store.add(dt, message, repeat)
        result_msg = f"通知を予約しました。ID: {notification_id}, 日時: {dt_str}, 内容: {message}"
        if repeat:
            result_msg += f", 繰り返し: {repeat}"
        return ToolResult(content=result_msg)


class ListNotificationsTool(ToolBase):
    name = "list_notifications"
    description = "予約済みの通知一覧を取得します。"
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
            line = f"- ID: {n['id']}, 日時: {n['datetime']}, 内容: {n['message']}"
            if n.get("repeat"):
                line += f", 繰り返し: {n['repeat']}"
            lines.append(line)
        return ToolResult(content=f"通知一覧 ({len(notifications)}件):\n" + "\n".join(lines))


class DeleteNotificationTool(ToolBase):
    name = "delete_notification"
    description = "指定IDの通知を削除します。削除前にlist_notificationsでIDを確認してください。"
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
