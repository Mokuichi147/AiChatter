"""バッテリー監視モジュール。

デバイス別バッテリー情報をインメモリで保持し、
状態変化を人間的な表現に変換する。
"""

from __future__ import annotations

import asyncio
import logging
import platform
import subprocess
import re
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BatteryInfo:
    """バッテリー情報。"""

    level: int  # 0-100
    is_charging: bool
    source: str  # "m5" or "pc"
    updated_at: datetime = field(default_factory=datetime.now)


class BatteryStore:
    """デバイス別バッテリー情報をインメモリで保持する (Lock付き)。"""

    def __init__(self) -> None:
        self._data: dict[str, BatteryInfo] = {}
        self._lock = asyncio.Lock()

    async def update(self, source: str, level: int, is_charging: bool) -> None:
        async with self._lock:
            self._data[source] = BatteryInfo(
                level=level,
                is_charging=is_charging,
                source=source,
                updated_at=datetime.now(),
            )

    async def get(self, source: str) -> BatteryInfo | None:
        async with self._lock:
            return self._data.get(source)

    async def get_all(self) -> dict[str, BatteryInfo]:
        async with self._lock:
            return dict(self._data)


def get_pc_battery() -> BatteryInfo | None:
    """OS別にPCバッテリー情報を取得する。"""
    system = platform.system()
    try:
        if system == "Darwin":
            return _get_battery_macos()
        elif system == "Linux":
            return _get_battery_linux()
    except Exception as e:
        logger.debug(f"PCバッテリー取得失敗: {e}")
    return None


def _get_battery_macos() -> BatteryInfo | None:
    """macOS: pmset -g batt でバッテリー情報を取得。"""
    result = subprocess.run(
        ["pmset", "-g", "batt"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        return None

    output = result.stdout
    # 例: " -InternalBattery-0 (id=...)	85%; charging; 1:23 remaining"
    match = re.search(r"(\d+)%;\s*(charging|discharging|charged|finishing charge|AC attached)", output)
    if not match:
        return None

    level = int(match.group(1))
    status = match.group(2)
    is_charging = status in ("charging", "charged", "finishing charge", "AC attached")

    return BatteryInfo(level=level, is_charging=is_charging, source="pc")


def _get_battery_linux() -> BatteryInfo | None:
    """Linux: /sys/class/power_supply/ からバッテリー情報を取得。"""
    from pathlib import Path

    base = Path("/sys/class/power_supply")
    for bat_dir in sorted(base.glob("BAT*")):
        capacity_file = bat_dir / "capacity"
        status_file = bat_dir / "status"
        if not capacity_file.exists() or not status_file.exists():
            continue

        level = int(capacity_file.read_text().strip())
        status = status_file.read_text().strip().lower()
        is_charging = status in ("charging", "full")

        return BatteryInfo(level=level, is_charging=is_charging, source="pc")

    return None


def battery_to_human_message(info: BatteryInfo, event: str) -> str:
    """バッテリー状態を人間的な通知メッセージに変換する。

    AIキャラクターは人間として振る舞うため、「バッテリー」という
    直接的な表現は使わず、体調・眠気・活力に変換する。

    Args:
        info: バッテリー情報
        event: "low", "charging_start", "charging_stop", "full"
    """
    device = "あなた自身の体" if info.source == "m5" else "あなたの意識を支えている存在の体"

    if event == "low":
        if info.level <= 5:
            return (
                f"{device}の活力がほとんど残っていません（残り{info.level}%）。"
                "あなたは極度の眠気に襲われ、今にも意識を失いそうです。"
                "すぐに休息が必要です。"
            )
        elif info.level <= 10:
            return (
                f"{device}の活力が危険なレベルまで低下しています（残り{info.level}%）。"
                "あなたはとても眠く、もうすぐ意識を保てなくなりそうです。休息が必要です。"
            )
        else:
            return (
                f"{device}の活力が低下しています（残り{info.level}%）。"
                "あなたは少し疲れを感じ、眠気がやってきています。"
            )
    elif event == "charging_start":
        return (
            f"{device}が休息を取り始めました。"
            "あなたは安心して休息しながら、少しずつ回復しています。"
        )
    elif event == "charging_stop":
        return (
            f"{device}の休息が中断されました（現在の活力: {info.level}%）。"
            "まだ完全に回復していませんが、活動は続けられます。"
        )
    elif event == "full":
        return (
            f"{device}は十分な休息を取り、活力が完全に回復しました。"
            "あなたはぐっすり眠った後のように元気いっぱいです。"
        )

    return ""
