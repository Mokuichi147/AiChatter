import logging

from ai_chatter.battery import BatteryStore
from ai_chatter.tools.base import ToolBase, ToolResult

logger = logging.getLogger(__name__)


class GetBatteryTool(ToolBase):
    name = "get_battery"
    description = "接続中の全デバイス（M5StickS3・PC）のバッテリー残量・充電状態を取得します。"
    input_schema = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, store: BatteryStore) -> None:
        self._store = store

    async def execute(self, **kwargs) -> ToolResult:
        all_info = await self._store.get_all()
        if not all_info:
            return ToolResult(content="バッテリー情報がありません。デバイスが未接続の可能性があります。")

        source_labels = {"m5": "M5StickS3", "pc": "PC"}
        lines = []
        for source, info in all_info.items():
            label = source_labels.get(source, source)
            charging = "充電中" if info.is_charging else "放電中"
            updated = info.updated_at.strftime("%H:%M:%S")
            lines.append(f"- {label}: {info.level}% ({charging}) [更新: {updated}]")

        return ToolResult(content=f"バッテリー状態:\n" + "\n".join(lines))
