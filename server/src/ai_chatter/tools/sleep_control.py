import logging
from typing import Callable

from ai_chatter.tools.base import ToolBase, ToolResult

logger = logging.getLogger(__name__)


class SetSleepTool(ToolBase):
    name = "set_sleep"
    description = "接続中のデバイスをスリープモードにします。マイクとスピーカーが停止し、画面が消灯します。"
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, get_pipelines: Callable) -> None:
        self._get_pipelines = get_pipelines

    async def execute(self, **kwargs) -> ToolResult:
        pipelines = self._get_pipelines()
        if not pipelines:
            return ToolResult(content="接続中のデバイスがありません。", is_error=True)

        for pipeline in list(pipelines):
            try:
                await pipeline.send_sleep()
            except Exception as e:
                logger.error(f"スリープ送信エラー: {e}", exc_info=True)
                return ToolResult(content=f"スリープ送信エラー: {e}", is_error=True)

        logger.info("デバイスをスリープモードに設定")
        return ToolResult(content="スリープを有効化しました。再度呼び出す必要はありません。")
