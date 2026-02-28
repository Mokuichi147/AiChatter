import logging

from tools.base import ToolBase, ToolResult

logger = logging.getLogger(__name__)


class SetVolumeTool(ToolBase):
    name = "set_volume"
    description = "音声の音量を変更します。レベル1（最小）〜10（最大）で指定してください。"
    input_schema = {
        "type": "object",
        "properties": {
            "level": {
                "type": "integer",
                "description": "音量レベル (1-10)",
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["level"],
    }

    def __init__(self, tts) -> None:
        self._tts = tts

    async def execute(self, **kwargs) -> ToolResult:
        level = kwargs.get("level")
        if level is None or not isinstance(level, int):
            return ToolResult(content="levelは1-10の整数で指定してください", is_error=True)

        level = max(1, min(10, level))
        # level 1=328, level 5=1640, level 10=3276 (max int16=32767の約10%)
        self._tts.volume_scale = int(328 * level)
        logger.info(f"音量変更: level={level}, scale={self._tts.volume_scale}")
        return ToolResult(content=f"音量をレベル{level}に設定しました。")
