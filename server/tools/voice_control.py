import json
import logging
from pathlib import Path

from config import character_data_path
from local_tts import _level_to_scale
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
        self._tts.volume_scale = _level_to_scale(level)
        logger.info(f"音量変更: level={level}, scale={self._tts.volume_scale}")

        # settings.jsonに永続化
        try:
            settings_path = Path(character_data_path("settings.json"))
            if not settings_path.is_absolute():
                settings_path = Path(__file__).resolve().parent.parent / settings_path
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if settings_path.exists():
                data = json.loads(settings_path.read_text(encoding="utf-8"))
            data["volume_level"] = level
            settings_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"音量設定の保存に失敗: {e}")

        return ToolResult(content=f"音量をレベル{level}に設定しました。")
