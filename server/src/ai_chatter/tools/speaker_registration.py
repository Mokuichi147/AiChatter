from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ai_chatter.tools.base import ToolBase, ToolResult

if TYPE_CHECKING:
    from ai_chatter.speaker_id import SpeakerIdentifier

logger = logging.getLogger(__name__)


class RegisterSpeakerTool(ToolBase):
    name = "register_speaker"
    description = "今話している人の声を名前と紐づけて覚えます。相手が名乗ったり自己紹介したときに使います。"
    input_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "登録する話者の名前",
            },
        },
        "required": ["name"],
    }

    def __init__(self, speaker_id: SpeakerIdentifier, get_pipelines) -> None:
        self._speaker_id = speaker_id
        self._get_pipelines = get_pipelines

    async def execute(self, **kwargs) -> ToolResult:
        name = kwargs.get("name", "").strip()
        if not name:
            return ToolResult(content="名前を指定してください。", is_error=True)

        # 現在の発話の音声データを使って即座に登録
        audio_data = None
        pipelines = self._get_pipelines()
        for pipeline in list(pipelines):
            if getattr(pipeline, "_current_audio_data", None):
                audio_data = pipeline._current_audio_data
                break

        if not audio_data:
            return ToolResult(content="登録に使える音声データがありません。", is_error=True)

        result = self._speaker_id.enroll(name, audio_data)
        logger.info(f"話者登録完了: {name} (embedding数: {result['embedding_count']})")

        # 履歴を遡及更新
        for pipeline in list(pipelines):
            self._speaker_id.retroactive_update(name, pipeline._history)

        return ToolResult(
            content=f"'{name}' の声を登録しました。声紋数: {result['embedding_count']}"
        )


class ListSpeakersTool(ToolBase):
    name = "list_speakers"
    description = "登録済みの話者（声の持ち主）の一覧を表示します。"
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, speaker_id: SpeakerIdentifier) -> None:
        self._speaker_id = speaker_id

    async def execute(self, **kwargs) -> ToolResult:
        speakers = self._speaker_id.list_speakers()
        if not speakers:
            return ToolResult(content="登録されている話者はいません。")
        return ToolResult(content=f"登録済み話者: {', '.join(speakers)}")


class UnregisterSpeakerTool(ToolBase):
    name = "unregister_speaker"
    description = "指定した名前の話者登録を解除します。"
    input_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "登録解除する話者の名前",
            },
        },
        "required": ["name"],
    }

    def __init__(self, speaker_id: SpeakerIdentifier) -> None:
        self._speaker_id = speaker_id

    async def execute(self, **kwargs) -> ToolResult:
        name = kwargs.get("name", "").strip()
        if not name:
            return ToolResult(content="名前を指定してください。", is_error=True)

        if self._speaker_id.unenroll(name):
            return ToolResult(content=f"'{name}' の声の登録を解除しました。")
        return ToolResult(content=f"'{name}' は登録されていません。", is_error=True)


class MergeSpeakersTool(ToolBase):
    name = "merge_speakers"
    description = "2つの話者を統合します。sourceの声紋をtargetに統合し、会話履歴も更新します。"
    input_schema = {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "統合元の話者名（統合後に削除される）",
            },
            "target": {
                "type": "string",
                "description": "統合先の話者名（声紋が追加される）",
            },
        },
        "required": ["source", "target"],
    }

    def __init__(self, speaker_id: SpeakerIdentifier, get_pipelines) -> None:
        self._speaker_id = speaker_id
        self._get_pipelines = get_pipelines

    async def execute(self, **kwargs) -> ToolResult:
        source = kwargs.get("source", "").strip()
        target = kwargs.get("target", "").strip()
        if not source or not target:
            return ToolResult(content="sourceとtargetの両方を指定してください。", is_error=True)

        result = self._speaker_id.merge_speakers(source, target)
        if "error" in result:
            return ToolResult(content=result["error"], is_error=True)

        # 会話履歴も更新
        pipelines = self._get_pipelines()
        for pipeline in list(pipelines):
            self._speaker_id.retroactive_merge(source, target, pipeline._history)

        return ToolResult(
            content=f"'{source}' を '{target}' に統合しました。声紋数: {result['embedding_count']}"
        )
