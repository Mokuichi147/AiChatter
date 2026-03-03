import json

from config import settings
from subagent.job_manager import SubAgentJobManager
from subagent.models import SubAgentJobRequest
from tools.base import ToolBase, ToolResult


class RunSubAgentResearchTool(ToolBase):
    name = "run_subagent_research"
    description = (
        "時間のかかる調査をバックグラウンドジョブとして開始します。"
        "すぐにjob_idを返し、完了後は get_subagent_job で結果を取得できます。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "調査目標",
            },
            "hints": {
                "type": "string",
                "description": "補足条件や検索ヒント（任意）",
            },
            "priority": {
                "type": "string",
                "description": "優先度 (low/normal/high)",
                "enum": ["low", "normal", "high"],
                "default": "normal",
            },
        },
        "required": ["goal"],
    }

    def __init__(self, job_manager: SubAgentJobManager) -> None:
        self._job_manager = job_manager

    async def execute(self, **kwargs) -> ToolResult:
        if not settings.subagent_enabled:
            return ToolResult(content="サブエージェント機能は無効です。", is_error=True)

        goal = str(kwargs.get("goal", "")).strip()
        hints = str(kwargs.get("hints", "")).strip()
        priority = str(kwargs.get("priority", "normal")).strip().lower()

        if not goal:
            return ToolResult(content="goalは必須です。", is_error=True)
        if priority not in {"low", "normal", "high"}:
            priority = "normal"

        job_id = await self._job_manager.submit_job(
            SubAgentJobRequest(goal=goal, hints=hints, priority=priority)
        )
        return ToolResult(
            content=(
                "サブエージェント調査を開始しました。"
                f"job_id: {job_id}。"
                "進捗確認は list_subagent_jobs、結果取得は get_subagent_job を使ってください。"
            )
        )


class ListSubAgentJobsTool(ToolBase):
    name = "list_subagent_jobs"
    description = "サブエージェント調査ジョブの一覧を取得します。"
    input_schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "状態でフィルタ (queued/running/succeeded/failed/timed_out)",
                "enum": [
                    "queued",
                    "running",
                    "succeeded",
                    "failed",
                    "timed_out",
                ],
            },
            "limit": {
                "type": "integer",
                "description": "取得件数 (1-100)",
                "default": 10,
            },
        },
    }

    def __init__(self, job_manager: SubAgentJobManager) -> None:
        self._job_manager = job_manager

    async def execute(self, **kwargs) -> ToolResult:
        status = str(kwargs.get("status", "")).strip()
        limit = kwargs.get("limit", 10)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 10

        jobs = await self._job_manager.list_jobs(status=status, limit=limit)
        if not jobs:
            return ToolResult(content="該当するサブエージェントジョブはありません。")

        return ToolResult(
            content="サブエージェントジョブ一覧:\n"
            + json.dumps(jobs, ensure_ascii=False, indent=2)
        )


class GetSubAgentJobTool(ToolBase):
    name = "get_subagent_job"
    description = "指定したjob_idのサブエージェント調査ジョブ詳細を取得します。"
    input_schema = {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "取得するジョブID",
            },
        },
        "required": ["job_id"],
    }

    def __init__(self, job_manager: SubAgentJobManager) -> None:
        self._job_manager = job_manager

    async def execute(self, **kwargs) -> ToolResult:
        job_id = str(kwargs.get("job_id", "")).strip()
        if not job_id:
            return ToolResult(content="job_idは必須です。", is_error=True)

        job = await self._job_manager.get_job(job_id)
        if job is None:
            return ToolResult(content=f"job_id '{job_id}' は見つかりません。", is_error=True)

        return ToolResult(
            content="サブエージェントジョブ詳細:\n"
            + json.dumps(job, ensure_ascii=False, indent=2)
        )
