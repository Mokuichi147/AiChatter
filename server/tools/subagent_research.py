import json

from config import settings
from subagent.job_manager import SubAgentJobManager
from subagent.models import SubAgentJobRequest
from tools.base import ToolBase, ToolResult


class RunSubAgentResearchTool(ToolBase):
    name = "run_subagent_research"
    description = (
        "時間のかかる調査をバックグラウンドジョブとして開始します。"
        "進捗は list_subagent_jobs、結果は get_subagent_job で確認できます。"
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

        await self._job_manager.submit_job(
            SubAgentJobRequest(goal=goal, hints=hints, priority=priority)
        )
        return ToolResult(
            content=(
                "調査を開始しました。"
                "進捗確認は list_subagent_jobs、結果取得は get_subagent_job を使ってください。"
            )
        )


class ListSubAgentJobsTool(ToolBase):
    name = "list_subagent_jobs"
    description = "調査ジョブの進捗一覧を取得します。"
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
            "include_job_id": {
                "type": "boolean",
                "description": "内部ID(job_id)を含めるか。既定はfalse。",
                "default": False,
            },
        },
    }

    def __init__(self, job_manager: SubAgentJobManager) -> None:
        self._job_manager = job_manager

    async def execute(self, **kwargs) -> ToolResult:
        status = str(kwargs.get("status", "")).strip()
        limit = kwargs.get("limit", 10)
        include_job_id = kwargs.get("include_job_id", False)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 10

        if isinstance(include_job_id, str):
            include_job_id = include_job_id.strip().lower() in {"1", "true", "yes", "on"}
        else:
            include_job_id = bool(include_job_id)

        jobs = await self._job_manager.list_jobs(status=status, limit=limit)
        if not jobs:
            return ToolResult(content="該当するサブエージェントジョブはありません。")

        if not include_job_id:
            jobs = [{k: v for k, v in job.items() if k != "job_id"} for job in jobs]

        return ToolResult(
            content="調査ジョブ一覧:\n"
            + json.dumps(jobs, ensure_ascii=False, indent=2)
        )


class GetSubAgentJobTool(ToolBase):
    name = "get_subagent_job"
    description = "調査ジョブ詳細を取得します。job_id未指定時は最新ジョブを返します。"
    input_schema = {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "取得するジョブID",
            },
            "selector": {
                "type": "string",
                "description": "job_id未指定時の取得対象",
                "enum": ["latest", "latest_running", "latest_succeeded"],
                "default": "latest",
            },
            "include_job_id": {
                "type": "boolean",
                "description": "内部ID(job_id)を結果に含めるか。既定はfalse。",
                "default": False,
            },
        },
    }

    def __init__(self, job_manager: SubAgentJobManager) -> None:
        self._job_manager = job_manager

    async def execute(self, **kwargs) -> ToolResult:
        job_id = str(kwargs.get("job_id", "")).strip()
        selector = str(kwargs.get("selector", "latest")).strip().lower()
        include_job_id = kwargs.get("include_job_id", False)

        if isinstance(include_job_id, str):
            include_job_id = include_job_id.strip().lower() in {"1", "true", "yes", "on"}
        else:
            include_job_id = bool(include_job_id)

        if job_id:
            job = await self._job_manager.get_job(job_id)
        else:
            status_map = {
                "latest": "",
                "latest_running": "running",
                "latest_succeeded": "succeeded",
            }
            if selector not in status_map:
                selector = "latest"
            job = await self._job_manager.get_latest_job(status=status_map[selector])

        if job is None:
            if job_id:
                return ToolResult(content="指定したjob_idは見つかりません。", is_error=True)
            return ToolResult(content="該当する調査ジョブがありません。", is_error=True)

        if not include_job_id:
            job = {k: v for k, v in job.items() if k != "job_id"}

        return ToolResult(
            content="調査ジョブ詳細:\n"
            + json.dumps(job, ensure_ascii=False, indent=2)
        )
