import asyncio
import logging
import uuid
from collections import deque

from ai_chatter.subagent.models import SubAgentJob, SubAgentJobRequest, now_str
from ai_chatter.subagent.runner import SubAgentRunner

logger = logging.getLogger(__name__)


class SubAgentJobManager:
    def __init__(self, runner: SubAgentRunner, timeout_sec: int = 45) -> None:
        self._runner = runner
        self._timeout_sec = max(5, timeout_sec)
        self._jobs: dict[str, SubAgentJob] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._completed_messages: deque[str] = deque()
        self._lock = asyncio.Lock()

    async def submit_job(self, req: SubAgentJobRequest) -> str:
        job_id = uuid.uuid4().hex[:10]
        job = SubAgentJob(job_id=job_id, request=req)

        async with self._lock:
            self._jobs[job_id] = job

        self._tasks[job_id] = asyncio.create_task(self._run_job(job_id))
        logger.info(f"サブエージェントジョブ登録: {job_id}")
        return job_id

    async def _run_job(self, job_id: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "running"
            job.started_at = now_str()

        try:
            async with asyncio.timeout(self._timeout_sec):
                result = await self._runner.run(job.request)
        except TimeoutError:
            logger.warning(f"サブエージェントジョブタイムアウト: {job_id}")
            partial = self._runner.get_partial_result()
            async with self._lock:
                job = self._jobs[job_id]
                if partial is not None:
                    partial.limitations.append(
                        f"タイムアウト({self._timeout_sec}秒)のため途中結果です"
                    )
                    job.status = "succeeded"
                    job.result = partial
                else:
                    job.status = "timed_out"
                    job.error = f"タイムアウトしました ({self._timeout_sec}秒)"
                job.finished_at = now_str()
                self._completed_messages.append(self._build_completion_message(job))
        except Exception as e:
            logger.error(f"サブエージェントジョブ失敗: {job_id} error={e}", exc_info=True)
            async with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.error = str(e)
                job.finished_at = now_str()
                self._completed_messages.append(self._build_completion_message(job))
        else:
            async with self._lock:
                job = self._jobs[job_id]
                job.status = "succeeded"
                job.result = result
                job.finished_at = now_str()
                self._completed_messages.append(self._build_completion_message(job))
        finally:
            self._tasks.pop(job_id, None)

    async def list_jobs(self, status: str = "", limit: int = 10) -> list[dict]:
        limit = min(max(1, limit), 100)
        status = status.strip()

        async with self._lock:
            jobs = list(self._jobs.values())

        jobs.sort(key=lambda j: j.created_at, reverse=True)

        if status:
            jobs = [j for j in jobs if j.status == status]

        return [j.to_summary_dict() for j in jobs[:limit]]

    async def get_job(self, job_id: str) -> dict | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            return job.to_detail_dict() if job else None

    async def get_latest_job(self, status: str = "") -> dict | None:
        status = status.strip()
        async with self._lock:
            jobs = list(self._jobs.values())

        jobs.sort(key=lambda j: j.created_at, reverse=True)
        if status:
            jobs = [j for j in jobs if j.status == status]

        if not jobs:
            return None
        return jobs[0].to_detail_dict()

    async def pop_completed_messages(self, limit: int = 10) -> list[str]:
        messages: list[str] = []
        limit = min(max(1, limit), 100)
        async with self._lock:
            while self._completed_messages and len(messages) < limit:
                messages.append(self._completed_messages.popleft())
        return messages

    async def requeue_completed_messages(self, messages: list[str]) -> None:
        if not messages:
            return
        async with self._lock:
            for message in reversed(messages):
                self._completed_messages.appendleft(message)

    async def shutdown(self) -> None:
        """実行中ジョブをキャンセルして終了する。"""
        tasks = list(self._tasks.values())
        if not tasks:
            return

        logger.info(f"サブエージェント実行中ジョブ停止: {len(tasks)}件")
        for task in tasks:
            task.cancel()

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception) and not isinstance(
                result, asyncio.CancelledError
            ):
                logger.warning(f"サブエージェント停止中エラー: {result}")

        self._tasks.clear()

    @staticmethod
    def _build_completion_message(job: SubAgentJob) -> str:
        header = (
            "[調査完了]\n"
            f"status: {job.status}\n"
            f"goal: {job.request.goal}\n"
        )
        if job.status == "succeeded" and job.result is not None:
            findings = "\n".join(f"- {v}" for v in job.result.findings[:5])
            evidence = "\n".join(f"- {v}" for v in job.result.evidence[:5])
            body = (
                f"answer: {job.result.answer}\n"
                "findings:\n"
                f"{findings or '- (なし)'}\n"
                "evidence:\n"
                f"{evidence or '- (なし)'}"
            )
            return header + body
        return header + f"error: {job.error or '不明なエラー'}"
