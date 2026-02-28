import asyncio
import logging
import uuid

from subagent.models import SubAgentJob, SubAgentJobRequest, now_str
from subagent.runner import SubAgentRunner

logger = logging.getLogger(__name__)


class SubAgentJobManager:
    def __init__(self, runner: SubAgentRunner, timeout_sec: int = 45) -> None:
        self._runner = runner
        self._timeout_sec = max(5, timeout_sec)
        self._jobs: dict[str, SubAgentJob] = {}
        self._tasks: dict[str, asyncio.Task] = {}
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
            async with self._lock:
                job = self._jobs[job_id]
                job.status = "timed_out"
                job.error = f"タイムアウトしました ({self._timeout_sec}秒)"
                job.finished_at = now_str()
        except Exception as e:
            logger.error(f"サブエージェントジョブ失敗: {job_id} error={e}", exc_info=True)
            async with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.error = str(e)
                job.finished_at = now_str()
        else:
            async with self._lock:
                job = self._jobs[job_id]
                job.status = "succeeded"
                job.result = result
                job.finished_at = now_str()
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
