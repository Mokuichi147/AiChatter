from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

SubAgentJobStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "timed_out",
]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class SubAgentJobRequest:
    goal: str
    hints: str = ""
    priority: str = "normal"


@dataclass
class SubAgentJobResult:
    answer: str
    findings: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    used_tools: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "findings": self.findings,
            "evidence": self.evidence,
            "limitations": self.limitations,
            "used_tools": self.used_tools,
        }


@dataclass
class SubAgentJob:
    job_id: str
    request: SubAgentJobRequest
    status: SubAgentJobStatus = "queued"
    created_at: str = field(default_factory=now_str)
    started_at: str = ""
    finished_at: str = ""
    result: SubAgentJobResult | None = None
    error: str = ""

    def to_summary_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "goal": self.request.goal,
            "priority": self.request.priority,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def to_detail_dict(self) -> dict:
        data = self.to_summary_dict()
        data["hints"] = self.request.hints
        data["error"] = self.error
        data["result"] = self.result.to_dict() if self.result else None
        return data
