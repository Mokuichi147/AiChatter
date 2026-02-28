import json
import logging
import re
from collections import OrderedDict

from subagent.models import SubAgentJobRequest, SubAgentJobResult
from subagent.tool_adapter import SubAgentToolAdapter
from subagent_llm import SubAgentLLM

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"(\{[\s\S]*\})")


class SubAgentRunner:
    def __init__(
        self,
        llm: SubAgentLLM,
        tool_adapter: SubAgentToolAdapter,
        max_rounds: int = 8,
        result_max_chars: int = 4000,
    ) -> None:
        self._llm = llm
        self._tool_adapter = tool_adapter
        self._max_rounds = max(1, max_rounds)
        self._result_max_chars = max(500, result_max_chars)

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "あなたは調査専用のサブエージェントです。"
            "目的達成のために必要なら複数回ツールを使ってください。"
            "最終出力は必ずJSONオブジェクトのみで返します。"
            "JSONのキーは answer, findings, evidence, limitations の4つです。"
            "findings/evidence/limitations は文字列配列にしてください。"
            "Markdownや前置き文章は出力しないでください。"
        )

    @staticmethod
    def _build_user_prompt(req: SubAgentJobRequest) -> str:
        hints = req.hints.strip() or "(なし)"
        return (
            "以下の調査を実行してください。\n"
            f"- goal: {req.goal}\n"
            f"- hints: {hints}\n"
            f"- priority: {req.priority}\n"
            "不足情報があればツールで補ってください。"
        )

    @staticmethod
    def _to_str_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _extract_json(text: str) -> dict:
        raw = text.strip()
        if not raw:
            return {}

        match = _JSON_BLOCK_RE.search(raw)
        if match:
            raw = match.group(1).strip()
        else:
            match = _JSON_OBJ_RE.search(raw)
            if match:
                raw = match.group(1).strip()

        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _trim(self, text: str) -> str:
        if len(text) <= self._result_max_chars:
            return text
        return text[: self._result_max_chars - 3] + "..."

    def _make_result(self, text: str, used_tools: list[str]) -> SubAgentJobResult:
        data = self._extract_json(text)

        answer = str(data.get("answer") or "").strip()
        findings = self._to_str_list(data.get("findings"))
        evidence = self._to_str_list(data.get("evidence"))
        limitations = self._to_str_list(data.get("limitations"))

        if not answer:
            answer = text.strip() or "調査結果を取得しました。"

        uniq_tools = list(OrderedDict.fromkeys(t for t in used_tools if t))

        return SubAgentJobResult(
            answer=self._trim(answer),
            findings=[self._trim(v) for v in findings],
            evidence=[self._trim(v) for v in evidence],
            limitations=[self._trim(v) for v in limitations],
            used_tools=uniq_tools,
        )

    async def run(self, req: SubAgentJobRequest) -> SubAgentJobResult:
        messages: list[dict] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_user_prompt(req)},
        ]
        tools = self._tool_adapter.to_openai_tools()
        used_tools: list[str] = []

        for round_num in range(self._max_rounds):
            logger.info(f"サブエージェント推論ラウンド {round_num + 1}/{self._max_rounds}")
            response = await self._llm.complete(messages, tools=tools if tools else None)

            if not response.tool_calls:
                return self._make_result(response.content, used_tools)

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments,
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
            )

            for tc in response.tool_calls:
                used_tools.append(tc.name)
                logger.info(f"サブエージェント ツール実行: {tc.name}")
                tool_content = await self._tool_adapter.execute(tc.name, tc.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_content,
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": "ここまでの情報で最終結果をJSONオブジェクトのみで返してください。",
            }
        )
        final = await self._llm.complete(messages, tools=None)
        return self._make_result(final.content, used_tools)
