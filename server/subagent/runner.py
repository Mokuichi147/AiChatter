import json
import logging
import re
from collections import OrderedDict

from config import prompt_config
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
        self._partial_result: SubAgentJobResult | None = None

    @staticmethod
    def _build_system_prompt() -> str:
        return prompt_config.subagent_system_prompt.strip()

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
        last_content: str = ""
        self._partial_result = None

        for round_num in range(self._max_rounds):
            logger.info(f"サブエージェント推論ラウンド {round_num + 1}/{self._max_rounds}")
            response = await self._llm.complete(messages, tools=tools if tools else None)

            if response.content:
                last_content = response.content
                self._partial_result = self._make_partial_result(
                    last_content, used_tools
                )

            if not response.tool_calls:
                return self._make_result(response.content, used_tools)

            # function_callアイテムをinputに追加
            for tc in response.tool_calls:
                messages.append(
                    {
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                )

            for tc in response.tool_calls:
                used_tools.append(tc.name)
                logger.info(f"サブエージェント ツール実行: {tc.name}")
                tool_content = await self._tool_adapter.execute(tc.name, tc.arguments)
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tc.id,
                        "output": tool_content,
                    }
                )

        logger.warning("サブエージェント ラウンド上限到達、最終結果を要求")
        messages.append(
            {
                "role": "user",
                "content": "ここまでの情報で最終結果をJSONオブジェクトのみで返してください。",
            }
        )
        try:
            final = await self._llm.complete(messages, tools=None)
            return self._make_result(final.content, used_tools)
        except Exception as e:
            logger.warning(f"サブエージェント最終応答取得失敗: {e}")
            return self._make_partial_result(last_content, used_tools)

    def get_partial_result(self) -> SubAgentJobResult | None:
        """タイムアウト等で中断された場合に途中結果を返す。"""
        return self._partial_result

    def _make_partial_result(
        self, last_content: str, used_tools: list[str]
    ) -> SubAgentJobResult:
        """途中結果から可能な限りの結果を生成する。"""
        data = self._extract_json(last_content) if last_content else {}
        answer = str(data.get("answer") or "").strip()
        if not answer:
            answer = last_content.strip() if last_content else "ラウンド上限に達しました。"

        uniq_tools = list(OrderedDict.fromkeys(t for t in used_tools if t))
        limitations = self._to_str_list(data.get("limitations"))
        limitations.append("ラウンド上限到達のため途中結果です")

        return SubAgentJobResult(
            answer=self._trim(answer),
            findings=[self._trim(v) for v in self._to_str_list(data.get("findings"))],
            evidence=[self._trim(v) for v in self._to_str_list(data.get("evidence"))],
            limitations=[self._trim(v) for v in limitations],
            used_tools=uniq_tools,
        )
