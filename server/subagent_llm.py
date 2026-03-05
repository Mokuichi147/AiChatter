import json
import logging
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class SubAgentToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class SubAgentLLMResponse:
    content: str
    tool_calls: list[SubAgentToolCall] = field(default_factory=list)


class SubAgentLLM:
    def __init__(self) -> None:
        api_key = settings.llm_sub_api_key or settings.llm_api_key or "no-key"
        base_url = settings.llm_sub_base_url or settings.llm_base_url or None
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> SubAgentLLMResponse:
        model = (settings.llm_sub_model or "").strip() or settings.llm_model
        logger.info(f"サブエージェントLLMリクエスト (モデル: {model})")

        kwargs: dict = {
            "model": model,
            "input": messages,
            "temperature": 0.3,
            "max_output_tokens": 1200,
        }
        reasoning = settings.llm_sub_reasoning or settings.llm_reasoning
        if reasoning:
            kwargs["reasoning"] = {"effort": reasoning}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await self._client.responses.create(**kwargs)

        content_parts: list[str] = []
        tool_calls: list[SubAgentToolCall] = []

        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if part.type == "output_text":
                        content_parts.append(part.text)
            elif item.type == "function_call":
                args = item.arguments
                if isinstance(args, (dict, list)):
                    arguments = json.dumps(args, ensure_ascii=False)
                else:
                    arguments = args or "{}"
                tool_calls.append(
                    SubAgentToolCall(
                        id=item.call_id,
                        name=item.name,
                        arguments=arguments,
                    )
                )

        return SubAgentLLMResponse(
            content="\n".join(content_parts),
            tool_calls=tool_calls,
        )
