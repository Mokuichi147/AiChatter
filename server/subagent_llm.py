import json
import logging
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from config import llm_config

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
        sub = llm_config.sub
        api_key = sub.api_key or llm_config.api_key or "no-key"
        base_url = sub.base_url or llm_config.base_url or None
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> SubAgentLLMResponse:
        sub = llm_config.sub
        model = sub.model or llm_config.model
        logger.info(f"サブエージェントLLMリクエスト (モデル: {model})")

        kwargs: dict = {
            "model": model,
            "input": messages,
            "temperature": 0.3,
            "max_output_tokens": 1200,
        }
        reasoning = sub.reasoning or llm_config.reasoning
        if reasoning:
            kwargs["reasoning"] = {"effort": reasoning}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await self._client.responses.create(**kwargs)

        content_parts: list[str] = []
        tool_calls: list[SubAgentToolCall] = []

        logger.debug(f"サブエージェント response.output: {response.output}")
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
