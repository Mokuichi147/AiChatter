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
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )

    @staticmethod
    def _content_to_text(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
                else:
                    chunks.append(str(item))
            return "\n".join(c for c in chunks if c)
        return "" if content is None else str(content)

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> SubAgentLLMResponse:
        model = (settings.llm_sub_model or "").strip() or settings.llm_model
        logger.info(f"サブエージェントLLMリクエスト (モデル: {model})")

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1200,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        tool_calls: list[SubAgentToolCall] = []
        for tc in msg.tool_calls or []:
            args = tc.function.arguments
            if isinstance(args, (dict, list)):
                arguments = json.dumps(args, ensure_ascii=False)
            else:
                arguments = args or "{}"
            tool_calls.append(
                SubAgentToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                )
            )

        return SubAgentLLMResponse(
            content=self._content_to_text(msg.content),
            tool_calls=tool_calls,
        )
