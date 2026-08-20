"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletion
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

# USD per 1M tokens, (input, output). Extend as needed; unknown models estimate as None.
_PRICING_USD_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client backed by the OpenAI Chat Completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.openai_model
        self.timeout = timeout if timeout is not None else float(settings.timeout_seconds)

        resolved_key = api_key if api_key is not None else settings.openai_api_key
        if not resolved_key:
            raise AgentExecutionError(
                "OPENAI_API_KEY is not set. Add it to .env before calling a real LLM."
            )
        self._client = OpenAI(api_key=resolved_key, timeout=self.timeout)

    def complete(
        self, system_prompt: str, user_prompt: str, *, temperature: float = 0.2
    ) -> LLMResponse:
        """Return a model completion, retrying transient provider errors.

        Retry, timeout, and token accounting live here rather than inside agents.
        """

        try:
            completion = self._call(system_prompt, user_prompt, temperature)
        except OpenAIError as exc:
            raise AgentExecutionError(f"LLM call failed: {exc}") from exc

        choice = completion.choices[0]
        usage = completion.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._estimate_cost(input_tokens, output_tokens),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(OpenAIError),
        reraise=True,
    )
    def _call(self, system_prompt: str, user_prompt: str, temperature: float) -> ChatCompletion:
        return self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

    def _estimate_cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        pricing = _PRICING_USD_PER_1M_TOKENS.get(self.model)
        if pricing is None or input_tokens is None or output_tokens is None:
            return None
        input_price, output_price = pricing
        return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price
