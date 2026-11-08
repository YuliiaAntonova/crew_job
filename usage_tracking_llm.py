"""Wrap any LlamaIndex LLM so complete/acomplete increment run_usage."""

from __future__ import annotations

from typing import Any

from run_usage import record_llm_completion


class UsageTrackingLLM:
    """Delegate to inner LLM; record prompt/response tokens on every completion."""

    def __init__(self, llm: Any, *, backend: str):
        object.__setattr__(self, "_llm", llm)
        object.__setattr__(self, "_backend", backend)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_llm"), name)

    def _record(self, prompt: str, response: Any, *, label: str | None = None) -> None:
        text = getattr(response, "text", None)
        if text is None:
            text = str(response)
        record_llm_completion(
            prompt,
            text,
            backend=object.__getattribute__(self, "_backend"),
            label=label,
            raw_response=response,
        )

    def complete(self, prompt: str, **kwargs: Any):
        label = kwargs.pop("label", None)
        resp = object.__getattribute__(self, "_llm").complete(prompt, **kwargs)
        self._record(prompt, resp, label=label)
        return resp

    async def acomplete(self, prompt: str, **kwargs: Any):
        label = kwargs.pop("label", None)
        resp = await object.__getattribute__(self, "_llm").acomplete(prompt, **kwargs)
        self._record(prompt, resp, label=label)
        return resp
