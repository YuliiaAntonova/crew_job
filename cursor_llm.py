"""LlamaIndex LLM backed by Cursor Agent (CURSOR_API_KEY).

Lab one-shot completions: ``helper.llm_complete`` (Cursor / OpenAI / Ollama).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from cursor_sdk import Agent, AgentOptions, LocalAgentOptions
from cursor_sdk.errors import CursorSDKError

from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms.custom import CustomLLM


class CursorLLM(CustomLLM):
    """Use Cursor Agent for RAG answers (not OpenAI-compatible HTTP)."""

    api_key: str = Field(repr=False)
    model: str = "composer-2.5"
    rag_cwd: str | None = None

    _agent: Any = PrivateAttr(default=None)
    _agent_cm: Any = PrivateAttr(default=None)

    @classmethod
    def class_name(cls) -> str:
        return "cursor_llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model, is_chat_model=False)

    def start_agent_session(self) -> None:
        """Reuse one Cursor Agent for multiple complete() calls (e.g. §6 pipeline)."""
        if self._agent is not None:
            return
        cwd = self.rag_cwd or os.getcwd()
        self._agent_cm = Agent.create(
            model=self.model,
            api_key=self.api_key,
            local=LocalAgentOptions(cwd=cwd),
        )
        self._agent = self._agent_cm.__enter__()

    def end_agent_session(self) -> None:
        if self._agent_cm is not None:
            self._agent_cm.__exit__(None, None, None)
        self._agent = None
        self._agent_cm = None

    @property
    def agent_session_active(self) -> bool:
        return self._agent is not None

    def _call_cursor(self, prompt: str) -> str:
        cwd = self.rag_cwd or os.getcwd()
        if self._agent is not None:
            text = self._agent.send(prompt).text().strip()
        else:
            with Agent.create(
                model=self.model,
                api_key=self.api_key,
                local=LocalAgentOptions(cwd=cwd),
            ) as agent:
                text = agent.send(prompt).text().strip()
        return text

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        del formatted, kwargs
        return CompletionResponse(text=self._call_cursor(prompt))

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        text = self.complete(prompt, formatted=formatted, **kwargs).text
        yield CompletionResponse(text=text, delta=text)


_PROJECT_ROOT = Path(__file__).resolve().parent
_LAB_VENV_PYTHON = _PROJECT_ROOT.parent / "env" / "bin" / "python"
_lab_usage_llm: Any = None


def strip_llm_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def register_lab_usage_llm(llm: Any) -> None:
    """Optional UsageTrackingLLM from get_llm(); used by reset_lab_cursor_llm()."""
    global _lab_usage_llm
    _lab_usage_llm = llm


def reset_lab_cursor_llm() -> None:
    """End reused CursorLLM session (if any) and archive excess SDK agents."""
    global _lab_usage_llm
    if _lab_usage_llm is not None:
        inner = getattr(_lab_usage_llm, "_llm", _lab_usage_llm)
        if hasattr(inner, "end_agent_session"):
            inner.end_agent_session()
    _lab_usage_llm = None
    from helper import get_cursor_api_key

    prune_cursor_agents(get_cursor_api_key(), max_active=5)


def cursor_agent_stats(api_key: str) -> tuple[int, int]:
    list_opts = {"apiKey": api_key}
    try:
        items = list(Agent.list(list_opts).items)
    except Exception:
        return -1, -1
    return sum(1 for i in items if not i.archived), len(items)


def prune_cursor_agents(api_key: str, *, max_active: int = 5) -> int:
    """Archive excess SDK agents (~50 account cap → status='error', empty body)."""
    list_opts = {"apiKey": api_key}
    try:
        items = list(Agent.list(list_opts).items)
    except Exception:
        return 0
    active = [info for info in items if not info.archived]
    archived = 0
    for info in active[max_active:]:
        try:
            Agent.archive(info.agent_id, list_opts)
            archived += 1
        except Exception:
            continue
    return archived


def _cursor_rag_cwd() -> str:
    return os.getenv("CURSOR_RAG_CWD") or str(_PROJECT_ROOT)


def _cursor_agent_options(api_key: str, model_slug: str) -> AgentOptions:
    return AgentOptions(
        api_key=api_key,
        model=model_slug,
        local=LocalAgentOptions(cwd=_cursor_rag_cwd()),
    )


def _cursor_prompt_inprocess(prompt: str, api_key: str, slug: str) -> tuple[str, str]:
    run = Agent.prompt(prompt, _cursor_agent_options(api_key, slug))
    text = strip_llm_fence((run.result or "").strip())
    return getattr(run, "status", "?"), text


def _cursor_prompt_subprocess(prompt: str, api_key: str, slug: str) -> tuple[str, str]:
    """Project venv when the Jupyter kernel cannot use the Cursor local bridge."""
    py = _LAB_VENV_PYTHON if _LAB_VENV_PYTHON.is_file() else Path(sys.executable)
    cwd = _cursor_rag_cwd()
    code = r"""
import os, sys
from dotenv import load_dotenv
load_dotenv()
from cursor_sdk import Agent, AgentOptions, LocalAgentOptions
from helper import get_cursor_api_key
prompt = sys.stdin.read()
slug = os.environ["CURSOR_PROMPT_MODEL"]
cwd = os.environ["CURSOR_PROMPT_CWD"]
opts = AgentOptions(
    api_key=get_cursor_api_key(),
    model=slug,
    local=LocalAgentOptions(cwd=cwd),
)
run = Agent.prompt(prompt, opts)
print(run.status)
print("---BODY---")
print(run.result or "")
"""
    env = {
        **os.environ,
        "CURSOR_PROMPT_MODEL": slug,
        "CURSOR_PROMPT_CWD": cwd,
    }
    proc = subprocess.run(
        [str(py), "-c", code],
        input=prompt,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        timeout=600,
    )
    if proc.returncode != 0:
        return "error", ""
    out = proc.stdout or ""
    if "---BODY---" not in out:
        return "error", ""
    status_line, body = out.split("---BODY---", 1)
    return status_line.strip(), strip_llm_fence(body.strip())


def cursor_prompt_complete(prompt: str, *, api_key: str, model_slug: str) -> str:
    """Cursor Agent.prompt with prune, model fallback, and subprocess retry."""
    prune_cursor_agents(api_key, max_active=5)
    last_status, last_text = "error", ""

    for slug in (model_slug, "default"):
        for attempt, fn in (
            ("inprocess", _cursor_prompt_inprocess),
            ("subprocess", _cursor_prompt_subprocess),
        ):
            try:
                status, text = fn(prompt, api_key, slug)
            except CursorSDKError as exc:
                last_status, last_text = "error", str(exc)
                continue
            last_status, last_text = status, text
            if text and status in (None, "finished"):
                if attempt == "subprocess":
                    print(
                        "Note: Cursor via subprocess (prefer kernel:",
                        _LAB_VENV_PYTHON,
                        ")",
                    )
                return text
            if status == "error":
                prune_cursor_agents(api_key, max_active=2)

    active, total = cursor_agent_stats(api_key)
    raise ValueError(
        "Cursor Agent returned no usable text "
        f"(status={last_status!r}, model={model_slug!r}, detail={last_text!r}, "
        f"python={sys.executable!r}, agents_active={active}, agents_total={total}). "
        "Fix: Jupyter kernel → agentic_rest_api/env; Cursor app open; "
        "prune_cursor_agents(..., max_active=2); retry."
    )


def ollama_complete(
    prompt: str,
    model: str,
    *,
    base_url: str | None = None,
    timeout: float = 120,
) -> str:
    import requests

    url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    response = requests.post(
        f"{url}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()
