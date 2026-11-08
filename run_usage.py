"""Accumulate Cursor (estimated) and Llama Cloud Parse usage for one pipeline run."""

from __future__ import annotations

from typing import Any

_CURSOR: dict[str, int] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "calls": 0,
}
_LLAMA_PARSE: list[dict[str, Any]] = []
_CALLS: list[dict[str, Any]] = []


def reset_run_usage() -> None:
    _CURSOR.update(
        input_tokens=0, output_tokens=0, total_tokens=0, calls=0,
    )
    _LLAMA_PARSE.clear()
    _CALLS.clear()


def _count_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return max(0, len(text or "") // 4)


def estimate_text_tokens(text: str) -> int:
    """Tiktoken cl100k_base count (same basis as §15 when API usage is missing)."""
    return _count_tokens(text)


def summarize_prompt_parts(parts: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Token/char breakdown for prompt debugging (e.g. §5 matrix)."""
    rows: list[dict[str, Any]] = []
    total = 0
    for name, text in parts:
        n = _count_tokens(text or "")
        total += n
        rows.append(
            {
                "part": name,
                "chars": len(text or ""),
                "tokens_est": n,
            },
        )
    rows.append({"part": "TOTAL (sum of parts)", "chars": None, "tokens_est": total})
    return rows


def _usage_from_raw_response(raw: Any) -> tuple[int, int] | None:
    """OpenAI / LlamaIndex usage when present."""
    if raw is None:
        return None
    usage = getattr(raw, "additional_kwargs", None) or {}
    if isinstance(usage, dict):
        u = usage.get("usage") or usage.get("token_usage")
        if isinstance(u, dict):
            inp = u.get("prompt_tokens") or u.get("input_tokens")
            out = u.get("completion_tokens") or u.get("output_tokens")
            if inp is not None and out is not None:
                return int(inp), int(out)
    raw_usage = getattr(raw, "raw", None)
    if raw_usage is not None:
        u = getattr(raw_usage, "usage", None)
        if u is not None:
            inp = getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", None)
            out = getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", None)
            if inp is not None and out is not None:
                return int(inp), int(out)
    return None


def record_llm_completion(
    prompt: str,
    response: str,
    *,
    backend: str = "cursor",
    label: str | None = None,
    raw_response: Any = None,
) -> None:
    """Accumulate token counts (API usage when available, else tiktoken estimate)."""
    parsed = _usage_from_raw_response(raw_response)
    if parsed:
        inp, out = parsed
    else:
        inp = _count_tokens(prompt)
        out = _count_tokens(response or "")
    _CURSOR["input_tokens"] += inp
    _CURSOR["output_tokens"] += out
    _CURSOR["total_tokens"] += inp + out
    _CURSOR["calls"] += 1
    _CALLS.append(
        {
            "call": _CURSOR["calls"],
            "label": label,
            "backend": backend,
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": inp + out,
        }
    )


def record_cursor_completion(
    prompt: str, response: str, *, label: str | None = None
) -> None:
    """Alias for Cursor path (tiktoken when no raw usage)."""
    record_llm_completion(prompt, response, backend="cursor", label=label)


def record_crewai_usage(
    usage_metrics: dict[str, Any] | None,
    *,
    label: str = "crew",
) -> None:
    """
    Fold CrewAI ``crew.usage_metrics`` (after ``kickoff``) into this run's totals.

    CrewAI TokenCalcHandler fields: prompt_tokens, completion_tokens,
    total_tokens, successful_requests (tiktoken estimates; completion may be
    undercounted when the LLM is not streamed).
    """
    if not usage_metrics:
        return
    inp = int(
        usage_metrics.get("prompt_tokens")
        or usage_metrics.get("input_tokens")
        or 0
    )
    out = int(
        usage_metrics.get("completion_tokens")
        or usage_metrics.get("output_tokens")
        or 0
    )
    total = int(usage_metrics.get("total_tokens") or (inp + out))
    calls = int(
        usage_metrics.get("successful_requests")
        or usage_metrics.get("calls")
        or 1
    )
    _CURSOR["input_tokens"] += inp
    _CURSOR["output_tokens"] += out
    _CURSOR["total_tokens"] += total
    _CURSOR["calls"] += calls
    _CALLS.append(
        {
            "call": _CURSOR["calls"],
            "label": label,
            "backend": "crewai",
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": total,
            "successful_requests": calls,
        }
    )


def record_llama_parse_job(
    *,
    job_id: str,
    file_name: str,
    tier: str | None,
    status: str,
    pages: int,
    credits: float | None,
) -> None:
    _LLAMA_PARSE.append(
        {
            "job_id": job_id,
            "file": file_name,
            "tier": tier,
            "status": status,
            "pages": pages,
            "credits": credits,
            "cached": False,
        }
    )


def record_llama_parse_cached(*, file_name: str, index_dir: str) -> None:
    _LLAMA_PARSE.append(
        {
            "file": file_name,
            "cached": True,
            "index_dir": index_dir,
            "pages": None,
            "credits": 0.0,
        }
    )


def get_run_usage_summary() -> dict[str, Any]:
    llama_credits = sum(
        (e.get("credits") or 0)
        for e in _LLAMA_PARSE
        if not e.get("cached")
    )
    return {
        "cursor": dict(_CURSOR),
        "llama_parse": list(_LLAMA_PARSE),
        "llama_parse_credits_total": llama_credits,
        "llm_calls": list(_CALLS),
    }


def format_run_usage_summary() -> str:
    u = get_run_usage_summary()
    c = u["cursor"]
    lines = [
        "=== Usage (this run) ===",
        "LLM (tiktoken estimate; OpenAI uses API usage when available):",
        f"  calls:         {c['calls']}",
        f"  input tokens:  {c['input_tokens']:,}",
        f"  output tokens: {c['output_tokens']:,}",
        f"  total tokens:  {c['total_tokens']:,}",
        "",
        "Llama Cloud Parse:",
    ]
    if not u["llama_parse"]:
        lines.append("  (no parse jobs recorded)")
    else:
        for entry in u["llama_parse"]:
            if entry.get("cached"):
                lines.append(
                    f"  cached index — no new parse "
                    f"({entry.get('file')}, {entry.get('index_dir')})"
                )
                continue
            cred = entry.get("credits")
            cred_s = f"{cred:g}" if cred is not None else "n/a (billing pending)"
            lines.append(
                f"  job {entry.get('job_id')} | {entry.get('file')} | "
                f"tier={entry.get('tier')} | pages={entry.get('pages')} | "
                f"credits={cred_s}"
            )
        if u["llama_parse_credits_total"]:
            lines.append(
                f"  parse credits (sum, when reported): {u['llama_parse_credits_total']:g}"
            )
    calls = u.get("llm_calls") or []
    if calls:
        lines.extend(["", "LLM calls (order of execution):"])
        for row in calls:
            lab = row.get("label") or "prompt"
            lines.append(
                f"  #{row.get('call')} {lab}: "
                f"in={row.get('input_tokens'):,} out={row.get('output_tokens'):,} "
                f"total={row.get('total_tokens'):,}"
            )
    return "\n".join(lines)


def print_run_usage_summary() -> None:
    print(format_run_usage_summary())
