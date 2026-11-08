
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv


def load_env():
    """Load environment variables from .env."""
    env_path = find_dotenv(usecwd=True)

    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()


def get_cursor_api_key():
    load_env()
    api_key = os.getenv("CURSOR_API_KEY")

    if not api_key:
        raise ValueError("CURSOR_API_KEY missing from .env")

    return api_key


def get_openai_api_key():
    """Get OpenAI API key, if using OpenAI separately."""
    load_env()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env")

    return api_key


def get_embed_model():
    """LlamaIndex embedding model for RAG.

    Default: local Hugging Face (no API credits). Set EMBED_BACKEND=openai only
    when you want OpenAI embeddings and have billing active.
    """
    load_env()
    backend = (os.getenv("EMBED_BACKEND") or "huggingface").strip().lower()
    if backend == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(api_key=get_openai_api_key())

    if backend not in ("huggingface", "hf", "local"):
        raise ValueError(f"Unknown EMBED_BACKEND={backend!r} (use huggingface or openai)")

    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    model_name = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    return HuggingFaceEmbedding(model_name=model_name)


def get_llm():
    """LlamaIndex LLM for query engines.

    Default: Cursor Agent (CURSOR_API_KEY). Set LLM_BACKEND=openai to use OpenAI.
    All completions are counted in run_usage (see §15 in prompts notebook).
    """
    load_env()
    backend = (os.getenv("LLM_BACKEND") or "cursor").strip().lower()
    if backend == "openai":
        from llama_index.llms.openai import OpenAI

        from usage_tracking_llm import UsageTrackingLLM

        inner = OpenAI(api_key=get_openai_api_key())
        return UsageTrackingLLM(inner, backend="openai")

    if backend != "cursor":
        raise ValueError(f"Unknown LLM_BACKEND={backend!r} (use cursor or openai)")

    from cursor_llm import CursorLLM
    from usage_tracking_llm import UsageTrackingLLM

    inner = CursorLLM(
        api_key=get_cursor_api_key(),
        model=os.getenv("CURSOR_MODEL", "composer-2.5"),
        rag_cwd=os.getenv("CURSOR_RAG_CWD") or str(Path(__file__).resolve().parent),
    )
    return UsageTrackingLLM(inner, backend="cursor")


def print_llm_step_usage(*, label: str | None = None) -> None:
    """Print tokens for the latest llm_complete / get_llm().complete call."""
    from run_usage import get_run_usage_summary

    summary = get_run_usage_summary()
    calls = summary.get("llm_calls") or []
    if not calls:
        print("LLM usage: (nothing recorded yet — re-run setup, then llm_complete cells)")
        return
    row = calls[-1]
    lab = label or row.get("label") or "prompt"
    total = summary["cursor"]
    print(
        f"LLM usage [{lab}]: +{row['total_tokens']:,} tokens "
        f"({row['input_tokens']:,} in, {row['output_tokens']:,} out) | "
        f"run total: {total['total_tokens']:,} ({total['calls']} calls)"
    )


def _cursor_model_slug(model: str | None) -> str:
    load_env()
    default = (os.getenv("CURSOR_MODEL") or "composer-2.5").strip()
    if not model or model == "cursor":
        return default
    if model.startswith("cursor:"):
        return model.removeprefix("cursor:").strip() or default
    return default


def llm_complete(
    prompt: str,
    *,
    label: str | None = None,
    model: str | None = None,
) -> str:
    """One-shot text completion for labs and scripts.

    ``model`` (optional override):
      - ``None`` — follow ``LLM_BACKEND`` (``cursor`` or ``openai`` in ``.env``)
      - ``cursor`` / ``cursor:<slug>`` — Cursor Agent.prompt (``CURSOR_MODEL`` default)
      - ``openai`` — ``get_llm()`` when ``LLM_BACKEND=openai``
      - ``ollama:<name>`` — local Ollama HTTP API
    """
    from cursor_llm import (
        cursor_prompt_complete,
        ollama_complete,
        register_lab_usage_llm,
        reset_lab_cursor_llm,
        strip_llm_fence,
    )
    from run_usage import record_llm_completion

    load_env()
    choice = model.strip() if model else None

    if choice and choice.startswith("ollama:"):
        name = choice.removeprefix("ollama:").strip()
        text = strip_llm_fence(ollama_complete(prompt, name))
        record_llm_completion(prompt, text, backend="ollama", label=label)
        print_llm_step_usage(label=label)
        return text

    if choice in ("openai",) or (choice and choice.startswith("openai:")):
        kwargs = {"label": label} if label else {}
        llm = get_llm()
        text = (llm.complete(prompt, **kwargs).text or "").strip()
        if not text:
            raise ValueError("OpenAI LLM returned empty text.")
        text = strip_llm_fence(text)
        print_llm_step_usage(label=label)
        return text

    backend = (os.getenv("LLM_BACKEND") or "cursor").strip().lower()
    if not choice and backend == "openai":
        kwargs = {"label": label} if label else {}
        text = (get_llm().complete(prompt, **kwargs).text or "").strip()
        if not text:
            raise ValueError("OpenAI LLM returned empty text.")
        text = strip_llm_fence(text)
        print_llm_step_usage(label=label)
        return text

    if choice and choice not in ("cursor",) and not choice.startswith("cursor:"):
        raise ValueError(
            f"Unknown model {model!r}; use cursor, openai, or ollama:<name>"
        )

    llm = get_llm()
    register_lab_usage_llm(llm)
    reset_lab_cursor_llm()
    slug = _cursor_model_slug(choice)
    text = cursor_prompt_complete(
        prompt,
        api_key=get_cursor_api_key(),
        model_slug=slug,
    )
    record_llm_completion(prompt, text, backend="cursor", label=label)
    print_llm_step_usage(label=label)
    return text


def get_llama_cloud_api_key():
    """Get LlamaCloud API key for Parse API v2."""
    load_env()

    api_key = os.getenv("LLAMA_CLOUD_API_KEY")

    if not api_key:
        raise ValueError("LLAMA_CLOUD_API_KEY not found in .env")

    return api_key


def extract_html_content(filename):
    """Read an HTML file and wrap it in a scrollable container."""

    path = Path(filename)

    if not path.is_file():
        raise FileNotFoundError(f"HTML file not found: {filename}")

    try:
        html_content = path.read_text(encoding="utf-8")

        return f"""
        <div style="width: 100%; height: 800px; overflow: auto;">
            {html_content}
        </div>
        """

    except OSError as e:
        raise RuntimeError(f"Error reading HTML file: {e}") from e
