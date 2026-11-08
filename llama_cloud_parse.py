"""Thin wrapper: Llama Cloud SDK v2 parse API with LlamaParse-like `.load_data()`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

from llama_cloud import LlamaCloud
from llama_cloud._polling import poll_until_complete
from llama_index.core import Document

from run_usage import record_llama_parse_job

ResultType = Literal["markdown", "text"]


class LlamaCloudParse:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        result_type: ResultType = "markdown",
        content_guideline_instruction: str | None = None,
        tier: Literal["fast", "cost_effective", "agentic", "agentic_plus"] = "agentic",
        version: str = "latest",
        timeout: float = 600,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/") if base_url else None
        self._result_type = result_type
        self._custom_prompt = content_guideline_instruction
        self._tier = tier
        self._version = version
        self._timeout = timeout

    def _client(self) -> LlamaCloud:
        kwargs: dict = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return LlamaCloud(**kwargs)

    def load_data(self, file_path: str | Path | Sequence[str | Path]) -> list[Document]:
        paths = [file_path] if isinstance(file_path, (str, Path)) else list(file_path)
        docs: list[Document] = []
        for path in paths:
            docs.extend(self._load_one(Path(path)))
        return docs

    def _load_one(self, pdf_path: Path) -> list[Document]:
        client = self._client()
        agentic_options = (
            {"custom_prompt": self._custom_prompt} if self._custom_prompt else None
        )
        expand = ["markdown"] if self._result_type == "markdown" else ["text"]
        output_options = (
            {"markdown": {}} if self._result_type == "markdown" else {"spatial_text": {}}
        )

        with pdf_path.open("rb") as fh:
            job = client.parsing.create(
                tier=self._tier,
                version=self._version,
                upload_file=fh,
                agentic_options=agentic_options,
                output_options=output_options,
            )

        result = poll_until_complete(
            lambda: client.parsing.get(job.id, expand=expand),
            is_complete_fn=lambda r: r.job.status == "COMPLETED",
            is_error_fn=lambda r: r.job.status in ("FAILED", "CANCELLED"),
            get_error_message_fn=lambda r: r.job.error_message or r.job.status,
            timeout=self._timeout,
        )

        pages = 0
        if self._result_type == "markdown":
            if not result.markdown:
                raise RuntimeError("Parse finished without markdown.")
            pages = len(result.markdown.pages)
            docs = _pages_to_documents(result.markdown.pages, "markdown")
        elif not result.text:
            raise RuntimeError("Parse finished without text.")
        else:
            pages = len(result.text.pages)
            docs = _pages_to_documents(result.text.pages, "text")

        usage = result.job.usage
        record_llama_parse_job(
            job_id=result.job.id,
            file_name=pdf_path.name,
            tier=result.job.tier,
            status=result.job.status,
            pages=pages,
            credits=usage.credits if usage else None,
        )
        return docs


# Drop-in name if you prefer the old import style:
LlamaParse = LlamaCloudParse


def _pages_to_documents(pages, text_attr: str) -> list[Document]:
    docs: list[Document] = []
    for page in pages:
        text = getattr(page, text_attr, None)
        if getattr(page, "success", False) and text:
            docs.append(Document(text=text, metadata={"page": page.page_number}))
    return docs
