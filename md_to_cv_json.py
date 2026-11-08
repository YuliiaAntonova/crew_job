"""Map a free-form tailored resume Markdown into data/cv.json shape."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from template_extractor import get_target_fields, validate_shape
from utils import get_openai_api_key


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _cert_to_scalar(item: Any) -> str:
    """Flatten LLM cert objects into one display string."""
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return str(item)
    name = str(item.get("name") or item.get("title") or "").strip()
    issuer = str(item.get("issuer") or item.get("organization") or "").strip()
    when = str(
        item.get("date")
        or item.get("year")
        or item.get("period")
        or item.get("issued")
        or ""
    ).strip()
    if name and issuer and name.lower().startswith(issuer.lower()):
        base = name
    elif name and issuer:
        base = f"{issuer} - {name}"
    else:
        base = name or issuer or json.dumps(item, ensure_ascii=False)
    if when and when not in base:
        return f"{base} ({when})"
    return base


def _coerce_empty_array_fields(filled: dict[str, Any], template: dict[str, Any]) -> None:
    """
    Template fields that are [] must be lists of scalars (see validate_shape).
    LLMs often return certifications as objects — flatten in place.
    """
    for key, shape in template.items():
        if shape != [] or key not in filled:
            continue
        raw = filled[key]
        if not isinstance(raw, list):
            continue
        if key == "certifications":
            filled[key] = [_cert_to_scalar(x) for x in raw if x is not None]
        else:
            filled[key] = [
                x if not isinstance(x, (dict, list)) else json.dumps(x, ensure_ascii=False)
                for x in raw
            ]


def markdown_to_cv_json(
    md_path: str | Path = "output/tailored_resume.md",
    template_path: str | Path = "data/cv.json",
    out_path: str | Path = "output/tailored_resume.json",
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Use OpenAI to fit tailored Markdown into the CV JSON template.
    Does not invent facts; missing fields stay null / [].
    """
    md_path = Path(md_path)
    template_path = Path(template_path)
    out_path = Path(out_path)

    md = md_path.read_text(encoding="utf-8")
    template = json.loads(template_path.read_text(encoding="utf-8"))
    shape = get_target_fields(template_path)
    model_name = model or os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")

    prompt = f"""
You convert a tailored resume written in Markdown into ONE JSON object.

Rules:
- Keep EXACTLY the target keys and hierarchy; add no keys.
- Fill values only from the Markdown; never invent employers, dates, skills, or degrees.
- Replace null placeholders with facts from the Markdown; missing scalars remain null.
- Expand arrays of objects to one object per relevant record.
- Fill empty arrays with relevant scalar strings when Markdown lists them; keep [] if unsupported.
- certifications must be a JSON array of strings only
  (e.g. "DeepLearning.ai - Agentic AI (May 2026)"), never objects.
- Map "Professional Summary" / Summary into basics.summary.
- Map Skills groups into skills[].group + skills[].items.
- skills[].items must be atomic keywords/short phrases, never full sentences.
  If a skill line is prose that lists domains/industries (e.g. "Experience in
  X, Y, and Z industries, with a focus on …"), extract only the named domains
  as separate items and drop the filler wording.
- Map jobs into experience[] (title, company, location, period, project_context, bullets).
- Earlier / older roles may go to experience_earlier[] when clearly secondary.
- Links (GitHub, LinkedIn, Xing) go to basics.links[] with:
  label = platform name (GitHub / LinkedIn / Xing),
  text = profile handle or display name,
  url = full https URL (never omit the URL).
- Return valid JSON only (no Markdown fences).

TARGET JSON TEMPLATE:
{json.dumps(template, ensure_ascii=False, indent=2)}

TAILORED RESUME MARKDOWN:
{md}
"""

    client = OpenAI(api_key=get_openai_api_key())
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You output only valid JSON matching the given CV template.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    filled = json.loads(_strip_code_fence(raw))
    _coerce_empty_array_fields(filled, template)
    validate_shape(filled, shape)

    out_path.write_text(
        json.dumps(filled, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return filled
