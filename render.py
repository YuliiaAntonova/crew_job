# /// script
# requires-python = ">=3.11"
# ///
"""
Render filled_cv.json (or any CV JSON) into Harvard-format, print-ready HTML.

    python render.py                          # input/filled_cv.json -> output/filled_cv.html
    python render.py data/cv.json out.html
    python render.py input/filled_cv.json --variant fde

Open the HTML in Chrome -> Print -> Save as PDF (A4 or Letter).
"""
from __future__ import annotations

import argparse
import html
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent

DEFAULT_JSON = HERE / "input" / "filled_cv.json"
DEFAULT_HTML = HERE / "output" / "filled_cv.html"


def esc(s: object) -> str:
    if s is None:
        return ""
    return html.escape(str(s), quote=True)


URL_RE = re.compile(r"https?://[^\s<>\"']+[^\s<>\"'.,;:!?)\]]")


def linkify(s: object) -> str:
    text = "" if s is None else str(s)
    out, pos = [], 0
    for m in URL_RE.finditer(text):
        out.append(esc(text[pos:m.start()]))
        url = m.group(0)
        out.append(f'<a class="link" href="{esc(url)}">{esc(url)}</a>')
        pos = m.end()
    out.append(esc(text[pos:]))
    return "".join(out)


def visible(entry: dict, variant: str | None) -> bool:
    if not variant:
        return True
    tag = entry.get("variant")
    if not tag:
        return True
    tags = [tag] if isinstance(tag, str) else list(tag)
    return variant in tags


def _clean_project_context(text: object) -> str:
    return str(text or "").strip()


def _project_context_text(entry: dict) -> str:
    return _clean_project_context(
        entry.get("context") or entry.get("project_context") or ""
    )


def _interests_text(cv: dict) -> str:
    if cv.get("interests"):
        return str(cv["interests"]).strip()
    note = (cv.get("basics") or {}).get("note") or ""
    note = str(note).strip()
    if note.lower().startswith("interests:"):
        return note.split(":", 1)[1].strip()
    return ""


def render_header(b: dict) -> str:
    links = []
    for ln in b.get("links") or []:
        if not isinstance(ln, dict):
            continue
        url = (ln.get("url") or "").strip()
        # Prefer platform label (GitHub / LinkedIn / Xing) over handle/text
        label = (ln.get("label") or "").strip()
        handle = (ln.get("text") or "").strip()
        visible = label or handle or url
        if url:
            title = handle if label and handle else url
            links.append(
                f'<a href="{esc(url)}" title="{esc(title)}">{esc(visible)}</a>'
            )
        elif visible:
            links.append(esc(visible))

    contact = [esc(b["location"])] if b.get("location") else []
    if b.get("email"):
        contact.append(f'<a href="mailto:{esc(b["email"])}">{esc(b["email"])}</a>')
    if b.get("phone"):
        contact.append(esc(b["phone"]))
    contact += links

    parts = [f'<h1>{esc(b.get("name", ""))}</h1>']
    if b.get("headline"):
        parts.append(f'<p class="headline">{esc(b["headline"])}</p>')
    if contact:
        parts.append('<p class="contact">' + ' <span>&bull;</span> '.join(contact) + "</p>")
    if b.get("note"):
        parts.append(f'<p class="note">{esc(b["note"])}</p>')
    return f'<header>{"".join(parts)}</header>'


def section(title: str, body: str) -> str:
    if not body.strip():
        return ""
    return f"<section><h2>{esc(title)}</h2>{body}</section>"


def render_summary(text: str) -> str:
    return section("Summary", f'<p class="summary">{esc(text)}</p>')


def render_skills(groups: list) -> str:
    if not groups:
        return ""

    rows = []
    for g in groups:
        if not isinstance(g, dict):
            continue
        items = [str(x) for x in (g.get("items") or []) if x is not None and str(x).strip()]
        if not g.get("group") and not items:
            continue
        rows.append(
            f'<div class="skill-row"><span class="skill-group">{esc(g.get("group", ""))}</span>'
            f'<span class="skill-items">{esc(", ".join(items))}</span></div>'
        )
    if not rows:
        return ""
    return section("Skills", f'<div class="skills">{"".join(rows)}</div>')


def _entry(
    title_left,
    title_right,
    sub_left,
    sub_right,
    bullets,
    context: str = "",
    title_left_html: str | None = None,
) -> str:
    def bl(bs):
        clean = [x for x in (bs or []) if x is not None and str(x).strip()]
        return "<ul>" + "".join(f"<li>{linkify(x)}</li>" for x in clean) + "</ul>" if clean else ""

    left = title_left_html if title_left_html is not None else esc(title_left)
    head = (
        f'<div class="entry-head"><span class="left">{left}</span>'
        f'<span class="right">{esc(title_right)}</span></div>'
    )
    sub = ""
    if sub_left or sub_right:
        sub = (
            f'<div class="entry-sub"><span class="left">{esc(sub_left)}</span>'
            f'<span class="right">{esc(sub_right)}</span></div>'
        )
    ctx = ""
    if context:
        ctx = (
            f'<p class="project-context">'
            f'<span class="project-context-label">Project Context:</span> '
            f"{esc(context)}</p>"
        )
    return f'<div class="entry">{head}{sub}{ctx}{bl(bullets)}</div>'


def render_experience(items: list, variant) -> str:
    body = "".join(
        _entry(
            e.get("company", ""),
            e.get("location", ""),
            e.get("title", ""),
            e.get("period", ""),
            e.get("bullets", []),
            context=_project_context_text(e),
        )
        for e in items
        if isinstance(e, dict) and visible(e, variant)
    )
    return section("Experience", body)


def render_earlier(items: list) -> str:
    if not items:
        return ""

    rows = "".join(
        f'<div class="earlier-row"><span class="left">'
        f'<strong>{esc(e.get("company", ""))}</strong> — {esc(e.get("role", ""))}</span>'
        f'<span class="right">{esc(e.get("period", ""))}</span></div>'
        f'<div class="earlier-stack">{esc(", ".join(e.get("stack") or []))}'
        f'{" · " + esc(e["location"]) if e.get("location") else ""}</div>'
        for e in items
        if isinstance(e, dict)
    )
    return section("Earlier Experience", f'<div class="earlier">{rows}</div>')


def _project_has_content(project: dict) -> bool:
    name = str(project.get("name") or "").strip()
    bullets = [str(x).strip() for x in (project.get("bullets") or []) if str(x).strip()]
    return bool(name or bullets)


def _project_title_html(project: dict) -> str:
    name = str(project.get("name") or "").strip()
    url = str(project.get("url") or "").strip()
    if url:
        return f'<a class="link" href="{esc(url)}">{esc(name or url)}</a>'
    return linkify(name)


def render_projects(items: list, variant) -> str:
    body = "".join(
        _entry("", "", "", "", p.get("bullets", []), title_left_html=_project_title_html(p))
        for p in items
        if isinstance(p, dict) and visible(p, variant) and _project_has_content(p)
    )
    return section("Selected Projects", body)


def render_education(items: list) -> str:
    body = "".join(
        _entry(
            e.get("degree", ""),
            e.get("period", ""),
            e.get("school", ""),
            "",
            [e["note"]] if e.get("note") else [],
        )
        for e in items
        if isinstance(e, dict)
    )
    return section("Education", body)


def render_certifications(items: list) -> str:
    if not items:
        return ""
    rows = []
    for c in items:
        if isinstance(c, str):
            rows.append(f'<div class="cert-row"><span class="left">{esc(c)}</span></div>')
            continue
        if not isinstance(c, dict):
            continue
        right = " · ".join(esc(c[k]) for k in ("issuer", "year") if c.get(k))
        rows.append(
            f'<div class="cert-row"><span class="left">{esc(c.get("name", ""))}</span>'
            f'<span class="right">{right}</span></div>'
        )
    return section("Certifications", "".join(rows))


def render_languages(items: list) -> str:
    if not items:
        return ""
    parts = []
    for lang in items:
        if isinstance(lang, str):
            parts.append(esc(lang))
        elif isinstance(lang, dict):
            lvl = (
                f' <span class="lang-level">({esc(lang["level"])})</span>'
                if lang.get("level")
                else ""
            )
            parts.append(f'{esc(lang.get("name", ""))}{lvl}')
    if not parts:
        return ""
    return section(
        "Languages",
        '<p class="languages">' + " <span>&bull;</span> ".join(parts) + "</p>",
    )


def render_recognition(rec: dict) -> str:
    if not rec:
        return ""
    parts = []
    for bk in rec.get("books", []):
        if not isinstance(bk, dict):
            continue
        label = f'{bk.get("title", "")} ({bk.get("publisher", "")}, {bk.get("year", "")})'
        link = f'<a href="{esc(bk["url"])}">{esc(label)}</a>' if bk.get("url") else esc(label)
        parts.append(f"<li>{link}</li>")
    for h in rec.get("highlights", []):
        parts.append(f"<li>{esc(h)}</li>")
    body = f"<ul>{''.join(parts)}</ul>" if parts else ""
    return section("Publications & Recognition", body)


def render_interests(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    lines = [ln.strip().lstrip("- ") for ln in raw.splitlines() if ln.strip()]
    if len(lines) > 1:
        body = "<ul>" + "".join(f"<li>{esc(ln)}</li>" for ln in lines) + "</ul>"
    else:
        body = f'<p class="interests">{esc(lines[0] if lines else raw)}</p>'
    return section("Interests", body)


def build_html(cv: dict, variant: str | None, fragment: bool = False) -> str:
    b = cv.get("basics") or {}
    body = "".join(
        [
            render_header(b),
            render_summary(b["summary"]) if b.get("summary") else "",
            render_skills(cv.get("skills", [])),
            render_experience(cv.get("experience", []), variant),
            render_earlier(cv.get("experience_earlier", [])),
            render_projects(cv.get("selected_projects") or cv.get("projects") or [], variant),
            render_education(cv.get("education", [])),
            render_certifications(cv.get("certifications", [])),
            render_languages(cv.get("languages", [])),
            render_recognition(cv.get("recognition", {})),
            render_interests(_interests_text(cv)),
        ]
    )
    page_class = "page compact" if b.get("density") == "compact" else "page"
    inner = f'<style>{CSS}</style>\n<main class="{page_class}">\n{body}\n</main>'
    if fragment:
        return inner
    title = f'{b.get("name", "CV")} — CV'
    return HTML_SHELL.format(title=esc(title), css=CSS, body=body, page_class=page_class)


HTML_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<main class="{page_class}">
{body}
</main>
</body>
</html>
"""

CSS = r"""
:root {
  --ink: #111;
  --muted: #444;
  --rule: #111;
  --accent: #111;
  --max: 800px;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: #ececec; color: var(--ink); }
body {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 10.5pt;
  line-height: 1.34;
}
a { color: inherit; text-decoration: none; }
a:hover { text-decoration: underline; }
a.link { text-decoration: underline; }

.page {
  max-width: var(--max);
  margin: 24px auto;
  background: #fff;
  padding: 40px 48px;
  box-shadow: 0 1px 8px rgba(0,0,0,.15);
}

header { text-align: center; margin-bottom: 12px; }
h1 {
  font-size: 22pt; margin: 0 0 2px; letter-spacing: .5px;
  font-weight: 700;
}
.headline { margin: 2px 0 5px; font-size: 11pt; color: var(--muted); font-style: italic; }
.contact { margin: 0; font-size: 9.5pt; color: var(--muted); }
.contact span { color: #999; padding: 0 2px; }
.note { margin: 3px 0 0; font-size: 9.5pt; color: var(--muted); }
.interests { margin: 0; }

section { margin-top: 13px; }
h2 {
  font-size: 11pt; text-transform: uppercase; letter-spacing: 1.2px;
  font-weight: 700; margin: 0 0 6px;
  border-bottom: 1.3px solid var(--rule); padding-bottom: 2px;
  break-after: avoid; page-break-after: avoid;
}
p.summary { margin: 0; text-align: justify; }

.entry { margin-bottom: 9px; page-break-inside: avoid; }
.entry:last-child { margin-bottom: 0; }
.entry-head, .entry-sub, .earlier-row {
  display: flex; justify-content: space-between; gap: 12px;
}
.entry-head .left { font-weight: 700; }
.entry-head .right { font-weight: 400; color: var(--muted); white-space: nowrap; }
.entry-sub { margin-top: 0; }
.entry-sub .left { font-style: italic; }
.entry-sub .right { color: var(--muted); white-space: nowrap; }
.project-context {
  margin: 3px 0 0;
  font-size: 9.5pt;
  color: var(--muted);
  font-style: italic;
}
.project-context + ul { margin-top: 7px; }
.project-context-label { font-style: normal; font-weight: 700; color: var(--ink); }

ul { margin: 4px 0 0; padding-left: 18px; }
li { margin: 1.5px 0; }

.skill-row { display: flex; gap: 10px; margin: 2px 0; }
.skill-group { flex: 0 0 155px; font-weight: 700; }
.skill-items { flex: 1; }

.cert-row {
  display: flex; justify-content: space-between; gap: 12px; margin: 2px 0;
}
.cert-row .left { font-weight: 400; }
.cert-row .right { color: var(--muted); white-space: nowrap; }

p.languages { margin: 0; }
p.languages span { padding: 0 2px; }
p.languages .lang-level { color: #555; font-style: italic; }

.earlier-row { margin-top: 5px; }
.earlier-row .right { color: var(--muted); white-space: nowrap; }
.earlier-stack { color: var(--muted); font-size: 9pt; font-style: italic; margin-bottom: 2px; }

.compact { font-size: 10pt; line-height: 1.26; }
.compact section { margin-top: 7px; }
.compact h2 { margin-bottom: 4px; }
.compact .entry { margin-bottom: 6px; }
.compact ul { margin-top: 3px; }
.compact li { margin: 1px 0; }
.compact p.summary { text-align: left; }

@page { size: A4; margin: 13mm 15mm; }
@media print {
  html, body { background: #fff; }
  .page { max-width: none; margin: 0; padding: 0; box-shadow: none; }
  a { color: inherit; }
  section { page-break-inside: auto; }
}
"""


def load_cv_json(src: pathlib.Path) -> dict:
    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("CV JSON must be an object at the root")
    return data


def render_cv_json_to_html(
    src: pathlib.Path | str,
    out: pathlib.Path | str,
    *,
    variant: str | None = None,
    fragment: bool = False,
) -> pathlib.Path:
    src_path = pathlib.Path(src).expanduser().resolve()
    out_path = pathlib.Path(out).expanduser().resolve()
    cv = load_cv_json(src_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(cv, variant, fragment=fragment), encoding="utf-8")
    return out_path


def _join_meta(*parts: object) -> str:
    return " · ".join(str(p).strip() for p in parts if p)


def cv_to_markdown(cv: dict, *, variant: str | None = None) -> str:
    """Convert filled CV JSON dict to a readable Markdown string."""
    lines: list[str] = []
    b = cv.get("basics") or {}
    name = b.get("name") or "Resume"
    lines.append(f"# {name}")
    lines.append("")
    if b.get("headline"):
        lines.append(f"**{b['headline']}**")
        lines.append("")
    meta = _join_meta(b.get("location"), b.get("email"), b.get("phone"))
    if meta:
        lines.append(meta)
        lines.append("")
    for link in b.get("links") or []:
        label = link.get("label") or link.get("text") or "Link"
        url = link.get("url") or ""
        if url:
            lines.append(f"- [{label}]({url})")
    if b.get("links"):
        lines.append("")
    if b.get("summary"):
        lines.append(str(b["summary"]).strip())
        lines.append("")

    skills = cv.get("skills") or []
    if skills:
        lines.append("## Skills")
        lines.append("")
        for group in skills:
            if not isinstance(group, dict):
                continue
            title = group.get("group") or "Skills"
            items = group.get("items") or []
            if items:
                lines.append(f"**{title}:** {', '.join(str(i) for i in items)}")
                lines.append("")

    experience = [
        e for e in (cv.get("experience") or []) if isinstance(e, dict) and visible(e, variant)
    ]
    if experience:
        lines.append("## Experience")
        lines.append("")
        for job in experience:
            title = job.get("title") or job.get("role") or "Role"
            company = job.get("company") or ""
            heading = f"### {title}" + (f" — {company}" if company else "")
            lines.append(heading)
            meta = _join_meta(job.get("period"), job.get("location"))
            if meta:
                lines.append(f"*{meta}*")
                lines.append("")
            if job.get("project_context"):
                lines.append(str(job["project_context"]).strip())
                lines.append("")
            for bullet in job.get("bullets") or []:
                lines.append(f"- {bullet}")
            lines.append("")

    earlier = cv.get("experience_earlier") or []
    if earlier:
        lines.append("## Earlier Experience")
        lines.append("")
        for job in earlier:
            if not isinstance(job, dict):
                continue
            stack = job.get("stack") or []
            stack_s = f" ({', '.join(str(s) for s in stack)})" if stack else ""
            role = job.get("role") or job.get("title") or ""
            company = job.get("company") or ""
            period = job.get("period") or ""
            lines.append(f"- **{period}** — {role}, {company}{stack_s}".rstrip(" ,"))
        lines.append("")

    projects = [
        p
        for p in (cv.get("selected_projects") or [])
        if isinstance(p, dict) and visible(p, variant)
    ]
    if projects:
        lines.append("## Selected Projects")
        lines.append("")
        for proj in projects:
            name = proj.get("name") or "Project"
            url = proj.get("url")
            lines.append(f"### [{name}]({url})" if url else f"### {name}")
            lines.append("")
            for bullet in proj.get("bullets") or []:
                lines.append(f"- {bullet}")
            lines.append("")

    education = cv.get("education") or []
    if education:
        lines.append("## Education")
        lines.append("")
        for edu in education:
            if not isinstance(edu, dict):
                continue
            degree = edu.get("degree") or ""
            school = edu.get("school") or ""
            period = edu.get("period")
            note = edu.get("note")
            line = f"- **{degree}** — {school}" if school else f"- **{degree}**"
            if period:
                line += f", {period}"
            if note:
                line += f" ({note})"
            lines.append(line)
        lines.append("")

    certs = cv.get("certifications") or []
    if certs:
        lines.append("## Certifications")
        lines.append("")
        for c in certs:
            lines.append(f"- {c}")
        lines.append("")

    languages = cv.get("languages") or []
    if languages:
        lines.append("## Languages")
        lines.append("")
        for lang in languages:
            if isinstance(lang, dict):
                lines.append(f"- **{lang.get('name', '')}** — {lang.get('level', '')}")
            else:
                lines.append(f"- {lang}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_cv_json_to_md(
    src: pathlib.Path | str = "input/filled_cv.json",
    out: pathlib.Path | str = "input/filled_cv.md",
    *,
    variant: str | None = None,
) -> tuple[str, pathlib.Path]:
    """Write Markdown from CV JSON; return (markdown_text, output_path)."""
    src_path = pathlib.Path(src).expanduser().resolve()
    out_path = pathlib.Path(out).expanduser().resolve()
    md = cv_to_markdown(load_cv_json(src_path), variant=variant)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    return md, out_path


def resolve_paths(src: str | None, out: str | None) -> tuple[pathlib.Path, pathlib.Path]:
    if src is None:
        return DEFAULT_JSON, DEFAULT_HTML
    sp = pathlib.Path(src)
    if sp.suffix.lower() == ".json":
        src_path = sp
    else:
        src_path = sp / "input" / "filled_cv.json"
    if out:
        out_path = pathlib.Path(out)
    else:
        out_path = src_path.with_suffix(".html")
        if out_path.parent.name == "input":
            out_path = HERE / "output" / out_path.name
    return src_path, out_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Render CV JSON (default: input/filled_cv.json) to Harvard-format HTML.",
        epilog="Examples: `render.py`  |  `render.py input/filled_cv.json`  |  "
        "`render.py data/cv.json cv.html --variant fde`",
    )
    ap.add_argument("src", nargs="?", help="input JSON path (default: input/filled_cv.json)")
    ap.add_argument("out", nargs="?", help="output HTML (default: output/<name>.html)")
    ap.add_argument(
        "--variant",
        choices=["applied-ai", "ai-platform", "fde"],
        help="filter entries tagged with this variant",
    )
    ap.add_argument(
        "--fragment",
        action="store_true",
        help="emit style+body only (no full HTML document)",
    )
    args = ap.parse_args()

    src_path, out_path = resolve_paths(args.src, args.out)
    if not src_path.is_file():
        print(f"Input not found: {src_path}", file=sys.stderr)
        return 1
    render_cv_json_to_html(src_path, out_path, variant=args.variant, fragment=args.fragment)
    written = out_path.read_text(encoding="utf-8")
    print(
        f"Wrote {out_path} ({len(written):,} bytes)"
        + (f" [variant: {args.variant}]" if args.variant else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
