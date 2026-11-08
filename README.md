# crew_job — resume prepare → tailor for a vacancy

Two-step local pipeline built on **LlamaIndex** (RAG + LlamaCloud Parse) and **CrewAI** (multi-agent tailor).

### What you provide (inputs)

| Input | Path (example) | Role |
|---|---|---|
| **Resume PDF** | `data/Yuliia_Antonova_resume - data_ai.pdf` | source CV as PDF |
| **CV JSON schema** | `data/cv.json` | empty/structured template the RAG fill must match |
| Job posting text | `input/job_posting.txt` | vacancy body for the tailor step (plain text) |

Everything else under `input/` / `output/` is **generated** by the notebooks (filled CV, tailored resume, HTML/PDF previews).

### Pipeline

```text
INPUTS
  resume.pdf  +  cv.json schema
        │
        ▼
llamaindex_rag_competency_matrix_prepare.ipynb   (LlamaIndex / LlamaCloud)
        │
        ├─► input/filled_cv.json      # schema filled from PDF
        ├─► input/filled_cv.md
        └─► output/filled_cv.html
        │
        │   + input/job_posting.txt   # you paste the vacancy
        ▼
L7_job_application_crew.ipynb                    (CrewAI agents)
        │
        ├─► output/tailored_resume.md
        ├─► output/tailored_resume.json   # same shape as cv.json
        └─► output/tailored_resume.html   # → PDF via Chrome headless if needed
```

Kernel: **Python 3.11** (project venv: `.venv311`).

---

## 1. Setup

```bash
cd /Users/yuliiaantonova/PycharmProjects/PythonProject/crew_job
python3.11 -m venv .venv311
source .venv311/bin/activate

# Prepare notebook stack (LlamaIndex / LlamaCloud / embeddings)
# pip install -r requirements-llama-rag.txt   # if you keep such a file

# L7 crewAI (versions from the notebook)
pip install crewai==0.28.8 crewai_tools==0.1.6 langchain_community==0.0.29 \
  openai python-dotenv tiktoken
```

Copy env template and fill real keys:

```bash
cp .env.example .env
```

| Variable | Needed by | What for |
|---|---|---|
| `OPENAI_API_KEY` | **L7** (+ optional prepare) | crewAI LLM; `md_to_cv_json` |
| `LLAMA_CLOUD_API_KEY` | **Prepare** | LlamaCloud Parse of the PDF |
| `LLAMA_CLOUD_BASE_URL` | **Prepare** | usually `https://api.cloud.llamaindex.ai` |
| `CURSOR_API_KEY` | **Prepare** (default LLM) | `helper.get_llm()` when `LLM_BACKEND=cursor` |
| `SERPER_API_KEY` | optional | only if you turn Serper/web tools back on in L7 |
| `OPENAI_MODEL_NAME` | L7 | default in code/notebook: `gpt-4o-mini` |

Optional prepare knobs (via `.env`):

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `cursor` | `cursor` or `openai` |
| `EMBED_BACKEND` | `huggingface` | `huggingface` (local, no credits) or `openai` |
| `EMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | HF embed model |
| `CURSOR_MODEL` | `composer-2.5` | Cursor agent model slug |

`data/` holds (or symlinks to) the **primary inputs**: your resume **PDF** and the **`cv.json` schema**, plus any extra PDFs/templates.

---

## 2. Notebook order

### Step A — `llamaindex_rag_competency_matrix_prepare.ipynb`

**Goal:** take **resume PDF + `cv.json` schema** → fill the schema via LlamaIndex RAG / LlamaCloud Parse.

**Set before / in the run cell:**

| Parameter | Example | Role |
|---|---|---|
| `RESUME_FILE` | `data/<your_resume>.pdf` | **input:** source resume PDF |
| `template_file` | `data/cv.json` | **input:** CV JSON schema/template |

**Also needed in `.env`:** `LLAMA_CLOUD_API_KEY`, `CURSOR_API_KEY` (or switch `LLM_BACKEND=openai` + `OPENAI_API_KEY`).

**You get (intermediates for L7):**

| Path | Content |
|---|---|
| `input/filled_cv.json` | `cv.json` schema filled from the PDF |
| `input/filled_cv.md` | Markdown export for the crew |
| `output/filled_cv.html` | Harvard-style HTML preview |
| SQLite / ids in notebook | `cv_id`, `entities_id`, `job_id`, optional `matrix_id` for a follow-up prompts notebook (if you use it) |

Run top → bottom. After it finishes, spot-check `input/filled_cv.md` / HTML before L7.

---

### Step B — put the vacancy text

Second human input for the tailor step — edit:

```text
input/job_posting.txt
```

Plain text of the job posting (no URL scrape in the current L7 setup — tools are off; text is passed via kickoff inputs).

---

### Step C — `L7_job_application_crew.ipynb`

**Goal:** tailor the (already structured) resume to that vacancy with CrewAI agents (researcher → profiler → resume strategist).

**Needed in `.env`:** `OPENAI_API_KEY` (and `OPENAI_MODEL_NAME` if you want a non-default model).

**Reads prepare outputs + vacancy (not the raw PDF):**

| Path | Role |
|---|---|
| `input/filled_cv.md` | `{resume_text}` — from PDF→schema fill |
| `input/filled_cv.json` | `personal_writeup` via `cv_inputs.build_crew_inputs_from_cv` |
| `input/job_posting.txt` | `{job_posting_text}` |

**Main knobs in the notebook:**

| Setting | Where | Notes |
|---|---|---|
| `OPENAI_MODEL_NAME = 'gpt-4o-mini'` | env cell | crewAI model |
| `RESUME_MD` / `JOB_POSTING_TXT` | tools/inputs cell | already point to `input/` |
| `output_file="output/tailored_resume.md"` | resume strategy task | crew writes Markdown here |
| kickoff cell | `job_application_crew.kickoff(...)` | takes a few minutes; records usage via `run_usage` |

**You get:**

| Path | Content |
|---|---|
| `output/tailored_resume.md` | tailored resume (crew output) |
| `output/tailored_resume.json` | same content mapped into `data/cv.json` shape (`md_to_cv_json`) |
| `output/tailored_resume.html` | HTML render (`render.py`) |
| usage summary | printed by `print_run_usage_summary()` |

Interview prep task is currently commented out; if you re-enable it, it writes `output/interview_materials.md`.

---

## 3. Folder layout

```text
crew_job/
  .env / .env.example
  data/                      # PRIMARY INPUTS (symlink ok)
    <resume>.pdf             #   resume PDF
    cv.json                  #   CV JSON schema/template
  input/                     # prepare outputs + vacancy text
    filled_cv.json           #   filled schema (from PDF)
    filled_cv.md
    job_posting.txt          #   vacancy (you edit)
    fake_resume.md           #   sample / legacy
  output/                    # final / preview artifacts
    filled_cv.html
    tailored_resume.md
    tailored_resume.json     # same shape as data/cv.json
    tailored_resume.html
    tailored_resume.pdf      # optional: Chrome --print-to-pdf
  helper.py                  # prepare: keys, LLM, embeddings
  utils.py                   # L7: OpenAI / Serper keys
  cv_inputs.py
  md_to_cv_json.py
  render.py
  run_usage.py
```

Useful CLI (from project root, venv on):

```bash
python render.py                              # input/filled_cv.json → output/filled_cv.html
python render.py input/filled_cv.json output/preview.html

# HTML → PDF
cd output
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --no-pdf-header-footer \
  --print-to-pdf="$PWD/tailored_resume.pdf" \
  "file://$PWD/tailored_resume.html"
```

---

## 4. Optional / not in the main path

- `L4 (2).ipynb` — older course notebook (LlamaParse form + `data/fake_resume.pdf`). Not required for the prepare → L7 flow.
- Follow-up competency-matrix **prompts** notebook mentioned in prepare (`llamaindex_rag_competency_matrix_prompts_v2.ipynb`) — only if you continue the matrix/SQLite track; copy ids from prepare.

---

## 5. Troubleshooting

- **`401 token_invalidated` / AuthenticationError** — regenerate `OPENAI_API_KEY` (or the key for the backend you use), update `.env`, **restart Jupyter kernel**, re-run from the env cell.
- **Missing GitHub in prepare/L7 inputs** — `filled_cv.json` must have a GitHub link under `basics.links` (`cv_inputs` raises if absent).
- Paths assume CWD = project root (`crew_job/`). Start Jupyter / run notebooks from there.
