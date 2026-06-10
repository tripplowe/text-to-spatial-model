# text-to-model

Turn a plain-English GIS question into a structured **ArcGIS Pro vector-overlay
workflow** — a step-by-step geoprocessing model, the matching **ArcPy code**, a
keyword breakdown, and teaching notes explaining *why* each tool was chosen.

It is a **teaching tool** for an introductory GIS course in natural resource
management: a student types something like *"How many acres of each land cover
type within 200 feet of a stream?"* and gets back the workflow
(Buffer → Clip → Calculate Geometry Attributes → Summary Statistics) as a diagram
plus runnable ArcPy — making the reasoning of building a ModelBuilder workflow
visible. It does **not** execute the analysis; it teaches how to construct it.

Everything runs against an LLM **you provide** — either a model running locally on
your own machine (via [Ollama](https://ollama.com)) or any hosted
OpenAI-compatible API with your own key. No cloud account is required to develop
with it locally.

---

## What it does

Given a natural-language request, the app returns JSON containing:

- **keywords** — which words mapped to which tool/parameter/input, and why
- **workflow** — nodes (inputs, tools, derived data, output) and edges, forming
  the geoprocessing graph
- **ArcPy script** — the workflow as runnable ArcGIS Python, in execution order
- **notes** — assumptions, ambiguities, alternatives, and teaching points

A built-in **validation layer** catches common LLM mistakes (out-of-scope tools,
hallucinated tool names, incomplete graphs, self-clips, geometry/measure
mismatches) and adds corrective warnings.

**Scope (current):** vector overlay analysis only — ten tools: Buffer, Clip,
Intersect, Erase, Union, Dissolve, Select Layer By Attribute, Select Layer By
Location, Summary Statistics, Calculate Geometry Attributes. Raster/terrain tools
are intentionally out of scope.

---

## Requirements

- **Python 3.11+**
- **An LLM**, one of:
  - **Local:** [Ollama](https://ollama.com) with a pulled model (recommended:
    `qwen3:30b-a3b-instruct-2507-q4_K_M` if you have ~20 GB RAM/VRAM; a smaller
    `llama3.1:8b-instruct-q8_0` works but produces weaker multi-step workflows), **or**
  - **Hosted:** any OpenAI-compatible API (OpenAI, Together, Groq, OpenRouter, a
    hosted vLLM endpoint, …) and your own API key.
- *(Optional)* Docker, if you prefer containers over a local Python env.

> **Note on model quality:** this task needs a model that reliably returns valid
> JSON and follows multi-step instructions. In testing, 8B–14B local models
> frequently truncated workflows or invented tool names; a ~30B-class model was
> markedly more reliable. Use a capable model for real work; small models are fine
> for quick experimentation.

---

## Quick start

### 1. Clone and configure

```bash
git clone https://github.com/tripplowe/text-to-model.git
cd text-to-model
cp .env.example .env          # then edit .env for your setup
```

### 2A. Run with local Ollama (default)

```bash
# install and start Ollama (see https://ollama.com), then pull a model:
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M

# install Python deps and run:
pip install -r requirements.txt
python app.py
```

Open <http://localhost:8080>.

### 2B. Run with a hosted API (your own key)

Edit `.env`:

```ini
LLM_BACKEND=openai
OPENAI_BASE_URL=https://api.openai.com/v1
API_KEY=sk-your-key-here
MODEL_NAME=gpt-4o-mini
```

Then:

```bash
pip install -r requirements.txt
python app.py
```

Open <http://localhost:8080>. (No Ollama needed in this mode.)

### Or run with Docker

```bash
cp .env.example .env
docker compose up -d --build
docker exec -it ttm-ollama ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M
```

Open <http://localhost:8080>. To use a GPU, uncomment the `gpus:` block in
`docker-compose.yml`. To use a hosted API instead of the bundled Ollama, set
`LLM_BACKEND=openai` (etc.) in `.env` and run only the app:
`docker compose up -d --build app`.

---

## Configuration

All settings come from environment variables (or a local `.env`). The most
important:

| Variable | Purpose | Example |
|----------|---------|---------|
| `LLM_BACKEND` | `ollama`, `vllm`, `sglang`, or `openai` | `ollama` |
| `MODEL_NAME` | model to use | `qwen3:30b-a3b-instruct-2507-q4_K_M` |
| `OLLAMA_HOST` | Ollama URL (local backend) | `http://localhost:11434` |
| `OPENAI_BASE_URL` | API root (hosted backend) | `https://api.openai.com/v1` |
| `API_KEY` | secret key (hosted backend) | `sk-…` *(never commit)* |
| `NUM_CTX` | context window for Ollama | `24576` |
| `TEMPERATURE` | sampling temperature | `0.1` |
| `BIND_HOST` / `BIND_PORT` | where the server listens | `127.0.0.1` / `8080` |

> **`NUM_CTX` matters.** The system prompt is ~17K tokens. Many Ollama models
> default to a 4096-token context and will reject every request with a
> "context size" error. Keep `NUM_CTX` above ~20000.

See `.env.example` for the fully-commented list.

---

## How it works

```
Browser (chat UI)
   │  query (+ optional layer list)
   ▼
app.py (FastAPI)
   │  assembles prompt: system_prompt.txt + gis_ontology.json + few-shot examples
   ▼
LLM backend (Ollama / vLLM / SGLang / hosted API)
   │  returns JSON: keywords, workflow (nodes/edges), notes
   ▼
validate_workflow()  ← geospatial sanity checks
   ▼
UI renders: keyword table, workflow diagram, ArcPy script, notes, warnings
```

Three files shape behavior:
- **`system_prompt.txt`** — instructions, output schema, and worked few-shot
  examples (most behavior lives here).
- **`gis_ontology.json`** — the tool "textbook": tools, inputs/outputs,
  parameters, ArcPy syntax, and trigger vocabulary.
- **`app.py`** — FastAPI server: prompt assembly, LLM call, JSON parse,
  validation, and the single-page web UI.

> Editing `app.py`, `system_prompt.txt`, or `gis_ontology.json` and running in
> Docker requires a rebuild: `docker compose up -d --build app`.

---

## Optional: declare your data layers

The UI has an optional **available layers** field. Declaring your layers in
`name (geometry)` form — e.g. `streams (line), parcels (polygon)` — makes the
model use only real inputs instead of guessing, and enables stricter validation
(undeclared layers and geometry/measure mismatches are flagged). Leaving it blank
lets the model infer inputs from the query.

---

## Project status & roadmap

- **v1 / v2 (current):** working end-to-end. Diagram + ArcPy output, optional
  layer declaration, validation layer, query logging.
- **v3 (in progress):** read available layers directly from a **file geodatabase**
  (modern Esri `.gdb` folder) via `gdb_reader.py`, so the layer inventory and real
  attribute fields come from actual data instead of being typed. See
  `gdb_reader.py` and `CONTRIBUTING.md`.
- **Later:** optional fine-tuning using collected query logs.

---

## Repository layout

```
app.py                 FastAPI app: inference + validation + web UI + ArcPy panel
system_prompt.txt      Instructions + output schema + few-shot examples
gis_ontology.json      Vector tool definitions, data types, keyword vocabulary
gdb_reader.py          v3: read a file geodatabase into the layer inventory
batch_test.py          Fire a query set at the API; pass/warn/fail summary
requirements.txt       Python dependencies
Dockerfile             Builds the app image
docker-compose.yml     Local two-container setup (app + Ollama)
.env.example           Copy to .env and configure
CONTRIBUTING.md        Dev setup, architecture, how to extend
LICENSE                MIT
```

---

## A note on security

This app has **no authentication** and is intended to run **locally**
(`127.0.0.1`). Do not expose it to a public network as-is — anyone who can reach
it can use your LLM/compute. Keep your `API_KEY` in `.env` (gitignored); never
commit it.

---

## License

MIT — see [LICENSE](LICENSE).
