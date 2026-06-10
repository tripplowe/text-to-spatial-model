# Contributing / Developer Guide

This guide is for developers (including student interns) continuing work on
text-to-model. Read the [README](README.md) first for what the project does.

---

## Development setup

```bash
git clone https://github.com/tripplowe/text-to-model.git
cd text-to-model
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
cp .env.example .env                                # configure your LLM
python app.py                                       # http://localhost:8080
```

You need a working LLM backend (local Ollama or a hosted API key) — see the
README. For fast iteration, a small local model (`llama3.1:8b-instruct-q8_0`) is
fine even though its output quality is lower; switch to a capable model to judge
real behavior.

### Running the test set

`batch_test.py` fires a set of queries at the running app and reports
pass/warn/fail per query:

```bash
python batch_test.py --url http://localhost:8080
```

"Pass" means well-formed + no validation warnings; it does **not** guarantee the
workflow is correct for the query — review those yourself.

---

## Architecture

```
app.py            FastAPI server. Assembles the prompt, calls the LLM backend,
                  parses + validates JSON, serves the inline single-page UI.
system_prompt.txt Instructions, the JSON output schema, behavioral rules, and the
                  few-shot examples. MOST behavior is shaped here.
gis_ontology.json The tool "textbook": tool definitions, data types, parameters,
                  ArcPy syntax, and the keyword vocabulary that triggers each.
                  Injected into the system prompt at a placeholder.
gdb_reader.py     (v3) Reads a file geodatabase into the layer inventory.
batch_test.py     Evaluation harness.
```

**Request flow:** browser → `app.py` assembles `system_prompt.txt` +
`gis_ontology.json` + few-shot examples (+ optional layer context) → LLM backend →
JSON response → `validate_workflow()` → UI renders keyword table, diagram, ArcPy
script, notes, and warnings. Each interaction is logged to `logs/queries.jsonl`.

**The validation layer (`validate_workflow` in `app.py`)** is a hand-written
geospatial sanity check — it is where you encode rules the LLM tends to get wrong
(out-of-scope tools, hallucinated/aliased tool names with corrective hints,
incomplete graphs, self-clips, group-by without a case field, geometry/measure
mismatches when a layer list is declared). When the model makes a new class of
mistake, the fix usually lives here and/or in a new few-shot example.

---

## Lessons learned (save yourself the pain)

These are hard-won from earlier development. Heed them.

- **Model capacity dominates.** Small models (8B–14B) persistently truncated
  workflows and invented tool names, and this could **not** be fixed by prompt
  edits. A ~30B-class model followed the same prompt cleanly. If outputs are
  structurally broken, try a bigger model before rewriting the prompt.
- **Ollama context window.** The system prompt is ~17K tokens; many models
  default to a 4096 context and reject every request with a "context size" error.
  Set `NUM_CTX` (≥ ~20000). This is **not** a memory error.
- **Ollama "thinking" model tags break JSON.** Some tags (e.g. a bare
  `qwen3:30b-a3b`) resolve to a *thinking* variant that emits reasoning text and
  breaks JSON parsing. Use the explicit **instruct** tag.
- **The web UI is inline JavaScript inside a Python triple-quoted string in
  `app.py`.** A `\n` written in that JS becomes a *real newline* when Python builds
  the page, which breaks the script (and a `\n` inside a JS comment can turn the
  rest of the line into executable code). Avoid `\n` literals in the embedded JS —
  use `String.fromCharCode(10)`. **After any frontend edit, verify the *served*
  JavaScript actually parses/runs**, not just that Python compiles. (A quick way:
  render the HTML the app serves, extract the `<script>` contents, and run it
  through `node --check`.)
- **Docker: rebuild after editing baked-in files.** `app.py`,
  `system_prompt.txt`, and `gis_ontology.json` are copied into the image at build
  time. Editing them requires `docker compose up -d --build app`; a plain restart
  won't pick up changes. If `COPY app.py` shows `CACHED` but the container still
  runs old code, force it: `docker compose build --no-cache app`.

---

## How to extend

### Add or change a tool's behavior
1. Update `gis_ontology.json` (definition, parameters, ArcPy syntax, trigger
   keywords). Use the **current ArcGIS Pro tool name** (e.g. *Calculate Geometry
   Attributes*, *Select Layer By Attribute*).
2. If the model needs to *see* the pattern, add or adjust a **few-shot example**
   in `system_prompt.txt`. Examples teach far more reliably than prose rules.
3. If there's a logical constraint, add a check in `validate_workflow()`.
4. Re-run `batch_test.py` and confirm the embedded examples still validate clean.

### Add a validation rule
Add it to `validate_workflow()` in `app.py`. Prefer warnings with a **corrective
hint** ("did you mean X?") over bare rejections — they're more useful and double
as teaching output.

### Keep JSON valid
The model must return only JSON. `parse_json_response()` tolerates code fences and
minor issues, but new prompt changes should preserve "return ONLY valid JSON, no
preamble," and any new backend must request JSON mode where supported.

---

## Version 3: file geodatabase integration (current priority)

**Goal:** replace the hand-typed "available layers" list with layers read directly
from a **file geodatabase** (the modern Esri `.gdb` *folder* — NOT the obsolete
`.mdb` personal geodatabase). The downstream consumer (prompt injection +
validation) already accepts a layer inventory; v3 only changes the *source*.

**Decisions already made:**
- Read from a `.gdb` on disk (file upload deferred).
- Library: **pyogrio** (GDAL's built-in OpenFileGDB driver — no Esri SDK / ArcGIS
  license). Uncomment `pyogrio` in `requirements.txt`.
- Extract layer **name + geometry type + attribute fields** (vector layers only).
- Inject all discovered vector layers; let the user narrow.

**What exists:** `gdb_reader.py` (drafted) reads a `.gdb` → layer inventory +
diagnostic report, and has a CLI self-test:

```bash
pip install pyogrio
python gdb_reader.py path/to/your.gdb
```

**Roadmap:**
1. Verify `gdb_reader.py` against a real `.gdb`; confirm `pyogrio.read_info`'s
   return shape matches what the reader expects (adjust if your GDAL version
   differs).
2. Add `pyogrio` to `requirements.txt`/`Dockerfile`; mount a data dir in compose.
3. Wire the reader into `app.py` (endpoint → `inventory_to_context_string()` →
   existing layer-context path; pass parsed layers + fields to
   `validate_workflow()`).
4. Surface discovered layers in the UI (with selection).
5. **Field-aware prompting:** feed real attribute field names so the model stops
   guessing case-field names for group-by / attribute selection. Biggest accuracy
   win the geodatabase unlocks.

---

## Pull request checklist

- [ ] `app.py` compiles (`python -m py_compile app.py`).
- [ ] If you touched the frontend JS, the **served** JS parses/runs (no `\n`
      literals introduced).
- [ ] If you touched the prompt/ontology, the embedded few-shot examples still
      validate clean and `batch_test.py` was run.
- [ ] No secrets committed (`.env`, API keys). `.env.example` updated if you added
      a config variable.
- [ ] README/this guide updated if behavior or setup changed.
