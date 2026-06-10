"""
Text-to-Spatial-Model (text-to-model)
======================================
Turns a plain-English GIS question into a structured ArcGIS Pro vector-overlay
workflow: a node/edge diagram, the ArcPy code, keyword analysis, and teaching
notes. Phase 1 uses prompt engineering with an LLM (no fine-tuning).

This script:
  1. Loads the GIS ontology and system prompt
  2. Sends natural-language requests to an LLM backend (local Ollama, local
     vLLM/SGLang, or any hosted OpenAI-compatible API with your own key)
  3. Parses and validates the structured JSON response
  4. Serves a web interface

Quick start (local Ollama):
  1. cp .env.example .env       # then edit if needed
  2. ollama pull <model>        # e.g. qwen3:30b-a3b-instruct-2507-q4_K_M
  3. pip install -r requirements.txt
  4. python app.py              # open http://localhost:8080

Quick start (hosted API):
  Set LLM_BACKEND=openai, OPENAI_BASE_URL, API_KEY, MODEL_NAME in .env, then
  run steps 3-4 above. See README.md and .env.example for details.
"""

import json
import os
import re
import httpx
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load a local .env file if present (optional dependency). This lets students
# keep their config/keys in .env instead of exporting env vars by hand.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on real environment variables


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Which LLM backend to use. Options:
#   "ollama"  → Ollama API (local or remote)        — no API key
#   "vllm"    → vLLM OpenAI-compatible API (local)   — no API key
#   "sglang"  → SGLang OpenAI-compatible API (local) — no API key
#   "openai"  → ANY OpenAI-compatible HOSTED API     — uses API_KEY + OPENAI_BASE_URL
#               (e.g. OpenAI, Together, Groq, OpenRouter, a hosted vLLM, etc.)
# Pick the one matching how YOU are running a model. Students without a GPU box
# typically use "ollama" locally, or "openai" with their own hosted API key.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")

# Model name / id (adjust to what you've pulled locally or what your API offers).
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3.1:8b-instruct-q8_0")

# Ollama host. Defaults to a local Ollama. In Docker Compose this is overridden
# to the internal service name (http://ollama:11434).
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Hosted OpenAI-compatible API settings (only used when LLM_BACKEND="openai").
#   OPENAI_BASE_URL : the API root, e.g. https://api.openai.com/v1
#                     (or any OpenAI-compatible provider's base URL)
#   API_KEY         : your secret key. NEVER commit this — set it in .env / env.
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("API_KEY", "")

BACKEND_URLS = {
    "ollama": f"{OLLAMA_HOST}/api/chat",
    "vllm": "http://localhost:8000/v1/chat/completions",
    "sglang": "http://localhost:30000/v1/chat/completions",
    "openai": f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions",
}

# Backends that use the OpenAI-compatible request/response shape.
OPENAI_COMPATIBLE = {"vllm", "sglang", "openai"}

# Temperature: lower = more deterministic (important for structured output)
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.1"))

# Maximum tokens for the response.
# Detailed vector workflows with full keyword reasoning can exceed 4096 and get
# truncated (producing invalid JSON the parser cannot recover). IRENE has ample
# VRAM headroom on the 8B model, so default higher.
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "6144"))

# Context window (prompt + response). The assembled system prompt is ~16K
# tokens, so this must comfortably exceed that. Some models (e.g. qwen3:30b)
# default to only 4096 and will reject the request otherwise. 24576 leaves
# room for the prompt (~16K) plus a full response (~6K).
NUM_CTX = int(os.environ.get("NUM_CTX", "24576"))

# Query logging — captures every request/response pair as JSON Lines for Phase 2
# fine-tuning. Each line is one interaction; correct ones are kept as-is and wrong
# ones can be corrected later to build the training set. Disable with
# QUERY_LOG_ENABLED=0.
QUERY_LOG_ENABLED = os.environ.get("QUERY_LOG_ENABLED", "1") not in ("0", "false", "False")
QUERY_LOG_PATH = Path(os.environ.get("QUERY_LOG_PATH", "logs/queries.jsonl"))


# ═══════════════════════════════════════════════════════════════════════════
# LOAD PROMPT COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

def load_system_prompt() -> str:
    """
    Assembles the full system prompt from:
      1. The instruction template (system_prompt.txt)
      2. The GIS ontology (gis_ontology.json) — inserted at the placeholder

    This is where the ontology and few-shot examples come together.
    The ontology is NOT the same as the few-shot examples:
      - The ONTOLOGY defines WHAT tools exist and their specifications.
      - The FEW-SHOT EXAMPLES show HOW to use those tools in response to
        natural language — they demonstrate the expected reasoning and format.
    """
    base_dir = Path(__file__).parent

    # Load the system prompt template
    prompt_path = base_dir / "system_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"System prompt not found at {prompt_path}. "
            "Make sure system_prompt.txt is in the same directory as app.py."
        )
    prompt_template = prompt_path.read_text(encoding="utf-8")

    # Load the GIS ontology
    ontology_path = base_dir / "gis_ontology.json"
    if not ontology_path.exists():
        raise FileNotFoundError(
            f"GIS ontology not found at {ontology_path}. "
            "Make sure gis_ontology.json is in the same directory as app.py."
        )
    ontology = ontology_path.read_text(encoding="utf-8")

    # Insert ontology into the system prompt at the placeholder
    full_prompt = prompt_template.replace(
        "<<INSERT CONTENTS OF gis_ontology.json HERE>>",
        ontology
    )

    return full_prompt


# Load once at startup
SYSTEM_PROMPT = load_system_prompt()
print(f"System prompt loaded: {len(SYSTEM_PROMPT):,} characters")
print(f"  Approximate tokens: ~{len(SYSTEM_PROMPT) // 4:,}")


# ═══════════════════════════════════════════════════════════════════════════
# AVAILABLE-LAYERS PARSING (user-declared input inventory)
# ═══════════════════════════════════════════════════════════════════════════

# Recognized geometry tags (normalized to the ontology's vector data types).
_GEOM_MAP = {
    "point": "vector-point", "points": "vector-point",
    "line": "vector-line", "lines": "vector-line", "polyline": "vector-line",
    "polygon": "vector-polygon", "polygons": "vector-polygon", "poly": "vector-polygon",
    "table": "table",
}

# Matches "name (geometry)" entries, separated by commas or newlines.
_LAYER_RE = re.compile(r"^\s*(.+?)\s*\(\s*([A-Za-z]+)\s*\)\s*$")


def parse_layers(raw: str):
    """
    Parse a user-declared layer list in 'name (geometry)' syntax.

    Returns (layers, ok):
      layers = [{"name": str, "geometry": "vector-polygon"|..., "raw": str}, ...]
      ok     = True  if at least one well-formed, geometry-tagged entry was found
               False if the field is empty OR not in the required tagged format
                     (caller then reverts to Case 1: infer inputs, note insufficient info)
    """
    if not raw or not raw.strip():
        return [], False
    # split on newlines and commas (commas inside parens are not expected)
    parts = re.split(r"[\n,]+", raw.strip())
    layers = []
    for part in parts:
        if not part.strip():
            continue
        m = _LAYER_RE.match(part)
        if not m:
            continue  # untagged / malformed entry — ignored
        name = m.group(1).strip()
        geom = _GEOM_MAP.get(m.group(2).strip().lower())
        if not name or not geom:
            continue
        layers.append({"name": name, "geometry": geom, "raw": part.strip()})
    return layers, (len(layers) > 0)


def build_layer_context(raw: str):
    """
    Build the instruction block appended to the user message, plus the parsed
    layers (for the validator). Implements the three policies:
      - well-formed list  -> authoritative; do not invent layers
      - malformed/untagged -> Case 1 (infer + insufficient-info note)
      - empty/omitted      -> Case 1 (infer + assumed-inputs note)
    Returns (context_text, layers, list_ok).
    """
    layers, ok = parse_layers(raw)
    if ok:
        inv = "; ".join(f"{l['name']} ({l['geometry']})" for l in layers)
        ctx = (
            "\n\n--- AVAILABLE LAYERS (authoritative) ---\n"
            f"The ONLY data layers available are: {inv}.\n"
            "Use ONLY these as input nodes. Do NOT invent or assume any layer that "
            "is not listed. If the request references or implies a layer that is "
            "not in this list (e.g. a property/boundary that was not declared), do "
            "NOT fabricate it: add a clear entry to notes.ambiguities naming the "
            "missing layer, and produce the best workflow possible WITHOUT it. "
            "Match each input's geometry to the list: do not request AREA from a "
            "line layer or LENGTH from a polygon layer, and a Clip/Erase boundary "
            "must be a polygon layer from this list."
        )
        return ctx, layers, True
    else:
        note = (
            "\n\n--- AVAILABLE LAYERS ---\n"
            "No valid layer list was provided (expected 'name (geometry)' format, "
            "e.g. 'streams (line), property (polygon)'). Infer the likely input "
            "layers from the request, and add a note to notes.assumptions that "
            "inputs were inferred because insufficient layer information was given. "
            "Mark each inferred input clearly (e.g. 'ASSUMED INPUT: streams (line) — "
            "not declared')."
        )
        return note, layers, False


# ═══════════════════════════════════════════════════════════════════════════
# LLM INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

async def call_llm(user_message: str, layer_context: str = "") -> dict:
    """
    Send a user message to the local LLM and parse the JSON response.

    The full prompt sent to the model looks like:

        [System message]
        - Role definition and instructions
        - Complete GIS ontology (tools, data types, keywords)
        - Domain conversion rules
        - Three few-shot examples showing input → output format
        - "Now process the following user request. Return ONLY the JSON object."

        [User message]
        "Identify the forested areas within a 200 foot buffer of a stream"

    The model returns a JSON object with keywords, workflow, and notes.
    """
    url = BACKEND_URLS.get(LLM_BACKEND)
    if not url:
        raise ValueError(f"Unknown LLM backend: {LLM_BACKEND}")

    # The layer context (if any) is appended to the USER message rather than the
    # system prompt, so the large system prompt stays identical/cacheable.
    user_content = user_message + layer_context

    # Build the request based on backend type
    headers = {}
    if LLM_BACKEND == "ollama":
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS,
                # Context window must hold the large system prompt (~16K tokens)
                # plus the response. Some models default to only 4096, which
                # rejects the request outright, so set it explicitly.
                "num_ctx": NUM_CTX,
            },
            # Request JSON mode if supported by the model
            "format": "json",
        }
    else:
        # OpenAI-compatible API (vLLM, SGLang, or any hosted "openai" provider)
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }
        # Hosted APIs require a bearer token; local vLLM/SGLang usually don't.
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"

    # Hosted APIs (and big local models) can be slow; allow a generous timeout.
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to LLM backend at {url}. "
                f"Is the '{LLM_BACKEND}' backend running/reachable? "
                f"For local Ollama: run 'ollama serve'. "
                f"For a hosted API: check OPENAI_BASE_URL and your network."
            )
        except httpx.HTTPStatusError as e:
            hint = ""
            if e.response.status_code in (401, 403):
                hint = " — check your API_KEY (authentication failed)."
            raise HTTPException(
                status_code=502,
                detail=f"LLM backend error: {e.response.status_code} — "
                       f"{e.response.text[:400]}{hint}"
            )

    # Extract the response text
    data = response.json()
    if LLM_BACKEND == "ollama":
        raw_text = data.get("message", {}).get("content", "")
    else:
        raw_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Parse JSON from the response
    result = parse_json_response(raw_text)
    return result


def parse_json_response(text: str) -> dict:
    """
    Extract and parse JSON from the LLM response.
    Handles common issues: markdown fences, preamble text, trailing commas.
    """
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text.strip())

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object within the text
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        json_str = text[brace_start : brace_end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # If all parsing fails, return error structure
    return {
        "error": "Failed to parse LLM response as JSON",
        "raw_response": text[:1000],
        "keywords": [],
        "workflow": {"nodes": [], "edges": []},
        "notes": {
            "ambiguities": ["LLM response was not valid JSON — check model and prompt"],
            "assumptions": [],
            "alternatives": [],
            "teaching_points": [],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# WORKFLOW VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_workflow(result: dict, layers: list = None, list_ok: bool = False) -> list[str]:
    """
    Validates the LLM-generated workflow against the GIS ontology rules.
    Returns a list of validation warnings/errors.

    This is the GEOSPATIAL LOGIC LAYER — it catches mistakes the LLM makes.

    Optional args:
      layers  : parsed authoritative layer list [{"name","geometry","raw"}, ...]
      list_ok : True if a well-formed declared list was provided (Case 2/3 checks
                only run when the inputs are authoritative).
    """
    warnings = []
    layers = layers or []
    workflow = result.get("workflow", {})
    nodes = {n["id"]: n for n in workflow.get("nodes", [])}
    edges = workflow.get("edges", [])

    # Check: all edge references exist
    for edge in edges:
        if edge["from"] not in nodes:
            warnings.append(f"Edge references non-existent node: {edge['from']}")
        if edge["to"] not in nodes:
            warnings.append(f"Edge references non-existent node: {edge['to']}")

    # Check: every tool node has at least one input
    tool_nodes = [n for n in nodes.values() if n["type"] == "tool"]
    for tool in tool_nodes:
        incoming = [e for e in edges if e["to"] == tool["id"]]
        if not incoming:
            warnings.append(
                f"Tool '{tool['label']}' has no input connections — "
                "every tool needs at least one input"
            )

    # Check: every tool node has at least one output
    for tool in tool_nodes:
        outgoing = [e for e in edges if e["from"] == tool["id"]]
        if not outgoing:
            warnings.append(
                f"Tool '{tool['label']}' has no output connection — "
                "tool results must feed into derived data or final output"
            )

    # Check: input nodes should not have incoming edges
    input_nodes = [n for n in nodes.values() if n["type"] == "input"]
    for inp in input_nodes:
        incoming = [e for e in edges if e["to"] == inp["id"]]
        if incoming:
            warnings.append(
                f"Input data '{inp['label']}' has incoming edges — "
                "input nodes should only have outgoing connections"
            )

    # Check: output node exists
    output_nodes = [n for n in nodes.values() if n["type"] == "output"]
    if not output_nodes:
        warnings.append("No output node found — workflow must produce a final result")

    # ── VECTOR-ONLY SCOPE CHECKS ──────────────────────────────────────────
    # This phase is vector overlay only. The ontology contains no raster tools
    # or types, so any raster tool/type in the output is a hallucination.
    VECTOR_TYPES = {"vector-point", "vector-line", "vector-polygon"}
    ALLOWED_TYPES = VECTOR_TYPES | {"table", None, ""}
    ALLOWED_TOOLS = {
        "Buffer", "Clip", "Intersect", "Erase", "Union", "Dissolve",
        "Select Layer By Attribute", "Select Layer By Location",
        "Summary Statistics", "Calculate Geometry Attributes",
    }
    # Common out-of-scope tools the model may hallucinate from prior training.
    RASTER_TOOLS = {
        "Slope", "Aspect", "Hillshade", "Viewshed2", "Viewshed", "Reclassify",
        "Weighted Overlay", "Extract by Mask", "Raster Calculator",
        "Zonal Statistics as Table", "Zonal Statistics", "Contour",
        "Feature to Raster", "Raster to Polygon",
    }
    # Known hallucinated / informal tool names mapped to the canonical ontology
    # tool, with a short corrective hint. Keys are compared case-insensitively.
    TOOL_ALIASES = {
        "group by": ("Summary Statistics",
                     "grouping by a category is the 'case_field' PARAMETER of "
                     "Summary Statistics, not a separate tool"),
        "groupby": ("Summary Statistics",
                    "grouping is the 'case_field' parameter of Summary Statistics"),
        "summarize": ("Summary Statistics", "use the canonical name 'Summary Statistics'"),
        "summary": ("Summary Statistics", "use the canonical name 'Summary Statistics'"),
        "statistics": ("Summary Statistics", "use the canonical name 'Summary Statistics'"),
        "summarystatistics": ("Summary Statistics", "use the display name 'Summary Statistics', not the ArcPy form"),
        "summarize attributes": ("Summary Statistics", "in this course use 'Summary Statistics'"),
        "tabulate area": ("Summary Statistics",
                          "Tabulate Area is a RASTER tool; for vector acreage per "
                          "category use Calculate Geometry Attributes + Summary Statistics"),
        "calculate geometry": ("Calculate Geometry Attributes",
                               "the full tool name is 'Calculate Geometry Attributes'"),
        "calculategeometry": ("Calculate Geometry Attributes",
                              "use the display name 'Calculate Geometry Attributes'"),
        "calculategeometryattributes": ("Calculate Geometry Attributes",
                                        "use the display name (with spaces)"),
        "update geometry": ("Calculate Geometry Attributes",
                            "'Update Geometry' is the course nickname; the tool_name "
                            "must be 'Calculate Geometry Attributes'"),
        "add geometry attributes": ("Calculate Geometry Attributes",
                                    "Add Geometry Attributes is DEPRECATED; use "
                                    "'Calculate Geometry Attributes'"),
        "add xy coordinates": ("Calculate Geometry Attributes",
                               "Add XY Coordinates lacks coordinate-format control; use "
                               "'Calculate Geometry Attributes' with POINT_X/POINT_Y"),
        "calculate field": ("Calculate Geometry Attributes",
                            "for area/length/coordinates use 'Calculate Geometry Attributes'"),
        "calculatefield": ("Calculate Geometry Attributes",
                           "for geometry values use 'Calculate Geometry Attributes', "
                           "not the generic Calculate Field"),
        "feature to point": ("Calculate Geometry Attributes",
                             "to read point coordinates do NOT convert geometry; use "
                             "'Calculate Geometry Attributes' with POINT_X/POINT_Y"),
        # Interactive selection phrasings -> the geoprocessing tool form
        "select by attribute": ("Select Layer By Attribute",
                                "'Select by Attribute' is the interactive selection; the "
                                "workflow TOOL is 'Select Layer By Attribute'"),
        "select by location": ("Select Layer By Location",
                               "'Select by Location' is the interactive selection; the "
                               "workflow TOOL is 'Select Layer By Location'"),
        "selectlayerbyattribute": ("Select Layer By Attribute",
                                   "use the display name with spaces"),
        "selectlayerbylocation": ("Select Layer By Location",
                                  "use the display name with spaces"),
        "select": ("Select Layer By Attribute",
                   "specify 'Select Layer By Attribute' or 'Select Layer By Location'"),
        "spatial join": ("Intersect",
                         "for vector overlay in this phase use Intersect (or Clip)"),
        "buffer analysis": ("Buffer", "the tool name is simply 'Buffer'"),
        # Pairwise overlay variants are valid ArcGIS tools; accept but normalize
        "pairwise buffer": ("Buffer", "Pairwise Buffer is an equivalent; this course uses 'Buffer'"),
        "pairwise clip": ("Clip", "Pairwise Clip is an equivalent; this course uses 'Clip'"),
        "pairwise dissolve": ("Dissolve", "Pairwise Dissolve is an equivalent; this course uses 'Dissolve'"),
        "pairwise erase": ("Erase", "Pairwise Erase is an equivalent; this course uses 'Erase'"),
        "pairwise intersect": ("Intersect", "Pairwise Intersect differs in output; this course uses 'Intersect'"),
    }

    # Check: no raster data types appear (scope violation / hallucination)
    for n in nodes.values():
        dt = n.get("data_type")
        if dt not in ALLOWED_TYPES:
            warnings.append(
                f"SCOPE: node '{n.get('label', n['id'])}' has data_type '{dt}', "
                "which is out of scope for this vector-only phase "
                "(allowed: vector-point, vector-line, vector-polygon, table)."
            )

    # Check: every tool is in the active vector ontology
    for tool in tool_nodes:
        tname = tool.get("tool_name")
        if tname in ALLOWED_TOOLS:
            continue
        if tname in RASTER_TOOLS:
            warnings.append(
                f"SCOPE: tool '{tname}' is a raster/terrain tool and is NOT in "
                "the vector ontology for this phase. Re-model using vector tools "
                "(e.g., Clip/Intersect for containment), or flag the request as "
                "out of scope."
            )
        elif tname and tname.strip().lower() in TOOL_ALIASES:
            canonical, hint = TOOL_ALIASES[tname.strip().lower()]
            warnings.append(
                f"TOOL NAME: '{tname}' is not a tool — did you mean "
                f"'{canonical}'? ({hint}.)"
            )
        else:
            warnings.append(
                f"UNKNOWN TOOL: '{tname}' is not in the active ontology. "
                f"Allowed tools: {', '.join(sorted(ALLOWED_TOOLS))}."
            )

    # Helper: data types of nodes feeding into a tool
    def _incoming_types(tool_id):
        return [
            (nodes[e["from"]].get("label", e["from"]), nodes[e["from"]].get("data_type"))
            for e in edges
            if e["to"] == tool_id and e["from"] in nodes
        ]

    for tool in tool_nodes:
        tname = tool.get("tool_name")
        params = tool.get("parameters", {}) or {}

        # Clip / Erase: need exactly two inputs; the clip/erase feature must be polygon.
        # This is the CLIP-DIRECTION sanity check — a polygon boundary is required.
        if tname in ("Clip", "Erase"):
            # Self-clip detection: the model sometimes inserts a Clip with the same
            # layer as both input and boundary (a meaningless no-op) to satisfy a
            # learned pattern. Flag it — the correct fix is to remove the Clip.
            p = tool.get("parameters", {}) or {}
            inf = str(p.get("in_features", "")).strip().lower()
            clf = str(p.get("clip_features", p.get("erase_features", ""))).strip().lower()
            if inf and clf and inf == clf:
                warnings.append(
                    f"SELF-CLIP: {tname} '{tool['label']}' uses the same layer "
                    f"('{p.get('in_features')}') as both the input and the "
                    "clip/erase feature — a meaningless no-op. There is no distinct "
                    "boundary to clip against, so this tool should be REMOVED from "
                    "the workflow entirely."
                )
            incoming = _incoming_types(tool["id"])
            # A self-clip often appears as a single incoming edge feeding the tool
            # twice; treat 1-input Clip as the same error class.
            if len(incoming) != 2:
                warnings.append(
                    f"{tname} '{tool['label']}' has {len(incoming)} input(s); it needs "
                    "exactly two distinct layers: the layer being trimmed and the "
                    f"polygon boundary. If there is no separate boundary layer, remove "
                    f"the {tname} step rather than clipping a layer by itself."
                )
            else:
                # At least one incoming feature must be a polygon to act as the boundary.
                if not any(dt == "vector-polygon" for _, dt in incoming):
                    warnings.append(
                        f"CLIP DIRECTION: {tname} '{tool['label']}' has no polygon "
                        "boundary among its inputs. The clip/erase feature (the "
                        "'cookie cutter') must be a vector-polygon."
                    )

        # Summary Statistics: the "each"/group-by case must have a case field.
        # Also catch the case where the model used an alias name ("Summarize",
        # "Group By") that resolves to Summary Statistics.
        _alias_canon = TOOL_ALIASES.get((tname or "").strip().lower(), (None,))[0]
        if tname == "Summary Statistics" or _alias_canon == "Summary Statistics":
            has_case = bool(params.get("case_field") or params.get("group_field"))
            if not has_case:
                warnings.append(
                    "GROUP-BY: Summary Statistics has no 'case_field'. If the request "
                    "asked for a value 'per' or 'for each' category, a case_field is "
                    "required to produce one row per category (otherwise it returns a "
                    "single grand total)."
                )

            # If it sums an area/length measure, a Calculate Geometry Attributes
            # step must exist somewhere upstream to populate that field first.
            stat = " ".join(str(params.get(k, "")) for k in (
                "statistics_fields", "summarize_field", "field", "fields",
                "sum_field", "value_field")).lower()
            measures = ("acre", "area", "hect", "sq", "square", "length",
                        "mile", "feet", "foot", "meter", "perimeter",
                        "my_acres", "my_hecta", "my_miles", "my_feet")
            if any(tok in stat for tok in measures):
                # walk all ancestors looking for Calculate Geometry Attributes
                ancestors, frontier, seen = set(), [tool["id"]], set()
                while frontier:
                    cur = frontier.pop()
                    if cur in seen:
                        continue
                    seen.add(cur)
                    for e in edges:
                        if e["to"] == cur and e["from"] in nodes:
                            ancestors.add(e["from"])
                            frontier.append(e["from"])
                has_calc = any(
                    nodes[a].get("tool_name") == "Calculate Geometry Attributes"
                    for a in ancestors
                )
                if not has_calc:
                    shown = (params.get("statistics_fields")
                             or params.get("summarize_field")
                             or params.get("field") or stat.strip())
                    warnings.append(
                        "MISSING STEP: Summary Statistics sums an area/length field "
                        f"('{shown}') but no Calculate "
                        "Geometry Attributes step appears upstream. Area/length must "
                        "be computed into a field (e.g. my_acres) BEFORE it can be "
                        "summed per category."
                    )

        # Calculate Geometry Attributes ("Update Geometry"): warn if it precedes a
        # Clip/Intersect downstream, since area must be (re)calculated AFTER extents change.
        if tname == "Calculate Geometry Attributes":
            downstream = [e["to"] for e in edges if e["from"] == tool["id"]]
            # one hop downstream
            for d in downstream:
                dn = nodes.get(d, {})
                # if a derived node leads into a Clip/Intersect tool, geometry was calc'd too early
                nxt = [nodes.get(e["to"], {}) for e in edges if e["from"] == d]
                if any(x.get("tool_name") in ("Clip", "Intersect") for x in nxt):
                    warnings.append(
                        f"ORDER: '{tool['label']}' (Update Geometry) appears BEFORE a "
                        "Clip/Intersect. Area must be recalculated AFTER any tool that "
                        "changes feature extents, or acreage will be wrong."
                    )

    # ── CASE 2 & 3: declared-layer checks (only when an authoritative list given) ──
    if list_ok and layers:
        declared = {l["name"].strip().lower(): l["geometry"] for l in layers}

        # Build a lookup of input-node label -> declared geometry (best effort).
        def _declared_geom(label):
            if not label:
                return None
            return declared.get(label.strip().lower())

        # CASE 2: an input node names a layer not in the declared list.
        input_nodes = [n for n in nodes.values() if n.get("type") == "input"]
        for inp in input_nodes:
            label = (inp.get("label") or "").strip()
            if label and label.lower() not in declared:
                # Only flag if the model didn't already record it as an ambiguity.
                amb = " ".join(result.get("notes", {}).get("ambiguities", []) or []).lower()
                already = label.lower() in amb
                msg = (
                    f"UNDECLARED LAYER: input '{label}' is not in the available "
                    f"layers ({', '.join(sorted(declared))}). Do not fabricate it — "
                    "the workflow should flag it in notes.ambiguities and proceed "
                    "without it."
                )
                if not already:
                    warnings.append(msg)

        # CASE 3: geometry vs. requested measure mismatch.
        # Detect what is being measured (AREA vs LENGTH) on which input layer.
        for tool in tool_nodes:
            if tool.get("tool_name") != "Calculate Geometry Attributes":
                continue
            gp = str((tool.get("parameters", {}) or {}).get("geometry_property", "")).lower()
            wants_area = any(t in gp for t in ("area", "acre", "hect", "square", "sq "))
            wants_length = any(t in gp for t in ("length", "perimeter", "mile", "feet", "foot", "meter", "km"))
            # trace back to the originating input layer's geometry
            anc, frontier, seen = [], [tool["id"]], set()
            while frontier:
                cur = frontier.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                for e in edges:
                    if e["to"] == cur and e["from"] in nodes:
                        frontier.append(e["from"])
                        if nodes[e["from"]].get("type") == "input":
                            anc.append(nodes[e["from"]])
            for src in anc:
                geom = _declared_geom(src.get("label")) or src.get("data_type")
                if wants_area and geom == "vector-line":
                    warnings.append(
                        f"GEOMETRY MISMATCH: AREA/acres requested, but the measured "
                        f"layer '{src.get('label')}' is a LINE (length, not area). "
                        "Lines have length, not area — either measure LENGTH, or the "
                        "wrong layer is named for an area calculation."
                    )
                if wants_length and geom == "vector-polygon" and "perimeter" not in gp:
                    warnings.append(
                        f"GEOMETRY MISMATCH: LENGTH requested, but the measured layer "
                        f"'{src.get('label')}' is a POLYGON. Use AREA for polygons (or "
                        "PERIMETER if the boundary length is intended)."
                    )

        # CASE 3b: a Clip/Erase boundary must be a declared polygon layer.
        for tool in tool_nodes:
            if tool.get("tool_name") not in ("Clip", "Erase"):
                continue
            for e in edges:
                if e["to"] == tool["id"] and e["from"] in nodes:
                    src = nodes[e["from"]]
                    if src.get("type") == "input":
                        geom = _declared_geom(src.get("label"))
                        # only assert when we actually know the declared geometry
                        if geom and geom != "vector-polygon":
                            # could be the input-being-trimmed (legal) — only warn if
                            # NO declared polygon feeds this tool at all
                            feeders = [nodes[x["from"]] for x in edges
                                       if x["to"] == tool["id"] and x["from"] in nodes]
                            if not any(_declared_geom(f.get("label")) == "vector-polygon"
                                       or f.get("data_type") == "vector-polygon"
                                       for f in feeders):
                                warnings.append(
                                    f"GEOMETRY MISMATCH: {tool.get('tool_name')} needs a "
                                    "POLYGON boundary, but no polygon layer feeds it."
                                )
                                break

    return warnings

# ═══════════════════════════════════════════════════════════════════════════
# QUERY LOGGING (Phase 2 training-data collection)
# ═══════════════════════════════════════════════════════════════════════════

def log_interaction(query: str, result: dict, warnings: list[str]) -> None:
    """
    Append one request/response interaction to the JSON Lines query log.

    Each line is a self-contained JSON object. For Phase 2 fine-tuning, keep the
    correct interactions as-is and hand-correct the wrong ones; the resulting
    (query → corrected JSON) pairs become the training set. The 'reviewed' and
    'correct' fields are placeholders for that later triage step.

    Logging never raises into the request path — a logging failure must not break
    a student's query.
    """
    if not QUERY_LOG_ENABLED:
        return
    try:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_NAME,
            "backend": LLM_BACKEND,
            "temperature": TEMPERATURE,
            "query": query,
            "result": result,
            "validation_warnings": warnings,
            # placeholders for manual Phase-2 triage:
            "reviewed": False,
            "correct": None,
            "corrected_result": None,
        }
        QUERY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with QUERY_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        # Log to stderr but never propagate
        print(f"[query-log] WARNING: failed to write log entry: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Text-to-Spatial-Model",
    description="Translate natural language to GIS workflow diagrams",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/generate")
async def generate_model(request: dict):
    """
    Main API endpoint.
    Accepts: {"query": "...", "layers": "streams (line), property (polygon)"}
      - query  (required): natural language spatial analysis request
      - layers (optional): user-declared input inventory in 'name (geometry)'
                           syntax. If omitted or malformed, inputs are inferred.
    Returns: The LLM-generated workflow with keywords, workflow, and notes,
             plus any validation warnings.
    """
    query = request.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    raw_layers = request.get("layers", "") or ""
    layer_context, layers, list_ok = build_layer_context(raw_layers)

    # Call the LLM with the layer context appended to the user message
    result = await call_llm(query, layer_context)

    # Validate the workflow, passing the authoritative layer list (if any)
    validation_warnings = validate_workflow(result, layers=layers, list_ok=list_ok)
    result["validation_warnings"] = validation_warnings

    # Log the interaction for Phase 2 training-data collection
    log_interaction(query, result, validation_warnings)

    return JSONResponse(content=result)


@app.get("/api/health")
async def health_check():
    """Check if the LLM backend is reachable."""
    url = BACKEND_URLS.get(LLM_BACKEND, "")
    try:
        headers = {}
        if LLM_BACKEND in OPENAI_COMPATIBLE and API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
        async with httpx.AsyncClient(timeout=5.0) as client:
            if LLM_BACKEND == "ollama":
                resp = await client.get(f"{OLLAMA_HOST}/api/tags")
            else:
                resp = await client.get(
                    url.replace("/chat/completions", "/models"), headers=headers)
            return {
                "status": "ok",
                "backend": LLM_BACKEND,
                "model": MODEL_NAME,
                "backend_url": url,
            }
    except Exception as e:
        return {
            "status": "error",
            "backend": LLM_BACKEND,
            "error": str(e),
            "hint": f"Start {LLM_BACKEND} first. E.g.: docker compose up -d",
        }


@app.get("/api/ontology")
async def get_ontology():
    """Return the GIS ontology for the front end to display tool references."""
    ontology_path = Path(__file__).parent / "gis_ontology.json"
    ontology = json.loads(ontology_path.read_text())
    return JSONResponse(content=ontology)


@app.get("/")
async def serve_frontend():
    """Serve the visual frontend with keyword annotations and workflow diagram."""
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Text-to-Spatial-Model</title>
<style>
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  max-width:960px;margin:2rem auto;padding:0 1.5rem;color:#1a1a1a;background:#fff}
h1{font-size:1.3rem;font-weight:600;margin:0 0 .25rem}
.subtitle{font-size:.8rem;color:#666;margin:0 0 1rem}
.status{font-size:.8rem;padding:6px 12px;border-radius:6px;display:inline-block;margin-bottom:1rem}
.status-ok{background:#ecfdf5;color:#065f46}
.status-err{background:#fef2f2;color:#991b1b}
.status-wait{background:#f0f9ff;color:#1e40af}
.input-row{display:flex;gap:8px;margin-bottom:1.5rem}
.input-row input{flex:1;padding:10px 14px;font-size:14px;border:1px solid #d1d5db;border-radius:8px;outline:none}
.input-row input:focus{border-color:#3b82f6;box-shadow:0 0 0 3px rgba(59,130,246,.1)}
.input-row button{padding:10px 24px;border-radius:8px;background:#2563eb;color:#fff;border:none;cursor:pointer;font-size:14px;font-weight:500;white-space:nowrap}
.input-row button:hover{background:#1d4ed8}
.input-row button:disabled{background:#93c5fd;cursor:wait}
.section-label{font-size:.7rem;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:.5px;margin:1.5rem 0 .5rem}
.annotated{padding:12px 16px;border:1px solid #e5e7eb;border-radius:8px;background:#fafafa;font-size:14px;line-height:2.2}
.kw{padding:1px 0;border-bottom:2px solid;font-weight:600;cursor:default}
.kw-input{border-color:#2563eb;color:#1d4ed8}
.kw-tool{border-color:#d97706;color:#b45309}
.kw-param{border-color:#7c3aed;color:#6d28d9}
.kw-spatial{border-color:#059669;color:#047857}
table.mapping{width:100%;border-collapse:collapse;font-size:.8rem;margin-top:.5rem}
table.mapping th{text-align:left;padding:6px 8px;border-bottom:1px solid #d1d5db;color:#6b7280;font-weight:600;font-size:.7rem;text-transform:uppercase;letter-spacing:.3px}
table.mapping td{padding:6px 8px;border-bottom:1px solid #f3f4f6;vertical-align:top}
table.mapping tr:hover td{background:#f9fafb}
.pill{display:inline-block;font-size:.65rem;padding:1px 8px;border-radius:10px;font-weight:600}
.pill-input{background:#dbeafe;color:#1e40af}
.pill-tool{background:#fef3c7;color:#92400e}
.pill-param{background:#ede9fe;color:#5b21b6}
.pill-spatial{background:#d1fae5;color:#065f46}
.canvas{border:1px solid #e5e7eb;border-radius:12px;padding:20px;background:#fafafa;overflow-x:auto;margin-top:.5rem}
.legend{display:flex;gap:14px;margin-top:8px;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:5px;font-size:.7rem;color:#6b7280}
.legend-sw{width:16px;height:11px;border-radius:3px;border:1px solid}
.notes-box{margin-top:.5rem;padding:10px 14px;background:#fffbeb;border:1px solid #fde68a;border-radius:8px;font-size:.8rem;color:#92400e}
.notes-box strong{font-weight:600}
.notes-box ul{margin:.25rem 0 0;padding-left:1.2rem}
.notes-box li{margin:.15rem 0}
.warn-box{margin-top:.5rem;padding:8px 14px;background:#fef2f2;border:1px solid #fecaca;border-radius:8px;font-size:.8rem;color:#991b1b}
.toggle-row{display:flex;gap:8px;margin-top:10px}
.toggle-row button{font-size:.75rem;padding:4px 14px;border:1px solid #d1d5db;border-radius:6px;background:#fff;cursor:pointer;color:#374151}
.toggle-row button:hover{background:#f3f4f6}
.raw-json{margin-top:.5rem;padding:12px 16px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;font-size:.7rem;font-family:ui-monospace,monospace;white-space:pre-wrap;max-height:400px;overflow-y:auto;display:none}
.code-wrap{position:relative}
.code-block{margin:.5rem 0 0;padding:14px 16px;background:#0f172a;color:#e2e8f0;border-radius:8px;font-size:.75rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre;overflow-x:auto;line-height:1.5}
.copy-btn{position:absolute;top:10px;right:10px;font-size:.7rem;padding:3px 12px;border:1px solid #334155;border-radius:6px;background:#1e293b;color:#cbd5e1;cursor:pointer}
.copy-btn:hover{background:#334155}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid #93c5fd;border-top-color:#2563eb;border-radius:50%;animation:spin .6s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
.empty{text-align:center;padding:3rem 1rem;color:#9ca3af;font-size:.85rem}
.crs-note{font-size:.72rem;color:#6b7280;background:#f9fafb;border:1px solid #e5e7eb;border-left:3px solid #2563eb;border-radius:6px;padding:7px 12px;margin-bottom:1rem}
.layers-row{margin:-0.75rem 0 0.5rem}
.layers-row input{width:100%;padding:8px 12px;font-size:13px;border:1px solid #d1d5db;border-radius:8px;outline:none}
.layers-row input:focus{border-color:#3b82f6;box-shadow:0 0 0 3px rgba(59,130,246,.1)}
</style>
</head>
<body>
<h1>Text-to-Spatial-Model</h1>
<p class="subtitle">Translate natural language to GIS workflow models — Phase 1 Local LLM</p>
<div id="status" class="status status-wait">Checking LLM backend...</div>
<div class="input-row">
  <input id="query" placeholder="e.g. How many acres of each land cover type within 200 feet of a stream?">
  <button id="btn" onclick="generate()">Generate Model</button>
</div>
<div class="layers-row">
  <input id="layers" placeholder="Optional — available layers, e.g.  streams (line), property (polygon), landcover (polygon)">
</div>
<div class="crs-note">Unless otherwise indicated, analyses assume an appropriate projected UTM coordinate system (native units: meters). Declaring your layers above (in <code>name (geometry)</code> form) lets the model use only real inputs instead of guessing.</div>
<div id="output"><div class="empty">Enter a spatial analysis task above to generate a workflow model</div></div>

<script>
fetch('/api/health').then(r=>r.json()).then(d=>{
  const s=document.getElementById('status');
  if(d.status==='ok'){s.className='status status-ok';s.textContent='Connected: '+d.backend+' / '+d.model}
  else{s.className='status status-err';s.textContent='Error: '+(d.error||'unknown')+'. '+(d.hint||'')}
});

async function generate(){
  const q=document.getElementById('query').value.trim();
  if(!q)return;
  const btn=document.getElementById('btn');
  const out=document.getElementById('output');
  btn.disabled=true;btn.textContent='Generating...';
  out.innerHTML='<div class="empty"><span class="spinner"></span> Sending to LLM — this may take a few minutes with the 70B model...</div>';
  try{
    const resp=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,layers:(document.getElementById('layers').value||'')})});
    const data=await resp.json();
    if(data.error){out.innerHTML='<div class="warn-box">'+esc(data.error)+'</div>';return}
    let html='';
    // Keywords
    if(data.keywords&&data.keywords.length>0){
      html+='<div class="section-label">Keyword Analysis</div>';
      html+='<div class="annotated">'+buildAnnotated(q,data.keywords)+'</div>';
      html+='<div class="section-label">Phrase → Tool Mapping</div>';
      html+='<table class="mapping"><thead><tr><th>Phrase</th><th>Interpretation</th><th>Maps To</th><th>Role</th></tr></thead><tbody>';
      data.keywords.forEach(k=>{
        const t=k.type||'input';
        html+='<tr><td><span class="kw kw-'+t+'">'+esc(k.phrase)+'</span></td><td>'+esc(k.interpretation||'')+'</td><td style="font-family:monospace;font-size:.75rem">'+esc(k.maps_to||'')+'</td><td><span class="pill pill-'+t+'">'+t+'</span></td></tr>';
      });
      html+='</tbody></table>';
    }
    // Workflow diagram
    if(data.workflow&&data.workflow.nodes&&data.workflow.nodes.length>0){
      html+='<div class="section-label">Spatial Model</div>';
      html+='<div class="canvas" id="canvas"></div>';
      html+='<div class="legend">';
      html+='<div class="legend-item"><div class="legend-sw" style="background:#dbeafe;border-color:#2563eb"></div>Input data</div>';
      html+='<div class="legend-item"><div class="legend-sw" style="background:#fef3c7;border-color:#d97706"></div>Tool</div>';
      html+='<div class="legend-item"><div class="legend-sw" style="background:#d1fae5;border-color:#059669"></div>Derived data</div>';
      html+='<div class="legend-item"><div class="legend-sw" style="background:#ede9fe;border-color:#7c3aed"></div>Output</div>';
      html+='</div>';
    }
    // ArcPy script (assembled from each tool node's arcpy_call, in workflow order)
    if(data.workflow&&data.workflow.nodes&&data.workflow.nodes.length>0){
      const script=buildArcpy(data.workflow);
      if(script){
        html+='<div class="section-label">ArcPy Script</div>';
        html+='<div class="code-wrap"><button class="copy-btn" onclick="copyArcpy()">Copy</button>';
        html+='<pre class="code-block" id="arcpycode">'+esc(script)+'</pre></div>';
      }
    }
    // Validation warnings
    if(data.validation_warnings&&data.validation_warnings.length>0){
      html+='<div class="warn-box"><strong>Validation Warnings</strong>';
      data.validation_warnings.forEach(w=>{html+='<br>⚠ '+esc(w)});
      html+='</div>';
    }
    // Notes
    if(data.notes){
      const n=data.notes;
      let noteHtml='';
      if(n.ambiguities&&n.ambiguities.length) noteHtml+='<strong>Ambiguities:</strong><ul>'+n.ambiguities.map(a=>'<li>'+esc(a)+'</li>').join('')+'</ul>';
      if(n.assumptions&&n.assumptions.length) noteHtml+='<strong>Assumptions:</strong><ul>'+n.assumptions.map(a=>'<li>'+esc(a)+'</li>').join('')+'</ul>';
      if(n.teaching_points&&n.teaching_points.length) noteHtml+='<strong>Teaching Points:</strong><ul>'+n.teaching_points.map(a=>'<li>'+esc(a)+'</li>').join('')+'</ul>';
      if(n.alternatives&&n.alternatives.length) noteHtml+='<strong>Alternatives:</strong><ul>'+n.alternatives.map(a=>'<li>'+esc(a)+'</li>').join('')+'</ul>';
      if(noteHtml) html+='<div class="notes-box">'+noteHtml+'</div>';
    }
    // Toggle raw JSON
    html+='<div class="toggle-row"><button onclick="toggleRaw()">Show Raw JSON</button></div>';
    html+='<div class="raw-json" id="rawjson">'+esc(JSON.stringify(data,null,2))+'</div>';
    out.innerHTML=html;
    // Render SVG diagram after DOM update
    if(data.workflow&&data.workflow.nodes&&data.workflow.nodes.length>0){
      setTimeout(()=>renderDiagram(data.workflow),50);
    }
  }catch(e){out.innerHTML='<div class="warn-box">Error: '+esc(e.message)+'</div>'}
  finally{btn.disabled=false;btn.textContent='Generate Model'}
}

function esc(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}

function buildAnnotated(query,keywords){
  if(!keywords||!keywords.length)return esc(query);
  const sorted=[...keywords].filter(k=>k.start!=null&&k.end!=null).sort((a,b)=>a.start-b.start);
  if(!sorted.length)return esc(query);
  let html='';let last=0;
  sorted.forEach(k=>{
    if(k.start>last) html+=esc(query.slice(last,k.start));
    const t=k.type||'input';
    html+='<span class="kw kw-'+t+'" title="'+esc(k.interpretation||'')+'">'+esc(query.slice(k.start,k.end))+'</span>';
    last=k.end;
  });
  if(last<query.length) html+=esc(query.slice(last));
  return html;
}

function toggleRaw(){const el=document.getElementById('rawjson');el.style.display=el.style.display==='block'?'none':'block'}

function buildArcpy(workflow){
  const nodes=workflow.nodes||[];
  const edges=workflow.edges||[];
  if(!nodes.length)return '';
  // Order nodes the same way the diagram does (columns = topological layers),
  // then emit each TOOL node's arcpy_call in that order.
  const cols=layoutNodes(nodes,edges);
  const ordered=[];
  cols.forEach(c=>c.forEach(n=>ordered.push(n)));
  const lines=[];
  lines.push('import arcpy');
  lines.push('arcpy.env.overwriteOutput = True');
  lines.push('# Set your workspace, e.g.:');
  lines.push('# arcpy.env.workspace = "path/to/your.gdb"');
  lines.push('');
  let any=false;
  ordered.forEach(n=>{
    if(n.type==='tool'&&n.arcpy_call){
      if(n.label) lines.push('# '+n.label);
      lines.push(n.arcpy_call);
      lines.push('');
      any=true;
    }
  });
  if(!any)return '';
  var NL=String.fromCharCode(10);
  // join with newlines and trim trailing blank lines (NL via fromCharCode)
  while(lines.length && lines[lines.length-1]==='') lines.pop();
  return lines.join(NL)+NL;
}

function copyArcpy(){
  const el=document.getElementById('arcpycode');
  if(!el)return;
  const txt=el.innerText;
  navigator.clipboard.writeText(txt).then(()=>{
    const b=document.querySelector('.copy-btn');
    if(b){const o=b.textContent;b.textContent='Copied';setTimeout(()=>b.textContent=o,1200);}
  }).catch(()=>{});
}

const COLORS={
  input:{fill:'#dbeafe',stroke:'#2563eb',text:'#1e40af'},
  tool:{fill:'#fef3c7',stroke:'#d97706',text:'#92400e'},
  derived:{fill:'#d1fae5',stroke:'#059669',text:'#065f46'},
  output:{fill:'#ede9fe',stroke:'#7c3aed',text:'#5b21b6'},
  line:'#9ca3af'
};

function renderDiagram(workflow){
  const canvas=document.getElementById('canvas');
  if(!canvas)return;
  const nodes=workflow.nodes||[];
  const edges=workflow.edges||[];
  const cols=layoutNodes(nodes,edges);
  const colW=150,nodeH=52,padX=28,padY=20,gapX=56,gapY=16;
  const totalW=cols.length*(colW+gapX)-gapX+padX*2;
  const maxR=Math.max(...cols.map(c=>c.length));
  const totalH=maxR*(nodeH+gapY)-gapY+padY*2+16;
  let svg='<svg viewBox="0 0 '+totalW+' '+totalH+'" width="100%" xmlns="http://www.w3.org/2000/svg">';
  svg+='<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1.5L8 5L2 8.5" fill="none" stroke="'+COLORS.line+'" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>';
  const pos={};
  cols.forEach((col,ci)=>{
    const x=padX+ci*(colW+gapX);
    const colH=col.length*(nodeH+gapY)-gapY;
    const sy=padY+(totalH-padY*2-colH)/2;
    col.forEach((n,ri)=>{
      const y=sy+ri*(nodeH+gapY);
      pos[n.id]={x,y,cx:x+colW/2,cy:y+nodeH/2};
      const c=COLORS[n.type]||COLORS.derived;
      const isOval=n.type!=='tool';
      const rx=isOval?nodeH/2:4;
      const label=(n.label||'').split('\\n');
      if(label.length===1){const parts=n.label.split(/(?<=\\S)\\s+/);if(parts.length>2){label[0]=parts.slice(0,Math.ceil(parts.length/2)).join(' ');label[1]=parts.slice(Math.ceil(parts.length/2)).join(' ')}}
      const ly=label.length>1?y+nodeH/2-7:y+nodeH/2;
      svg+='<rect x="'+x+'" y="'+y+'" width="'+colW+'" height="'+nodeH+'" rx="'+rx+'" fill="'+c.fill+'" stroke="'+c.stroke+'" stroke-width="1"/>';
      label.forEach((l,li)=>{svg+='<text x="'+(x+colW/2)+'" y="'+(ly+li*14)+'" text-anchor="middle" dominant-baseline="central" fill="'+c.text+'" font-size="12" font-weight="600" font-family="-apple-system,sans-serif">'+esc(l)+'</text>'});
    });
  });
  edges.forEach(e=>{
    const fromId=e.from||e.source;const toId=e.to||e.target;
    const a=pos[fromId],b=pos[toId];if(!a||!b)return;
    const x1=a.x+colW,y1=a.cy,x2=b.x,y2=b.cy;
    if(Math.abs(x2-x1)<10){svg+='<line x1="'+a.cx+'" y1="'+(a.y+nodeH)+'" x2="'+b.cx+'" y2="'+b.y+'" stroke="'+COLORS.line+'" stroke-width="1" marker-end="url(#ah)"/>';}
    else{const mx=x1+(x2-x1)*.5;svg+='<path d="M'+x1+' '+y1+'C'+mx+' '+y1+' '+mx+' '+y2+' '+x2+' '+y2+'" fill="none" stroke="'+COLORS.line+'" stroke-width="1" marker-end="url(#ah)"/>';}
  });
  svg+='</svg>';
  canvas.innerHTML=svg;
}

function layoutNodes(nodes,edges){
  const nm={};nodes.forEach(n=>nm[n.id]=n);
  const cols=[];const placed=new Set();let rem=new Set(nodes.map(n=>n.id));
  while(rem.size>0){
    const ready=[...rem].filter(id=>edges.filter(e=>(e.to||e.target)===id&&!placed.has(e.from||e.source)).length===0);
    const col=[];
    if(ready.length===0){const id=[...rem][0];col.push(nm[id]);placed.add(id);rem.delete(id)}
    else ready.forEach(id=>{col.push(nm[id]);placed.add(id);rem.delete(id)});
    cols.push(col);
  }
  return cols;
}

document.getElementById('query').addEventListener('keydown',e=>{if(e.key==='Enter')generate()});
</script>
</body>
</html>""")


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Text-to-Spatial-Model — Phase 1 (vector overlay only)")
    print(f"Host:    IRENE")
    print(f"Backend: {LLM_BACKEND} ({BACKEND_URLS[LLM_BACKEND]})")
    print(f"Model:   {MODEL_NAME}")
    print(f"Logging: {'ON → ' + str(QUERY_LOG_PATH) if QUERY_LOG_ENABLED else 'OFF'}")
    print(f"{'='*60}\n")
    bind_host = os.environ.get("BIND_HOST", "127.0.0.1")
    bind_port = int(os.environ.get("BIND_PORT", "8080"))
    print(f"Starting web server on http://{bind_host}:{bind_port}")
    if bind_host == "127.0.0.1":
        print("Bound to loopback. Reach it from your laptop via SSH tunnel:")
        print("  ssh -L 8080:localhost:8080 <user>@128.192.48.10")
        print("  then open http://localhost:8080 on your laptop")
    print("Make sure your LLM backend is running first!\n")

    uvicorn.run(app, host=bind_host, port=bind_port)
