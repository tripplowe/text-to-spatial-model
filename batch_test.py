#!/usr/bin/env python3
"""
batch_test.py — fire a list of test queries at the Text-to-Spatial-Model API
and print a compact pass/fail summary for each.

This is an evaluation aid: it removes the browser round-trip so you can run many
queries quickly and eyeball the tools chosen, the workflow shape, and any
validation warnings. It does NOT render the diagram (that is browser-side and
not needed for evaluating correctness).

USAGE
  # Run the built-in sample query set against a locally reachable API:
  python3 batch_test.py

  # Use your own newline-separated query file:
  python3 batch_test.py --queries my_queries.txt

  # Point at a different host (e.g. via SSH tunnel localhost:8080):
  python3 batch_test.py --url http://localhost:8080

  # Show the full workflow (nodes + edges), not just the summary:
  python3 batch_test.py --verbose

  # Save every full JSON response to a file for later review:
  python3 batch_test.py --out results.jsonl

NOTES
  - Reads results from the same /api/generate endpoint the web UI uses.
  - "PASS" here means only that the response was well-formed and produced a
    workflow with no validation warnings. It does NOT verify the workflow is
    semantically correct for the query — that still needs your judgment. Treat
    it as a fast triage, not ground truth.
  - Requires only the Python standard library (urllib). No extra installs.
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


# A small starter set covering the patterns exercised by the few-shot examples.
# Edit freely or supply your own with --queries.
DEFAULT_QUERIES = [
    "How many acres of each landcover type?",
    "How many acres of each landcover type within 200 feet of a stream?",
    "How many acres do we have on the property within 200 feet of a stream?",
    "What are the UTM coordinates of each turtle nest?",
    "What are the long/lat coordinates of the sample locations?",
    "Determine the unprojected coordinates of the loading dock.",
    "How many miles of each stream type are on the property?",
    "Select all the parcels zoned residential.",
    "Show the parcels that are outside a 100 foot buffer of the wetlands.",
]


def call_api(url: str, query: str, timeout: float) -> dict:
    """POST one query to /api/generate and return the parsed JSON response."""
    endpoint = url.rstrip("/") + "/api/generate"
    payload = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def summarize(result: dict) -> dict:
    """Pull the evaluation-relevant fields out of a response."""
    wf = result.get("workflow", {}) or {}
    nodes = wf.get("nodes", []) or []
    edges = wf.get("edges", []) or []
    tools = [n.get("tool_name") for n in nodes if n.get("tool_name")]
    warnings = result.get("validation_warnings", []) or []
    has_error = "error" in result
    return {
        "tools": tools,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "warnings": warnings,
        "error": result.get("error") if has_error else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch-test the spatial-model API.")
    ap.add_argument("--url", default="http://localhost:8080",
                    help="Base URL of the API (default: http://localhost:8080)")
    ap.add_argument("--queries", default=None,
                    help="Path to a newline-separated file of queries.")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="Per-request timeout in seconds (default: 180).")
    ap.add_argument("--verbose", action="store_true",
                    help="Print full node/edge detail for each query.")
    ap.add_argument("--out", default=None,
                    help="Optional path to write full JSON responses (JSON Lines).")
    args = ap.parse_args()

    if args.queries:
        with open(args.queries, encoding="utf-8") as f:
            queries = [ln.strip() for ln in f if ln.strip()]
    else:
        queries = DEFAULT_QUERIES

    out_fh = open(args.out, "w", encoding="utf-8") if args.out else None

    n_pass = n_warn = n_fail = 0
    print(f"\nRunning {len(queries)} queries against {args.url}\n" + "=" * 70)

    for i, q in enumerate(queries, 1):
        t0 = time.time()
        try:
            result = call_api(args.url, q, args.timeout)
        except urllib.error.URLError as e:
            print(f"\n[{i}] {q}\n    REQUEST FAILED: {e}")
            n_fail += 1
            continue
        except Exception as e:  # noqa: BLE001 - want any failure surfaced per-query
            print(f"\n[{i}] {q}\n    ERROR: {e}")
            n_fail += 1
            continue
        dt = time.time() - t0

        s = summarize(result)
        if out_fh:
            out_fh.write(json.dumps({"query": q, "result": result}) + "\n")

        # Classify
        if s["error"]:
            status = "FAIL (bad/unparseable response)"
            n_fail += 1
        elif s["warnings"]:
            status = f"WARN ({len(s['warnings'])})"
            n_warn += 1
        else:
            status = "PASS"
            n_pass += 1

        print(f"\n[{i}] {q}")
        print(f"    {status}   ({dt:.1f}s)")
        print(f"    tools: {' -> '.join(s['tools']) if s['tools'] else '(none)'}")
        print(f"    nodes: {s['n_nodes']}  edges: {s['n_edges']}")
        if s["error"]:
            print(f"    error: {s['error']}")
        for w in s["warnings"]:
            print(f"    ⚠ {w}")
        if args.verbose and not s["error"]:
            for n in result.get("workflow", {}).get("nodes", []):
                tn = f" [{n['tool_name']}]" if n.get("tool_name") else ""
                print(f"      - {n.get('id')}: {n.get('label')}"
                      f" ({n.get('type')}, {n.get('data_type')}){tn}")

    if out_fh:
        out_fh.close()

    print("\n" + "=" * 70)
    print(f"Summary: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL "
          f"(of {len(queries)})")
    print("Note: PASS = well-formed + no validation warnings. It does NOT "
          "confirm the workflow is correct for the query — review those "
          "yourself.\n")
    # Non-zero exit if anything failed outright, so this can gate a script.
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
