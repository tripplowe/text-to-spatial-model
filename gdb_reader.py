"""
gdb_reader.py — read a file geodatabase into the layer inventory the app consumes.

Phase 3 (v3): the user places a FILE GEODATABASE (modern Esri .gdb *folder* —
NOT the obsolete .mdb personal geodatabase) on the host, and the app reads its
vector feature classes (names, geometry types, attribute fields) to build the
authoritative layer inventory that previously had to be typed by hand.

Reading is done with pyogrio (GDAL's built-in OpenFileGDB driver — no Esri SDK
or ArcGIS license required). Vector layers only: raster and non-spatial tables
are reported but flagged out-of-scope for this vector phase.

The inventory structure matches what build_layer_context()/validate_workflow()
already consume, with an added optional "fields" list per layer:

    {
      "name": "streams",
      "geometry": "vector-line",          # normalized to the ontology's types
      "fields": [{"name": "strm_type", "type": "String"}, ...],
      "raw": "streams (line)"
    }
"""

import os
from pathlib import Path

# Map OGR/pyogrio geometry-type strings to the ontology's vector data types.
# pyogrio returns geometry types like "Point", "LineString", "Polygon",
# "MultiPolygon", "3D MultiLineString", etc. We normalize by keyword.
_GEOM_NORMALIZE = [
    ("point", "vector-point"),
    ("line", "vector-line"),       # LineString, MultiLineString, etc.
    ("polygon", "vector-polygon"), # Polygon, MultiPolygon
]


def _normalize_geometry(geom_type: str):
    """
    Map a pyogrio/OGR geometry-type string to a vector ontology type, or None
    if it is not a recognized vector geometry (e.g. None/raster/table).
    """
    if not geom_type:
        return None
    g = geom_type.strip().lower()
    for needle, vtype in _GEOM_NORMALIZE:
        if needle in g:
            return vtype
    return None


def read_geodatabase(gdb_path: str):
    """
    Read a file geodatabase and return (inventory, report).

      inventory : list of vector-layer dicts (see module docstring) — only
                  layers whose geometry maps to a vector type are included.
      report    : dict with diagnostics:
                  {
                    "ok": bool,
                    "gdb": str,
                    "vector_layers": [names...],
                    "skipped": [{"name":..., "reason":...}],  # non-vector / unreadable
                    "error": str | None,
                  }

    Never raises for normal "bad path / not a gdb / unreadable layer" cases;
    those come back in report so the caller can surface them cleanly.
    """
    report = {"ok": False, "gdb": gdb_path, "vector_layers": [],
              "skipped": [], "error": None}
    inventory = []

    p = Path(gdb_path)
    if not p.exists():
        report["error"] = f"Path does not exist: {gdb_path}"
        return inventory, report
    if not p.is_dir() or not str(p).lower().endswith(".gdb"):
        report["error"] = (
            f"Not a file geodatabase (expected a '.gdb' directory): {gdb_path}. "
            "Note: the obsolete .mdb personal geodatabase is not supported."
        )
        return inventory, report

    try:
        import pyogrio
    except ImportError:
        report["error"] = (
            "pyogrio is not installed. Add 'pyogrio' to requirements.txt and "
            "rebuild the image (it bundles GDAL's OpenFileGDB driver)."
        )
        return inventory, report

    # Enumerate layers. pyogrio.list_layers returns an ndarray of
    # [[name, geometry_type], ...]; geometry_type is None for nonspatial tables.
    try:
        layers = pyogrio.list_layers(str(p))
    except Exception as e:  # noqa: BLE001 — surface any read failure cleanly
        report["error"] = f"Could not read geodatabase layers: {e}"
        return inventory, report

    for row in layers:
        name = str(row[0])
        geom_type = row[1] if len(row) > 1 else None
        vtype = _normalize_geometry(str(geom_type) if geom_type is not None else "")
        if vtype is None:
            report["skipped"].append({
                "name": name,
                "reason": f"non-vector or unsupported geometry ({geom_type!r})",
            })
            continue

        # Read field (attribute) definitions without loading geometry/rows.
        fields = []
        try:
            info = pyogrio.read_info(str(p), layer=name)
            fnames = list(info.get("fields", []) or [])
            ftypes = list(info.get("dtypes", []) or [])
            for i, fn in enumerate(fnames):
                ftype = ftypes[i] if i < len(ftypes) else ""
                fields.append({"name": str(fn), "type": str(ftype)})
        except Exception as e:  # noqa: BLE001 — fields are best-effort
            report["skipped"].append({
                "name": name,
                "reason": f"geometry OK but field read failed: {e}",
            })
            # still include the layer with empty fields rather than dropping it

        geom_label = {"vector-point": "point", "vector-line": "line",
                      "vector-polygon": "polygon"}[vtype]
        inventory.append({
            "name": name,
            "geometry": vtype,
            "fields": fields,
            "raw": f"{name} ({geom_label})",
        })
        report["vector_layers"].append(name)

    report["ok"] = True
    return inventory, report


def inventory_to_context_string(inventory):
    """
    Render the parsed inventory back into the same 'name (geometry)' text format
    that build_layer_context() already understands, optionally with field hints.
    This lets the gdb inventory flow through the EXISTING v2 consumer path with
    no change to the prompt-injection logic.
    """
    parts = []
    for layer in inventory:
        geom_label = {"vector-point": "point", "vector-line": "line",
                      "vector-polygon": "polygon"}.get(layer["geometry"], "polygon")
        entry = f"{layer['name']} ({geom_label})"
        fields = layer.get("fields") or []
        if fields:
            fnames = ", ".join(f["name"] for f in fields)
            entry += f" [fields: {fnames}]"
        parts.append(entry)
    return "; ".join(parts)


# Self-test / CLI: run directly on IRENE to verify against the real .gdb.
#   python3 gdb_reader.py ~/geollm/geodata/geollm_tonfdata.gdb
if __name__ == "__main__":
    import sys
    import json
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "GDB_PATH", "geodata/geollm_tonfdata.gdb")
    inv, rep = read_geodatabase(path)
    print("=== REPORT ===")
    print(json.dumps(rep, indent=2))
    print("\n=== VECTOR INVENTORY ===")
    print(json.dumps(inv, indent=2))
    print("\n=== CONTEXT STRING (as the model would receive) ===")
    print(inventory_to_context_string(inv))
