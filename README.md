# stl2step

Converts STL meshes into real B-rep solids (STEP/IGES/BREP) that CAD programs can edit —
like Fusion 360's "Mesh to BRep" with the prismatic option, built on OpenCASCADE
(`cadquery-ocp`).

    ./stl2step part.stl                 # -> part.step next to it
    ./stl2step *.stl                    # batch
    ./stl2step part.stl -o part.iges
    ./stl2step                          # GUI with file picker

Options:
- `--faceted` — skip surface recognition; only merge coplanar triangles
- `--fit-tol MM` — how far a vertex may sit from a plane/cylinder and still belong to it
  (default: 2e-5 × bounding-box diagonal, at least 0.001 mm)
- `--angle DEG` — max angle between triangles merged into one flat face (default 0.5)
- `--no-merge` — one face per triangle
- `--weld MM` — vertex weld tolerance

Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`

## What you get

By default the mesh is **segmented** into planes, cylinders and cones (`segment.py`) and
**rebuilt** as true analytic faces (`rebuild.py`): a drilled hole becomes one cylinder you
can select and resize, a countersink becomes a cone, flat walls become single planes.
Edges between them are real lines, circles and ellipses (exact surface intersections).

Anything that isn't recognised (organic/freeform areas, threads) stays as flat facets,
locally — the rest of the part still gets clean faces. If the rebuild can't produce a
valid closed solid with the mesh's volume, it falls back to the faceted conversion, so
the result is never worse than `--faceted`.

Non-watertight or non-manifold meshes export as an open surface body with a warning.

## Testing

`tests/` holds a deterministic ground-truth corpus (CAD parts meshed at two resolutions,
with per-triangle labels) — see `tests/README.md`.

    .venv/bin/pip install -r requirements-dev.txt
    .venv/bin/python tests/score.py      # per-file table: validity, volume error, faces, cylinders
    .venv/bin/python -m pytest tests

## OCCT traps worth knowing
- OCP 8 exposes `TopoDS.Edge(...)` (no `_s`), but other static methods still use `_s`,
  and returned handles are already their concrete type (no `DownCast`).
- `ShapeFix_Face` replaces every edge of the face it fixes. Without a shared
  `ShapeBuild_ReShape` context applied to the whole shell, curved faces silently detach
  from their neighbours and `ShapeFix_Shape` splits the solid into open shells — which
  still pass `BRepCheck` and integrate to the right volume. Check for solids explicitly.
- A curved face whose two boundary loops both circle the axis needs a seam;
  `ShapeFix_Face` only adds one for plain circular ends. rebuild.py splits such regions
  into half-shells instead and lets `ShapeUpgrade_UnifySameDomain` rejoin them.
- `GeomAPI_IntSS` returns trimmed curves; trim/reverse their `BasisCurve()` instead.
