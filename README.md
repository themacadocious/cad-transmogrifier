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
- `--no-repair` — skip mesh repair

Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`

## What you get

By default the mesh is **segmented** into planes, cylinders, cones, spheres and tori
(`segment.py`) and
**rebuilt** as true analytic faces (`rebuild.py`): a drilled hole becomes one cylinder you
can select and resize, a countersink becomes a cone, flat walls become single planes.
Edges between them are real lines, circles and ellipses (exact surface intersections).

Anything that isn't recognised (organic/freeform areas, threads) stays as flat facets,
locally — the rest of the part still gets clean faces. If the rebuild can't produce a
valid closed solid with the mesh's volume, it falls back to the faceted conversion, so
the result is never worse than `--faceted`.

Broken meshes are repaired first (`repair.py`): non-manifold edges, T-junction cracks,
small holes, inconsistent winding, slivers and cap triangles. Internal voids stay voids;
bodies that overlap are fused into one solid, bodies that only touch stay separate.
A mesh that still isn't closed after repair exports as an open surface with a warning.

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

## Known limitations
- **Speed on big freeform meshes**: a 30k-triangle part with mostly organic/threaded
  surface takes ~20 s (most of it building and writing ~12k facet faces in OCCT);
  some freeform meshes (78k triangles) still spend ~50 s in segmentation on failed
  cylinder attempts. Prismatic/CAD-like parts are fast (corpus: 3 s for 22 files).
- **Large sphere patches** (hemisphere domes) come back as a few sphere pieces rather
  than one face — exact geometry, just split.
- **Lossy or mixed-precision exports** can still pick up a few loose stray spheres/tori
  on organic areas (on exact CAD exports a tightness rule suppresses them).
- **Polygonal prisms vs coarse cylinders** are told apart by chordal deviation (a real
  export's facets hug the curve; an octagon's don't). A lone 8-facet hole of ~3 mm
  radius is still read as a cylinder. `--min-cyl-facets N` overrides (0 = always
  cylinder).
- **Overlapping bodies** are fused with a boolean, which can take a while with many
  faceted bodies; touching-only bodies stay separate solids.
- **Mesh repair** fixes non-manifold edges, cracks, small holes, winding, slivers, cap
  and needle triangles, and small self-intersections; larger self-intersecting areas
  (e.g. crossing thread flanks) are left in place with a warning. It may make tiny,
  reported edge flips even on otherwise clean meshes. `--no-repair` skips it.
