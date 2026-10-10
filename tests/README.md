# Tests, corpus and scoring

    .venv/bin/pip install -r requirements-dev.txt
    .venv/bin/python -m pytest tests -q          # ~30 s
    .venv/bin/python tests/score.py              # score table for the whole corpus
    .venv/bin/python tests/make_corpus.py        # regenerate tests/corpus (deterministic, ~1 s)

## Corpus (`tests/corpus/`)
11 CAD parts built with OCP (`make_corpus.py` → `PARTS`): box, box_hole, plate_holes_cbore,
l_bracket (R5 cylindrical fillet), boss_plate, slot_plate, tube, countersink_dome (cone +
sphere), flanged_shaft (torus fillet), rounded_block (12 cylinders + 8 sphere corners),
two_bodies (2 disjoint solids). Each has:

- `<name>.json` — ground truth: `volume`, `bbox`, `solids`, `faces`, `faces_by_type`
  (plane/cylinder/cone/sphere/torus/bspline/other), every cylinder (`origin` = axis point
  nearest the world origin, canonical-sign `axis`, `radius`), and per-mesh stats under `meshes`.
- `<name>_coarse.stl` (deflection 0.1 mm / 0.5 rad) and `<name>_fine.stl` (0.01 mm / 0.1 rad).
- `<name>_<res>.labels.npz` — per-triangle segmentation in STL order: `labels[i]` is the
  B-rep face of triangle i; per-face `face_type` (index into `type_names`), `face_body`,
  `face_sense` (+1 convex/boss, -1 concave/hole), `plane_origin/plane_normal` (outward),
  `cyl_origin/cyl_axis/cyl_radius`, and cone/sphere/torus params. NaN where not applicable.

Truth faces are counted after `ShapeUpgrade_UnifySameDomain`, i.e. one face per logical
surface patch. STLs are written by our own code (not `StlAPI_Writer`) so triangle order
matches the labels; generation asserts the float32 file is closed and within 1% of the true
volume.

## Scoring (`score.py`)
Converts each STL with `--impl module:function` (default `stl2step:convert`; also accepts
`path/to/file.py:function`), re-reads the STEP and reports validity, solids, volume error vs
the true B-rep (`vol%`) and vs the input mesh (`meshvol%`), faces (got/truth), cylinders
recovered (radius within 1%, axis within 1°, axis line within max(1% r, 0.05 mm); one-to-one,
coaxial pieces merged), time and STEP size. Pass `.step` files to score them without
converting. Exits 1 if any output is invalid, has the wrong solid count, or `|vol%| > 1`.
`--res coarse` for a quick run, `--json out.json` for a machine-readable dump.

`test_convert.py` requires valid solids and <0.5% volume error everywhere; cylinder recovery
and face-count checks are `xfail(strict=False)` until segment/rebuild land. Point it at
another converter with `STL2STEP_IMPL=module:function`.

## Traps found while building this
- OCP 8 has no `TopTools_IndexedMapOfShape`; use
  `OCP.collections.IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher`.
- `Poly_Triangle.Get()` returns a 1-based `(n1, n2, n3)` tuple; nodes need the face
  `TopLoc_Location` applied, and REVERSED faces need their winding flipped.
- BRepMesh emits a zero-area triangle at sphere poles; it duplicates an edge, so the corpus
  writer drops triangles that collapse in float32 (`meshes.<res>.dropped_degenerate`).
- Coarse meshes are inscribed, so convex parts lose volume (up to -0.4% on flanged_shaft)
  while holes gain it; `meshvol%` isolates what the converter itself changed.
