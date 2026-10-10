# stl2step

Converts STL meshes into real B-rep solids (STEP/IGES/BREP) that CAD programs can edit —
the same idea as Fusion 360's "Mesh to BRep", built on OpenCASCADE (`cadquery-ocp`).

    ./stl2step part.stl                 # -> part.step next to it
    ./stl2step *.stl                    # batch
    ./stl2step part.stl -o part.iges
    ./stl2step                          # GUI with file picker

Options: `--no-merge` (one face per triangle), `--angle DEG` (how far apart triangle normals
can be and still merge into one flat face, default 0.5), `--weld MM` (vertex weld tolerance).

Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`

## What you get
- Flat regions become single planar faces, so you can sketch on them, push/pull, fillet edges.
- Curved regions (holes, fillets, bosses) stay faceted — strips of flat faces.
- Non-watertight meshes export as an open surface body with a warning, not a solid.

## Notes
- OCP 8 exposes `TopoDS.Edge(...)` (no `_s`), but other static methods still use `_s`.
- The coplanar merge leaves vertices slightly off the fitted planes; the `ShapeFix_Shape`
  pass afterwards is what makes BRepCheck (and FreeCAD) accept the solid.
