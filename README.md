# cad-transmogrifier

*Named after Calvin's cardboard-box Transmogrifier from Bill Watterson's Calvin and Hobbes —
you put one thing in and something else comes out. (No affiliation, and no artwork here;
just a fan's nod.)*

Turn an STL or 3MF mesh into a real, editable solid — a STEP file you can open in
**FreeCAD**, **Onshape** or any other CAD program and actually work with: select a hole
and change its diameter, sketch on a flat face, add fillets, cut it with booleans.

```
./transmogrify Orin_nano_case_main.3MF
Orin_nano_case_main.3MF: 3304 triangles, 1612 vertices
  prismatic: 77 cone, 47 cylinder, 334 plane faces
  solid volume 76673.817 mm^3, valid
  wrote Orin_nano_case_main.step (1.6 MB) in 1.6s
```

3,304 triangles became 458 faces: the screw holes and slot ends are true cylinders at
their design radii (1.25 / 2.25 / 2.6 / 3.0 mm), the chamfers are exact 45° cones, every
flat wall is one face.

## Why this exists

Fusion 360 has the best mesh-to-solid workflow I've used: import an STL or 3MF, convert
it to a B-rep with the *prismatic* option, and you get real cylinders and planes you can
edit. But Fusion 360 doesn't run well on Linux, costs a decent amount of money, and wants
your files saved to Autodesk's cloud first. Every other tool I tried (FreeCAD included)
just imports the mesh as a mesh — at best a "solid" made of thousands of triangles that
you can't meaningfully edit.

cad-transmogrifier is that one conversion step, offline, free, on Linux: mesh in, editable STEP
out, then do the actual CAD work in FreeCAD or Onshape. It is not affiliated with
Autodesk.

## Similar projects (go look at these first)

I built this before checking what already existed — which I should have done. Other people
have put real time and care into this problem, and their projects deserve the credit:

- **[BlinkingSun/stl2step](https://github.com/BlinkingSun/stl2step)** (MIT, C++) — a mature
  OpenCASCADE-based STL→STEP engine with a CLI, an embeddable library, a WebAssembly build
  and the SolidOut desktop app. Recovers planes, cylinders and fillets, with mesh repair.
- **[The-Dorkknight/stl2step](https://github.com/The-Dorkknight/stl2step)** (PolyForm
  Noncommercial) — a portable browser-based app that also reconstructs cones, spheres, tori
  and fits NURBS to organic areas; reads STL/OBJ/PLY/3MF and OpenSCAD, with FreeCAD and
  Fusion plugins.
- **[build123d](https://github.com/gumyr/build123d)**'s `brep_from_stl.detect_primitives`
  (Apache 2.0) — detects planes, spheres and cylinders on a mesh as analytic faces inside
  the build123d CAD library.
- **FreeCAD** itself: *Part → Create shape from mesh → Convert to solid → Refine shape*
  gives a faceted solid with coplanar faces merged, and the Reverse Engineering workbench
  fits surfaces interactively.

What this one adds to that list, for what it's worth: it is open source (LGPL), installs
with pip, rebuilds cones/spheres/tori as true analytic surfaces, reads slicer 3MF files
(PrusaSlicer, Bambu Studio, Orca), and repairs broken meshes first. How the tools compare on
the same files is in [`docs/similar-projects.md`](docs/similar-projects.md).

## Made with AI — vibe-coded with Claude

Up front: this project was **vibe-coded**. The code, the tests and most of this README
were written by Anthropic's Claude (Claude Code, with a few parallel sub-agents working on
segmentation, mesh repair and the test corpus), directed and tested by me on my own
printer files. Commits carry `Co-Authored-By: Claude` trailers.

What that means for you:
- It works on the files I've thrown at it (a ground-truth test corpus plus ~70 real
  STL/3MF downloads), and every output is checked by OpenCASCADE's validator and against
  the mesh volume before it's written — but it has not been reviewed line by line by a
  human CAD-kernel expert.
- If a conversion looks wrong, `--faceted` gives the plain, conservative result, and
  issues with the offending file attached are very welcome.

## Install (Linux)

Needs Python 3.11+ (tested on 3.13, Debian 13). Everything else comes from pip —
OpenCASCADE via [`cadquery-ocp`](https://pypi.org/project/cadquery-ocp/), plus numpy and
scipy.

```bash
git clone https://github.com/themacadocious/cad-transmogrifier.git
cd cad-transmogrifier
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./transmogrify part.stl
```

## Usage

    ./transmogrify part.stl                 # -> part.step next to it
    ./transmogrify case.3mf                 # 3MF works too (PrusaSlicer, Bambu Studio, Orca)
    ./transmogrify *.stl                    # batch
    ./transmogrify part.stl -o part.iges    # .step/.stp, .iges/.igs or .brep
    ./transmogrify                          # small GUI with a file picker

Then in FreeCAD: *File → Import* the `.step`; in Onshape: *Import* it into a Part Studio.

Options:
- `--faceted` — skip surface recognition; only merge coplanar triangles (safest)
- `--fit-tol MM` — how far a vertex may sit from a plane/cylinder and still belong to it
  (default: 2e-5 × bounding-box diagonal, at least 0.001 mm)
- `--angle DEG` — max angle between triangles merged into one flat face (default 0.5)
- `--min-cyl-facets N` — full-turn shapes with fewer than N facets stay flat-sided prisms
  (default: decided from the mesh; 0 = always cylinders)
- `--no-repair` — skip mesh repair
- `--fix-self-intersections` — cut out and re-fill small self-intersecting areas
  (default: only warn)
- `--no-merge` — one face per triangle
- `--weld MM` — vertex weld tolerance

Units: STL has none — it's treated as millimetres, like slicers do. 3MF units are honoured.

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
- **Speed on big freeform meshes**: ~15 s for a 30k-triangle mostly-organic part and
  ~1 min for an 80k-triangle one (segmentation tries cylinders on spline extrusions
  that look locally cylindrical; OCCT then builds and writes tens of thousands of facet
  faces). Prismatic/CAD-like parts are fast (corpus: 3 s for 22 files).
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

## License

LGPL-2.1, the same family as the OpenCASCADE kernel it is built on. See `LICENSE`.
