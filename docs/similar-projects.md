# Similar projects on the same files

This compares how three independent projects that turn triangle meshes into STEP solids handle the same eight files. It is **not** a contest. The two other projects have had far more of their authors' own time put into them than cad-transmogrifier, which was vibe-coded with Claude over two days, and they aim at different things (see "The projects" below). Every output was measured in exactly the same way. Where a tool did something unexpected, the table says so plainly, and the notes say whether it could be our own usage.

| Project | Version used | What it is |
|---|---|---|
| **cad-transmogrifier** (this repo, formerly "stl2step") | the commit that added this page (Python 3.13, cadquery-ocp 8.0.1.1) | Python CLI/GUI on OpenCASCADE (OCP) |
| **BlinkingSun/stl2step** | release **v1.4.4**, `stl2step-engine-v1.4.4-web-wasm.tar.gz` (sha256 `2afd6a3d…c37c357d`, matches `SHA256SUMS-engine.txt` and `engine-manifest.json`; engine commit `d3f66a4`, OCCT 7.9.3, Emscripten 6.0.5). Run under Node v26.9.0 | C++ engine + CLI on OCCT, plus the SolidOut desktop app and a browser version. MIT |
| **The-Dorkknight/stl2step** ("STL2STEP") | `main` @ `23e917e` (2026-10-11, "Update to 0.7.1-wip"). Its own portable runtime: Python 3.12.15, cadquery-ocp 8.0.1.1, trimesh 5.1.1 | Portable Python app (browser UI + CLI) on OCP/trimesh. PolyForm Noncommercial (run here only for this noncommercial comparison) |

Machine: AMD Ryzen AI 5 340 (12 threads), 46 GB RAM, Debian 13. All runs were made one after another, never in parallel.

## How each tool was run

- **cad-transmogrifier**: `transmogrify <input> -o out.step` (identical to `.venv/bin/python stl2step.py`), with default settings. Its default already runs the prismatic/analytic rebuild, so there is no separate mode to try. The 3MF was read natively.
- **BlinkingSun**: the release ships only Windows, macOS and WebAssembly engines, so the Linux runs used the **web-wasm** build under Node (no source build). The module exports an embind `convert(Uint8Array stl, string optionsJson)` that returns `{ok, stepBytes, resultJson, log}`. `resultJson` is the same `RESULT` object the CLI documents. A 15-line Node driver calls it once per process. Options were `{}` for the **default Verbatim** mode and `{"engine":"trueform"}` for **TrueForm**, which the README describes as the analytic mode for prismatic parts, so both were run. The RESULT confirmed that TrueForm ran: the `smooth*` keys appeared. An unknown option value silently falls back to Verbatim, so this check mattered. The engine is STL-only, so for the 3MF it got `orin_case_from3mf.stl`: a binary STL written from cad-transmogrifier's `read_mesh()` of the 3MF (same 3304 triangles; volume matches the 3MF to 4 mm³ in 76,690, which is float32 rounding).
- **Dorkknight**: `./stl2step-linux.sh <input> out.step`. The first start downloaded uv, Python 3.12 and the libraries into its own `runtime/` folder (1.7 GB). It ran with defaults (Smooth surfaces mode, Classic fitter, auto profile). The README describes `--engine ladder` as the engine that routes parts as prismatic, lathe, organic or mixed and claims exact shapes first, so that was run too. The README also mentions `--engine staged` for "turned, round or prismatic parts"; it was not run here.

## How every output was evaluated (identically)

A small evaluation script reads every output STEP with OCP's `STEPControl_Reader` and measures:

- the number of solids and shells;
- `BRepCheck_Analyzer.IsValid()`;
- whether it is closed: every shell `BRep_Tool.IsClosed` and zero free edges from `ShapeAnalysis_Shell`;
- the volume (`BRepGProp`) and its % difference against the **input mesh's** signed volume, plus against the **ground-truth B-rep volume** for the two corpus parts;
- the number of unique faces and the faces by surface type (P plane / C cylinder / K cone / S sphere / T torus / B B-spline / O other);
- the distinct cylinder radii, rounded to 0.01 mm, with counts;
- for corpus parts, how many true cylinders were matched by axis and radius (1 % tolerance).

**Wall s** is the elapsed time from launch to exit, including interpreter, Node or wasm start-up. For BlinkingSun, the number in parentheses is the time inside `convert()`, without the roughly 1–3 s it takes to load the wasm module.

## Caveats

1. **BlinkingSun times are not representative of its native engine.** The wasm build is single-threaded (`"threads": 1` in the manifest; the log says `(1 threads)`). The native CLI builds B-reps in parallel on all cores. TrueForm in particular was slow here: 160 s on the fine plate and 496 s on the coarse rounded block. Native Windows/macOS builds were not tested.
2. With **no CAD original**, the "Vol Δ vs mesh" on the six real-world files is not an error measure. Turning a faceted hole or fillet back into a true cylinder *should* change the volume a little: holes get bigger, convex rounds fuller. A faceted (Verbatim-style) output that exactly matches the mesh is no more "correct" than an analytic one that is +0.1 % off. Only the two corpus parts have a true answer.
3. **Dorkknight's README marks the project "WORK IN PROGRESS: MANY FEATURES MAY BE BROKEN" (0.7.1-wip).** We ran the head of `main` on the day it was committed, so the failures below may already be known or short-lived.
4. Each configuration was run once, so the timings are single samples.
5. Surface types are classified with the same helper our own test suite uses (`tests/corpus_common.py`), applied identically to every tool's output.

## Results

### Orin_nano_case_main.3MF

Input: 3304 triangles, bbox 114 × 103 × 34.77 mm, mesh volume 76,690.62 mm³.

| Tool / mode | Result | Wall s | STEP | Solids | Valid | Closed | Vol Δ vs mesh | Faces | P / C / K / S / T / B / O | Cylinder radii (mm, ×count) |
|---|---|---|---|---|---|---|---|---|---|---|
| cad-transmogrifier (default) | ok | 2.4 | 1.60 MB | 1 | yes | yes | -0.022% | 458 | 334 / 47 / 77 / 0 / 0 / 0 / 0 | 1.25×36, 2.25×4, 2.6×4, 3×3 |
| BlinkingSun Verbatim (default) on STL made from the 3MF | ok | 7.2 (4.6) | 3.56 MB | 1 | yes | yes | +0.000% | 1072 | 1072 / 0 / 0 / 0 / 0 / 0 / 0 | – |
| BlinkingSun TrueForm (`engine: trueform`) on STL made from the 3MF | ok | 148.5 (145.1) | 3.24 MB | 1 | yes | yes | +0.643% | 824 | 668 / 156 / 0 / 0 / 0 / 0 / 0 | 0.59×7, 0.6×80, 1.25×36, 1.47×2, 1.49×2, 1.5×17, 2.25×4, 2.6×4, +2 more |
| Dorkknight Classic (default) on .3MF | **failed**: `ModuleNotFoundError: No module named 'lxml'` | 1.4 | – | – | – | – | – | – | – | – |
| Dorkknight Classic (default) on STL made from the 3MF | ok | 15.5 | 2.58 MB | 1 | yes | yes | +0.024% | 530 | 385 / 59 / 0 / 74 / 0 / 12 / 0 | 1×2, 1.25×46, 2.25×3, 3×2, 3.51, 4, 4.07, 30.19×3 |
| Dorkknight Ladder (`--engine ladder`) on .3MF | **failed**: `ModuleNotFoundError: No module named 'lxml'` | 1.2 | – | – | – | – | – | – | – | – |
| Dorkknight Ladder (`--engine ladder`) on STL made from the 3MF | ok | 12.1 | 3.58 MB | 1 | yes | yes | +0.000% | 1072 | 1072 / 0 / 0 / 0 / 0 / 0 / 0 | – |

Notes: all three tools produced valid, closed single solids, but recovered different features. 
- **BlinkingSun TrueForm** recovered the most cylinders: 156, including 87 small **r 0.6 mm fillet strips**. Neither of the other tools turned those into cylinders. cad-transmogrifier and Dorkknight left them as planar facets. TrueForm also took 145 s single-threaded in wasm and gave the largest volume change, +0.64 %. That change matches the engine's own predicted change from its fits (`smoothVolPredictedMM3` = +590 mm³), so it is intended. Without the CAD original we can't tell which volume is closer to the true part. It also reported 10 `IntAna cyl|cyl … keeping mesh polyline` warnings.
- The same 72 small features around the r 1.25 mm holes came out as **45° cones** in cad-transmogrifier and as **r 2.23 mm spheres** in Dorkknight Classic. A 45° countersink or chamfer seems the likelier design intent, but that is our reading, not something we checked against the CAD.
- **Dorkknight could not read the .3MF** in this checkout. trimesh's 3MF loader needs `lxml`, and `app/requirements.txt` does not list it, so the portable runtime doesn't have it. That is a packaging gap, not a converter limitation. We didn't change its runtime; we gave it the same STL that BlinkingSun got, and it converted normally. With `--engine ladder` the part was routed "prismatic" and came out fully faceted: 1072 planes, exactly the mesh volume.

### Antenna_bracket.stl

Input: 704 triangles, bbox 58.38 × 5.99 × 13.19 mm, mesh volume 924.76 mm³.

| Tool / mode | Result | Wall s | STEP | Solids | Valid | Closed | Vol Δ vs mesh | Faces | P / C / K / S / T / B / O | Cylinder radii (mm, ×count) |
|---|---|---|---|---|---|---|---|---|---|---|
| cad-transmogrifier (default) | ok | 1.0 | 0.15 MB | 1 | yes | yes | +0.708% | 26 | 18 / 8 / 0 / 0 / 0 / 0 / 0 | 1×2, 1.5×2, 2×2, 3×2 |
| BlinkingSun Verbatim (default) | ok | 4.0 (1.2) | 0.80 MB | 1 | yes | yes | +0.000% | 238 | 238 / 0 / 0 / 0 / 0 / 0 / 0 | – |
| BlinkingSun TrueForm (`engine: trueform`) | ok | 4.7 (2.0) | 0.80 MB | 1 | yes | yes | +0.000% | 238 | 238 / 0 / 0 / 0 / 0 / 0 / 0 | – |
| Dorkknight Classic (default) | ok | 2.2 | 0.43 MB | 1 | yes | yes | +0.188% | 80 | 66 / 12 / 0 / 0 / 0 / 2 / 0 | 1×2, 1.5×4, 2×2, 3.01×4 |
| Dorkknight Ladder (`--engine ladder`) | ok | 2.0 | 0.25 MB | 1 | yes | yes | +0.587% | 30 | 20 / 8 / 0 / 0 / 0 / 2 / 0 | 1×2, 1.5×2, 2×2, 3×2 |

Notes: cad-transmogrifier (26 faces) and Dorkknight Ladder (30 faces) both recovered the 8 hole/boss cylinders with clean radii (1, 1.5, 2, 3 mm). Dorkknight Classic split some of them (80 faces, r 3.01 ×4). BlinkingSun TrueForm reverted this part to its faceted result (`smooth: analytic rebuild reverted on one component -- kept faceted`), which is what its documented fallback does. **cad-transmogrifier changed the volume the most (+0.71 %, vs +0.59 % / +0.19 % for Dorkknight).** With no CAD original that may or may not be closer to the truth, but it is the largest change from the mesh, so it is worth a look.

### AMS Corner Dryer Lid.stl

Input: 6140 triangles, bbox 23.4 × 30.1 × 4.5 mm, mesh volume 953.11 mm³.

| Tool / mode | Result | Wall s | STEP | Solids | Valid | Closed | Vol Δ vs mesh | Faces | P / C / K / S / T / B / O | Cylinder radii (mm, ×count) |
|---|---|---|---|---|---|---|---|---|---|---|
| cad-transmogrifier (default) | ok | 2.0 | 1.19 MB | 1 | yes | yes | +0.000% | 355 | 309 / 36 / 2 / 0 / 8 / 0 / 0 | 0.5×24, 0.7×4, 3.1×4, 4.55×4 |
| BlinkingSun Verbatim (default) | ok | 15.5 (12.3) | 10.34 MB | 1 | yes | yes | -0.000% | 3900 | 3900 / 0 / 0 / 0 / 0 / 0 / 0 | – |
| BlinkingSun TrueForm (`engine: trueform`) | ok | 101.7 (98.8) | 10.34 MB | 1 | yes | yes | -0.000% | 3900 | 3900 / 0 / 0 / 0 / 0 / 0 / 0 | – |
| Dorkknight Classic (default) | ok | 14.7 | 2.95 MB | 1 | yes | yes | +0.013% | 690 | 634 / 33 / 0 / 22 / 0 / 1 / 0 | 0.46, 0.5×20, 0.51×2, 0.7×2, 3.1×2, 4.56×4, 4.94, 4.95 |
| Dorkknight Ladder (`--engine ladder`) | ok | 13.5 | 2.95 MB | 1 | yes | yes | +0.013% | 690 | 634 / 33 / 0 / 22 / 0 / 1 / 0 | 0.46, 0.5×20, 0.51×2, 0.7×2, 3.1×2, 4.56×4, 4.94, 4.95 |

Notes: cad-transmogrifier gave the most compact solid: 355 faces, with cylinders, 2 cones and 8 tori. Dorkknight gave 690 faces, including 22 spheres and a B-spline patch. Its own report says "max deviation 0.66 mm" from the mesh at one spot (mean 0.009 mm); cad-transmogrifier doesn't print a deviation figure, so there's no like-for-like number. Dorkknight's Ladder routed the part "prismatic" and gave a result identical to Classic. BlinkingSun TrueForm reverted to faceted (3900 planar faces) after 99 s.

### fit_ring.stl

Input: 440 triangles, bbox 62.8 × 62.8 × 4 mm, mesh volume 1,819.47 mm³.

| Tool / mode | Result | Wall s | STEP | Solids | Valid | Closed | Vol Δ vs mesh | Faces | P / C / K / S / T / B / O | Cylinder radii (mm, ×count) |
|---|---|---|---|---|---|---|---|---|---|---|
| cad-transmogrifier (default) | ok | 0.9 | 0.01 MB | 1 | yes | yes | +0.118% | 4 | 2 / 2 / 0 / 0 / 0 / 0 / 0 | 29, 31.4 |
| BlinkingSun Verbatim (default) | ok | 3.6 (0.9) | 0.42 MB | 1 | yes | yes | -0.000% | 112 | 112 / 0 / 0 / 0 / 0 / 0 / 0 | – |
| BlinkingSun TrueForm (`engine: trueform`) | ok | 2.7 (0.3) | 0.01 MB | 1 | yes | yes | +0.118% | 4 | 2 / 2 / 0 / 0 / 0 / 0 / 0 | 29, 31.4 |
| Dorkknight Classic (default) | ok | 1.9 | 0.44 MB | 1 | yes | yes | -0.000% | 112 | 112 / 0 / 0 / 0 / 0 / 0 / 0 | – |
| Dorkknight Ladder (`--engine ladder`) | ok | 1.9 | 0.44 MB | 1 | yes | yes | -0.000% | 112 | 112 / 0 / 0 / 0 / 0 / 0 / 0 | – |

Notes: a plain ring with 55 facets around. **cad-transmogrifier and BlinkingSun TrueForm produced the identical 4-face ideal ring** (2 planes, cylinders r 29 and r 31.4) with identical volumes. Dorkknight treated every wall facet as a flat region (112 planes) in Classic, in Ladder (routed "lathe"), and with `--plane-rule length`. Its `--tol`, `--plane-angle` and `--crease-angle` options might change that; we didn't tune them, since this comparison uses default settings.

### Battery Cage Test.stl

Input: 2776 triangles, bbox 77.2 × 108.69 × 77 mm, mesh volume 33,937.87 mm³, **376 non-manifold edges**.

| Tool / mode | Result | Wall s | STEP | Solids | Valid | Closed | Vol Δ vs mesh | Faces | P / C / K / S / T / B / O | Cylinder radii (mm, ×count) |
|---|---|---|---|---|---|---|---|---|---|---|
| cad-transmogrifier (default) | ok | 1.4 | 0.81 MB | 1 | yes | yes | -0.005% | 205 | 193 / 12 / 0 / 0 / 0 / 0 / 0 | 2.6×4, 10×4, 12×4 |
| BlinkingSun Verbatim (default) | ok, but 251 open shells, no solid | 8.6 (5.8) | 4.35 MB | 0 | yes | **no** (1504 free edges) | (n/a: open shells) | 1202 | 1202 / 0 / 0 / 0 / 0 / 0 / 0 | – |
| BlinkingSun TrueForm (`engine: trueform`) | ok, but 251 open shells, no solid | 8.9 (5.9) | 4.35 MB | 0 | yes | **no** (1504 free edges) | (n/a: open shells) | 1202 | 1202 / 0 / 0 / 0 / 0 / 0 / 0 | – |
| Dorkknight Classic (default) | **failed** (no STEP): `OCP.OCP.Standard.Standard_TypeMismatch: TopoDS::Solid` | 19.0 | – | – | – | – | – | – | – | – |
| Dorkknight Ladder (`--engine ladder`) | **failed** (no STEP): `OCP.OCP.Standard.Standard_TypeMismatch: TopoDS::Solid` | 17.1 | – | – | – | – | – | – | – | – |
| Dorkknight Exact facets (`--mode faceted`) | **failed** (no STEP): `OCP.OCP.Standard.Standard_TypeMismatch: TopoDS::Solid` | 5.1 | – | – | – | – | – | – | – | – |

Notes: the mesh has 376 non-manifold edges (bodies touching face to face).
- **cad-transmogrifier** was the only tool here that produced a closed solid. Its repair step removed 124 coincident back-to-back patch pairs and fused the bodies; the result was 1 valid solid with 12 cylinders and −0.005 % volume.
- **BlinkingSun** did what its README promises for this case ("what still can't close is written as an open shell and flagged, never dropped silently"): 251 open shells, each with a `component is not a closed volume (repaired) -- exported as an open shell` warning, and `"watertight": false` in the RESULT. Its CLI would exit with code 2.
- **Dorkknight** warned "mesh is not watertight — the result will be a surface" and tried its fallbacks. In all three modes we tried (Classic, Ladder, `--mode faceted`) it then stopped with `Standard_TypeMismatch: TopoDS::Solid` in `_shells_to_solid`, and no STEP was written. That looks like an unhandled case in the faceted fallback, not a misuse. Its `--decimate` / repair-related options were not tried.

### screw_1.stl

Input: 34098 triangles, bbox 15 × 15 × 26.01 mm, mesh volume 1,228.55 mm³.

| Tool / mode | Result | Wall s | STEP | Solids | Valid | Closed | Vol Δ vs mesh | Faces | P / C / K / S / T / B / O | Cylinder radii (mm, ×count) |
|---|---|---|---|---|---|---|---|---|---|---|
| cad-transmogrifier (default) | ok | 15.9 | 36.95 MB | 1 | yes | yes | +0.039% | 12910 | 12906 / 4 / 0 / 0 / 0 / 0 / 0 | 4×2, 7.5×2 |
| BlinkingSun Verbatim (default) | ok | 91.6 (88.5) | 80.74 MB | 1 | yes | yes | +0.000% | 32620 | 32620 / 0 / 0 / 0 / 0 / 0 / 0 | – |
| BlinkingSun TrueForm (`engine: trueform`) | ok | 89.3 (86.4) | 80.74 MB | 1 | yes | yes | +0.000% | 32620 | 32620 / 0 / 0 / 0 / 0 / 0 / 0 | – |
| Dorkknight Classic (default) | ok | 387.7 | 38.91 MB | 1 | yes | yes | +0.054% | 14925 | 14838 / 32 / 0 / 0 / 0 / 55 / 0 | 1.01×6, 3.86×22, 4×2, 7.5×2 |
| Dorkknight Ladder (`--engine ladder`) | ok | 501.1 | 35.29 MB | 1 | yes | yes | +0.055% | 13482 | 13394 / 34 / 0 / 0 / 0 / 50 / 4 | 1.01×6, 3.22×2, 3.86×22, 4×2, 7.5×2 |

Notes: none of the tools turns a thread into analytic geometry, and none claims to. The results differ mainly in face count and time. cad-transmogrifier was the fastest at 16 s, with the fewest faces: 12,910, of which 4 are cylinders. Its repair step reported 11 self-intersecting triangle pairs that it left in place; the output still passes BRepCheck. BlinkingSun kept all 32,620 facets in both modes in about 90 s (single-threaded wasm). TrueForm reverted to faceted. **Dorkknight recovered noticeably more real geometry on this part**: 32–34 cylinders (r 3.86 ×22, r 1.01 ×6, …) and 50–55 B-spline patches. It took 6.5 min (Classic) and 8.4 min (Ladder, routed "lathe").

### plate_holes_cbore_fine.stl (corpus)

Input: 3056 triangles, bbox 80 × 60 × 8 mm, mesh volume 36,490.70 mm³. Ground truth: volume 36,489.91 mm³, 13 faces (7 plane, 6 cylinder), 80x60x8 plate, 4x Ø6 holes, Ø8 hole with Ø16x4 counterbore.

| Tool / mode | Result | Wall s | STEP | Solids | Valid | Closed | Vol Δ vs mesh | Vol Δ vs truth | Cyl matched | Faces | P / C / K / S / T / B / O | Cylinder radii (mm, ×count) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cad-transmogrifier (default) | ok | 1.0 | 0.04 MB | 1 | yes | yes | -0.002% | -0.0000% | 6/6 | 13 | 7 / 6 / 0 / 0 / 0 / 0 / 0 | 3×4, 4, 8 |
| BlinkingSun Verbatim (default) | ok | 7.6 (4.3) | 3.04 MB | 1 | yes | yes | -0.000% | +0.0022% | 0/6 | 763 | 763 / 0 / 0 / 0 / 0 / 0 / 0 | – |
| BlinkingSun TrueForm (`engine: trueform`) | ok | 162.5 (159.6) | 0.04 MB | 1 | yes | yes | -0.002% | -0.0000% | 6/6 | 13 | 7 / 6 / 0 / 0 / 0 / 0 / 0 | 3×4, 4, 8 |
| Dorkknight Classic (default) | ok | 1.8 | 0.12 MB | 1 | yes | yes | -0.002% | -0.0000% | 6/6 | 13 | 7 / 6 / 0 / 0 / 0 / 0 / 0 | 3×4, 4, 8 |
| Dorkknight Ladder (`--engine ladder`) | ok | 1.9 | 0.12 MB | 1 | yes | yes | -0.002% | -0.0000% | 6/6 | 13 | 7 / 6 / 0 / 0 / 0 / 0 / 0 | 3×4, 4, 8 |

Notes: **cad-transmogrifier, BlinkingSun TrueForm and Dorkknight (both engines) all rebuilt the exact ground-truth model**: 13 faces, all 6 cylinders matched, volume within 1e-6 % of the true B-rep. That is better than the mesh itself, which is +0.002 % off. BlinkingSun Verbatim is exact to the mesh by design (763 planar faces). TrueForm took 160 s in single-threaded wasm, against about 1 s for the others. That is caveat 1.

### rounded_block_coarse.stl (corpus)

Input: 780 triangles, bbox 50 × 40 × 20 mm, mesh volume 38,532.25 mm³. Ground truth: volume 38,574.91 mm³, 26 faces (6 plane, 12 cylinder, 8 sphere), 50x40x20 block, all edges R4 (cylinders + corner blends).

| Tool / mode | Result | Wall s | STEP | Solids | Valid | Closed | Vol Δ vs mesh | Vol Δ vs truth | Cyl matched | Faces | P / C / K / S / T / B / O | Cylinder radii (mm, ×count) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cad-transmogrifier (default) | ok | 1.1 | 0.10 MB | 1 | yes | yes | +0.111% | +0.0000% | 12/12 | 26 | 6 / 12 / 0 / 8 / 0 / 0 / 0 | 4×12 |
| BlinkingSun Verbatim (default) | ok | 4.5 (1.9) | 1.59 MB | 1 | yes | yes | +0.000% | -0.1106% | 0/12 | 682 | 682 / 0 / 0 / 0 / 0 / 0 / 0 | – |
| BlinkingSun TrueForm (`engine: trueform`) | ok | 498.7 (496.3) | 0.66 MB | 1 | yes | yes | +6.865% | +6.7471% | 0/12 | 132 | 132 / 0 / 0 / 0 / 0 / 0 / 0 | – |
| Dorkknight Classic (default) | ok | 4.1 | 1.34 MB | 1 | yes | yes | +0.021% | -0.0892% | 6/12 | 539 | 531 / 7 / 0 / 1 / 0 / 0 / 0 | 4×5, 4.01, 4.02 |
| Dorkknight Ladder (`--engine ladder`) | ok | 5.9 | 1.59 MB | 1 | yes | yes | -0.000% | -0.1106% | 0/12 | 682 | 682 / 0 / 0 / 0 / 0 / 0 / 0 | – |

Notes: **cad-transmogrifier was the only tool to recover the ground truth here**: 26 faces (6 P, 12 C, 8 S), 12/12 cylinders, volume +0.00001 % vs truth. Dorkknight Classic found 7 of the 12 edge rounds as cylinders and 1 sphere corner; the rest stayed faceted. Dorkknight Ladder routed the block "organic" and kept it fully faceted.
**BlinkingSun TrueForm** needs care here. Its README says spheres are out of scope, and the engine did report `smooth: analytic rebuild reverted on one component -- kept faceted`. After the revert, a "smooth-flat" step merged 682 → 132 planar faces, and the written solid is **+6.9 % larger than the mesh** (+6.7 % vs truth), with a bounding box that is still correct. The engine flagged it itself (`B-Rep volume differs from mesh volume by more than 0.01% -- inspect the result`), so its CLI would exit with code 2 rather than 0. We used the wasm build only, so we can't say whether native builds behave the same. BlinkingSun Verbatim's output is exact to the mesh.

## At a glance

| File | cad-transmogrifier | BlinkingSun Verbatim / TrueForm | Dorkknight Classic / Ladder |
|---|---|---|---|
| Orin case (3MF) | 458 faces, 47 C + 77 K | 1072 / **824 (156 C, incl. 0.6 mm fillets)** | 3MF unreadable (lxml); from STL: 530 / 1072 |
| Antenna bracket | 26 faces, 8 C (+0.71 % vol) | 238 / 238 (reverted) | 80 / 30 faces, 8 C |
| AMS lid | **355 faces** (C, K, T) | 3900 / 3900 (reverted) | 690 / 690 (C, S, B) |
| fit_ring | **4 faces (ideal)** | 112 / **4 faces (ideal)** | 112 / 112 |
| Battery cage (non-manifold) | **1 closed solid** | 251 open shells, flagged | failed (exception), no file |
| screw_1 (threads) | 12,910 faces, 16 s | 32,620 / 32,620 | **14,925 / 13,482, most cylinders + B-splines**, 6–8 min |
| Corpus plate + c'bore | **truth (13)** | 763 / **truth (13)** | **truth (13)** / **truth (13)** |
| Corpus rounded block | **truth (26)** | 682 / 132 faces, +6.9 % vol (flagged) | 539 (7 C, 1 S) / 682 |

Bold marks results that match the ground truth, or that are clearly the most complete reconstruction for that file. Everything else in this table is a trade-off, not a ranking.

### Additional try: Dorkknight `--plane-rule length`

The README describes this option as keeping "coarse cylinder facets from passing as flats", so we tried it on four parts. fit_ring was unchanged (112 planes). The rounded block was unchanged from Classic (539 faces). The antenna bracket gave the same result as Ladder (30 faces, 8 C). The Orin case went to 454 faces (309 P, 59 C, 74 S, 12 B) at −0.10 % volume. 

## The projects

**BlinkingSun/stl2step** is a carefully engineered conversion *engine* meant to be embedded. It has a one-header C++ API, a CLI with a stable `RESULT {json}` contract and meaningful exit codes, an `AGENTS.md` integration guide, signed desktop apps (SolidOut), and a WebAssembly build that runs the same engine in a browser with nothing uploaded. Its default **Verbatim** mode is deliberately conservative. It is byte-faithful to the mesh, and here it gave a valid solid with 0.000 % volume change on every closed input, re-read and verified its own output, and handled the broken battery cage exactly as documented (open shells, every one flagged, nothing silently dropped). **TrueForm** is its analytic mode for prismatic parts: planes, right cylinders and fillet strips. It reproduced the ground-truth plate and the ideal ring exactly, and on the Orin case it recovered small 0.6 mm fillets that neither other tool turned into cylinders. By its own documentation it leaves cones, spheres and tori faceted, and it falls back per body when it can't close analytically. The one surprising output was the rounded block (+6.9 % volume), and the engine flagged that one itself. All timings here come from the single-threaded wasm build, not its native multi-threaded engine.

**The-Dorkknight/STL2STEP** is a much broader toolbox than a mesh converter. Besides STL/OBJ/PLY/3MF → STEP, it has OpenSCAD → STEP, image/photo → STEP outline tracing with a printable scale token, "hardpoint" STLs that carry CAD face labels, a STEP clean-up tool (Jetwash), FreeCAD and Fusion plug-ins, a browser UI with queue and scheduling, and a portable runtime that installs nothing system-wide. Its mesh converter is the only one of the three that **fits freeform NURBS patches** to organic regions. It produced B-spline faces on the Orin case, the AMS lid and the screw, and recovered more real cylinders on the threaded screw than the other two. It offers several engines (Classic, Staged, Ladder, v2, Best) and outline-based modes (Extrude, Loft) for plates and tapered parts. Every successful run here was a valid closed solid with a clear self-report (faces, closed/valid, volume change, mean/max deviation, route taken). It is openly marked work-in-progress. In this checkout it couldn't open the 3MF (missing `lxml` in its runtime) and stopped with an exception on the non-manifold battery cage. It was also the slowest on the dense screw mesh, which fits its focus on quality fitting over speed.

**cad-transmogrifier** (this repo) is a small Python tool, written in two days with Claude, that focuses on recognising **prismatic CAD-style geometry**: planes, cylinders, cones, spheres and tori rebuilt as analytic faces, with mesh repair first. On these files it was the fastest, the only tool to recover the full ground truth of the rounded block (cylinders and sphere corners), and the only one to turn the non-manifold battery cage into a closed solid. Its gaps in this test: it does no freeform/NURBS fitting at all, so organic regions (thread flanks, sculpted areas) stay as planar facets. It did not recover the 0.6 mm fillets on the Orin case that BlinkingSun TrueForm found. On the antenna bracket it changed the volume the most (+0.71 %). It doesn't report a mesh-to-solid deviation figure the way Dorkknight does, and its repair left 11 self-intersecting triangle pairs in the screw as they were. It hasn't been tested beyond its own corpus and a handful of real parts.

## Method notes

Inputs were the author's own downloads (a Jetson Orin Nano case, a bracket, a dryer lid, a
ring, a broken battery cage, a threaded screw) plus two parts from this repo's ground-truth
corpus (`tests/corpus/`). Every tool ran with its default settings plus its documented
prismatic mode, once per file, sequentially, on the machine listed above, and every output
was evaluated by the same script. If you maintain one of these projects and think a result
here is wrong or unfair, please open an issue — happy to correct it.
