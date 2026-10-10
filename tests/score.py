#!/usr/bin/env python3
"""Score an STL->STEP converter against the ground-truth corpus.

    .venv/bin/python tests/score.py                       # all tests/corpus/*.stl via stl2step.convert
    .venv/bin/python tests/score.py --res coarse          # only the coarse meshes
    .venv/bin/python tests/score.py --impl mymod:convert tests/corpus/tube_fine.stl
    .venv/bin/python tests/score.py --impl /tmp/stl2step_old.py:convert   # score a file by path
    .venv/bin/python tests/score.py some_output.step --truth tests/corpus/tube.json
    .venv/bin/python tests/score.py --json results.json   # machine-readable dump too

The converter is called as fn(src, out) (plus progress=<quiet> if it accepts that keyword)
and must write a STEP file to `out`. Inputs ending in .step/.stp are scored as-is.

Exit status is nonzero if any output is invalid, fails to convert, or has a volume error
above --max-vol-err (default 1%) against the true B-rep volume.
"""

import argparse
import importlib
import importlib.util
import inspect
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_common import (  # noqa: E402
    CORPUS, ROOT, TYPE_NAMES, face_surface, match_cylinders, resolution_of, solids,
    truth_for, unique_faces,
)

from OCP.BRepCheck import BRepCheck_Analyzer  # noqa: E402
from OCP.BRepGProp import BRepGProp  # noqa: E402
from OCP.GProp import GProp_GProps  # noqa: E402
from OCP.IFSelect import IFSelect_RetDone  # noqa: E402
from OCP.Message import Message  # noqa: E402
from OCP.STEPControl import STEPControl_Reader  # noqa: E402

ABBR = {"plane": "P", "cylinder": "C", "cone": "K", "sphere": "S", "torus": "T", "bspline": "B", "other": "O"}


def quiet_occt():
    m = Message.DefaultMessenger_s()
    printers = m.Printers()
    while not printers.IsEmpty():
        m.RemovePrinter(printers.First())
        printers = m.Printers()


def load_impl(spec):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    mod, _, fn = spec.rpartition(":") if spec.count(":") else (spec, "", "")
    if mod.endswith(".py"):  # path/to/file.py:function — e.g. a baseline copy of an old version
        s = importlib.util.spec_from_file_location(Path(mod).stem, mod)
        module = importlib.util.module_from_spec(s)
        sys.modules[s.name] = module
        s.loader.exec_module(module)
    else:
        module = importlib.import_module(mod)
    f = getattr(module, fn or "convert")
    try:
        takes_progress = "progress" in inspect.signature(f).parameters
    except (TypeError, ValueError):
        takes_progress = False

    def run(src, out, log):
        if takes_progress:
            return f(src, out, progress=log)
        return f(src, out)
    return run


def read_step(path):
    r = STEPControl_Reader()
    if r.ReadFile(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEPControl_Reader could not read {path}")
    r.TransferRoots()
    return r.OneShape()


def analyse(shape):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    counts = {t: 0 for t in TYPE_NAMES}
    cyls = []
    faces = unique_faces(shape)
    for f in faces:
        t, p = face_surface(f)
        counts[t] += 1
        if t == "cylinder":
            cyls.append({k: p[k] for k in ("origin", "axis", "radius")})
    return {
        "valid": bool(BRepCheck_Analyzer(shape).IsValid()),
        "solids": len(solids(shape)),
        "volume": abs(props.Mass()),
        "faces": len(faces),
        "faces_by_type": counts,
        "cylinders": cyls,
    }


def score_step(step, truth, runtime=None, src=None):
    shape = read_step(step)
    r = analyse(shape)
    r["file"] = Path(src or step).name
    r["runtime_s"] = runtime
    r["step_bytes"] = Path(step).stat().st_size
    if truth:
        r["truth_faces"] = truth["faces"]
        r["truth_solids"] = truth["solids"]
        r["vol_err_pct"] = (r["volume"] - truth["volume"]) / truth["volume"] * 100
        res = resolution_of(src or step)
        mesh = truth.get("meshes", {}).get(res)
        if mesh:  # error against the STL's own volume: what the converter itself lost
            r["mesh_vol_err_pct"] = (r["volume"] - mesh["mesh_volume"]) / mesh["mesh_volume"] * 100
        r["cyl_hit"], r["cyl_truth"] = match_cylinders(truth["cylinders"], r["cylinders"])
    del r["cylinders"]
    return r


def fmt_types(c):
    return " ".join(f"{ABBR[t]}{n}" for t, n in c.items() if n) or "-"


def print_table(rows):
    hdr = (f"{'file':32} {'ok':>3} {'sol':>4} {'vol%':>8} {'meshvol%':>9} {'faces':>9} "
           f"{'cyl':>6} {'time':>7} {'step':>8}  types (P=plane C=cyl K=cone S=sphere T=torus B=bspline O=other)")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['file']:32} ERR  {r['error']}")
            continue
        sol = f"{r['solids']}" + (f"/{r['truth_solids']}" if "truth_solids" in r else "")
        faces = f"{r['faces']}" + (f"/{r['truth_faces']}" if "truth_faces" in r else "")
        cyl = f"{r['cyl_hit']}/{r['cyl_truth']}" if "cyl_hit" in r else "-"
        vol = f"{r['vol_err_pct']:+.3f}" if "vol_err_pct" in r else "-"
        mvol = f"{r['mesh_vol_err_pct']:+.3f}" if "mesh_vol_err_pct" in r else "-"
        t = f"{r['runtime_s']:.2f}s" if r.get("runtime_s") is not None else "-"
        print(f"{r['file']:32} {'yes' if r['valid'] else 'NO':>3} {sol:>4} {vol:>8} {mvol:>9} "
              f"{faces:>9} {cyl:>6} {t:>7} {r['step_bytes'] / 1024:7.0f}K  {fmt_types(r['faces_by_type'])}")


def failed(r, max_err):
    if "error" in r or not r["valid"]:
        return True
    if "truth_solids" in r and r["solids"] != r["truth_solids"]:
        return True
    return "vol_err_pct" in r and abs(r["vol_err_pct"]) > max_err


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="corpus .stl files or pre-made .step files (default: whole corpus)")
    ap.add_argument("--impl", default="stl2step:convert", help="converter as module:function (default stl2step:convert)")
    ap.add_argument("--res", choices=["coarse", "fine"], help="only score corpus files of this resolution")
    ap.add_argument("--truth", help="ground-truth JSON (default: inferred from the file name)")
    ap.add_argument("--max-vol-err", type=float, default=1.0, help="fail threshold, %% vs true volume")
    ap.add_argument("--keep", help="write STEP outputs to this directory instead of a temp dir")
    ap.add_argument("--json", help="also dump results to this JSON file")
    ap.add_argument("-v", "--verbose", action="store_true", help="show converter progress output")
    args = ap.parse_args(argv)

    files = [Path(f) for f in args.files] or sorted(CORPUS.glob("*.stl"))
    if args.res:
        files = [f for f in files if resolution_of(f) == args.res]
    if not files:
        ap.error("no input files")
    explicit_truth = json.loads(Path(args.truth).read_text()) if args.truth else None

    quiet_occt()
    convert = None
    rows = []
    with tempfile.TemporaryDirectory(prefix="stl2step-score-") as tmp:
        outdir = Path(args.keep) if args.keep else Path(tmp)
        outdir.mkdir(parents=True, exist_ok=True)
        for f in files:
            truth = explicit_truth or truth_for(f)
            try:
                if f.suffix.lower() in (".step", ".stp"):
                    rows.append(score_step(f, truth))
                    continue
                convert = convert or load_impl(args.impl)
                quiet_occt()  # importing the converter may have re-added printers
                out = outdir / (f.stem + ".step")
                log = (lambda m: print(m, file=sys.stderr)) if args.verbose else (lambda m: None)
                t0 = time.perf_counter()
                convert(str(f), str(out), log)
                dt = time.perf_counter() - t0
                rows.append(score_step(out, truth, dt, src=f))
            except Exception as e:  # noqa: BLE001 — one bad file must not stop the run
                if args.verbose:
                    traceback.print_exc()
                rows.append({"file": f.name, "error": f"{type(e).__name__}: {e}"})

    print_table(rows)
    bad = [r["file"] for r in rows if failed(r, args.max_vol_err)]
    ok = [r for r in rows if "cyl_hit" in r]
    if ok:
        hit = sum(r["cyl_hit"] for r in ok)
        tot = sum(r["cyl_truth"] for r in ok)
        print(f"\ncylinders recovered: {hit}/{tot}   "
              f"total time {sum(r.get('runtime_s') or 0 for r in rows):.1f}s   "
              f"failures: {len(bad)}" + (f" ({', '.join(bad)})" if bad else ""))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
