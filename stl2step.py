#!/usr/bin/env python3
"""stl2step — turn an STL mesh into a real B-rep solid that CAD tools can edit.

Pipeline (same idea as Fusion 360's "Mesh to BRep"):
  1. parse STL, weld duplicate vertices, drop degenerate triangles
  2. build an OpenCASCADE shell with *shared* edges (no slow sewing pass)
  3. close it into a solid and fix its orientation
  4. merge coplanar triangles into single planar faces (a cube becomes 6 faces, not 12)
  5. write STEP (or IGES / BREP)

By default it first tries a "prismatic" rebuild (segment.py + rebuild.py): planes,
cylinders, cones, spheres and tori are recognised and rebuilt as true analytic faces, so
a hole is one cylinder you can resize. If that fails validation it falls back to 1-4.
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeWire,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.IGESControl import IGESControl_Writer
from OCP.Interface import Interface_Static
from OCP.Message import Message
from OCP.ShapeFix import ShapeFix_Shape, ShapeFix_Shell, ShapeFix_Solid
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shell
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt


# OCCT's STEP writer prints transfer statistics to stdout; keep the console clean
_printers = Message.DefaultMessenger_s().Printers()
while not _printers.IsEmpty():
    Message.DefaultMessenger_s().RemovePrinter(_printers.First())
    _printers = Message.DefaultMessenger_s().Printers()


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- STL parsing

def read_stl(path):
    """Return an (N, 3, 3) float64 array of triangle vertices."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        (count,) = struct.unpack_from("<I", data, 80)
        if 84 + count * 50 == len(data):
            rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
            return np.frombuffer(data, rec, count, 84)["v"].astype(np.float64)
    # ASCII fallback (also catches binary files whose header lies about count)
    verts = [
        line.split()[1:4]
        for line in data.decode("ascii", "replace").splitlines()
        if line.strip().startswith("vertex")
    ]
    if not verts or len(verts) % 3:
        raise ValueError(f"{path}: not a readable STL")
    return np.array(verts, dtype=np.float64).reshape(-1, 3, 3)


def weld(tris, tol):
    """Merge vertices closer than ~tol. Returns (points, faces) with faces as indices."""
    flat = tris.reshape(-1, 3)
    keys = np.round(flat / tol).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    points = flat[first]
    faces = inverse.reshape(-1, 3)

    # degenerate: repeated index, or zero area after welding
    ok = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    p = points[faces]
    area2 = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    ok &= area2 > tol * tol
    # exact duplicate triangles (common in bad exports) break manifoldness
    canon = np.sort(faces, axis=1)
    _, uniq = np.unique(canon, axis=0, return_index=True)
    dup = np.ones(len(faces), bool)
    dup[uniq] = False
    ok &= ~dup
    return points, faces[ok], int((~ok).sum())


# ---------------------------------------------------------------- B-rep build

def build_shell(points, faces):
    """Build a shell where neighbouring triangles share the same TopoDS_Edge."""
    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)

    vtx = [BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, p))).Vertex() for p in points]
    edges = {}

    def edge(a, b):
        key = (a, b) if a < b else (b, a)
        e = edges.get(key)
        if e is None:
            e = BRepBuilderAPI_MakeEdge(vtx[key[0]], vtx[key[1]]).Edge()
            edges[key] = e
        return e if a < b else TopoDS.Edge(e.Reversed())

    p = points[faces]
    normals = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    failed = 0
    for (a, b, c), n, p0 in zip(faces.tolist(), normals, p[:, 0]):
        wire = BRepBuilderAPI_MakeWire(edge(a, b), edge(b, c), edge(c, a))
        if not wire.IsDone():
            failed += 1
            continue
        plane = gp_Pln(gp_Pnt(*map(float, p0)), gp_Dir(*map(float, n)))
        face = BRepBuilderAPI_MakeFace(plane, wire.Wire(), True)
        if not face.IsDone():
            failed += 1
            continue
        builder.Add(shell, face.Face())
    return shell, failed


def make_solid(shell):
    """Fix shell orientation and close it into a solid. Falls back to the shell if open."""
    fix = ShapeFix_Shell(shell)
    fix.SetNonManifoldFlag(False)
    fix.Perform()
    shape = fix.Shape()

    shells = []
    exp = TopExp_Explorer(shape, TopAbs_SHELL)
    while exp.More():
        shells.append(TopoDS.Shell(exp.Current()))
        exp.Next()
    if not shells or not all(BRep_Tool.IsClosed_s(s) for s in shells):
        return shape, False

    sf = ShapeFix_Solid()
    sf.LimitTolerance(1e-3)
    # multiple closed shells (e.g. several bodies in one STL) -> compound of solids
    builder = BRep_Builder()
    if len(shells) == 1:
        return sf.SolidFromShell(shells[0]), True
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    for s in shells:
        builder.Add(comp, sf.SolidFromShell(s))
    return comp, True


def merge_coplanar(shape, angle_deg, lin_tol, unify_edges=True):
    u = ShapeUpgrade_UnifySameDomain(shape, unify_edges, True, False)
    u.SetAngularTolerance(np.radians(angle_deg))
    u.SetLinearTolerance(lin_tol)
    u.AllowInternalEdges(False)
    u.Build()
    # merged planes are a best fit, so some vertices sit just off them; let ShapeFix
    # grow tolerances / rebuild pcurves so BRepCheck (and other CAD kernels) accept it
    fix = ShapeFix_Shape(u.Shape())
    fix.Perform()
    return fix.Shape()


def count_faces(shape):
    n, exp = 0, TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        n += 1
        exp.Next()
    return n


def volume(shape):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


# ---------------------------------------------------------------- output

def write(shape, out, name):
    ext = out.suffix.lower()
    if ext in (".step", ".stp"):
        Interface_Static.SetCVal_s("write.step.schema", "AP214IS")
        Interface_Static.SetCVal_s("write.step.unit", "MM")
        Interface_Static.SetCVal_s("write.step.product.name", name)
        w = STEPControl_Writer()
        w.Transfer(shape, STEPControl_AsIs)
        ok = w.Write(str(out)) == IFSelect_RetDone
    elif ext in (".iges", ".igs"):
        w = IGESControl_Writer("MM", 1)  # 1 = B-rep mode (solids, not trimmed surfaces)
        w.AddShape(shape)
        w.ComputeModel()
        ok = w.Write(str(out))
    elif ext == ".brep":
        ok = BRepTools.Write_s(shape, str(out))
    else:
        raise ValueError(f"unsupported output type {ext} (use .step, .iges or .brep)")
    if not ok:
        raise RuntimeError(f"failed to write {out}")


def convert(src, out=None, merge=True, angle=0.5, weld_tol=None, progress=log,
            prismatic=True, fit_tol=None):
    src = Path(src)
    out = Path(out) if out else src.with_suffix(".step")
    t0 = time.time()

    tris = read_stl(src)
    diag = float(np.linalg.norm(np.ptp(tris.reshape(-1, 3), axis=0)))
    tol = weld_tol or max(diag * 1e-7, 1e-6)
    points, faces, dropped = weld(tris, tol)
    progress(f"{src.name}: {len(tris)} triangles, {len(points)} vertices"
             + (f", dropped {dropped} degenerate/duplicate" if dropped else ""))

    shape = None
    if merge and prismatic:
        shape = prismatic_rebuild(points, faces, fit_tol or max(diag * 2e-5, 1e-3), angle, tol, progress)
    if shape is not None:  # rebuild() already proved it a valid closed solid
        return _finish(shape, True, src, out, t0, progress, valid=True)

    shell, failed = build_shell(points, faces)
    if failed:
        progress(f"  warning: {failed} triangles could not be built")
    shape, is_solid = make_solid(shell)
    if not is_solid:
        progress("  warning: mesh is not watertight — exporting an open surface body, not a solid")

    if merge:
        n_before = count_faces(shape)
        shape = merge_coplanar(shape, angle, tol * 10)
        progress(f"  merged coplanar faces: {n_before} -> {count_faces(shape)}")

    return _finish(shape, is_solid, src, out, t0, progress)


def prismatic_rebuild(points, faces, fit_tol, angle, tol, progress):
    """Recognise planes/cylinders and rebuild true analytic faces. None if it can't."""
    try:
        import rebuild
        import segment

        labels, surfaces = segment.segment(points, faces, fit_tol, angle)
        shape = rebuild.rebuild(points, faces, labels, surfaces, fit_tol)
    except Exception as e:  # never worse than faceted: any failure falls back
        progress(f"  prismatic rebuild not possible ({type(e).__name__}: {e}); using faceted conversion")
        return None
    kinds = face_types(shape)
    progress("  prismatic: " + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())) + " faces")
    return shape


def face_types(shape):
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    kinds, exp = {}, TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        k = BRepAdaptor_Surface(TopoDS.Face(exp.Current())).GetType().name.removeprefix("GeomAbs_").lower()
        kinds[k] = kinds.get(k, 0) + 1
        exp.Next()
    return kinds


def _finish(shape, is_solid, src, out, t0, progress, valid=None):
    if is_solid:
        if valid is None:
            valid = BRepCheck_Analyzer(shape).IsValid()
        progress(f"  solid volume {volume(shape):.3f} mm^3, "
                 + ("valid" if valid else "WARNING: BRepCheck reports problems"))

    write(shape, out, src.stem)
    progress(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")
    return out


# ---------------------------------------------------------------- GUI

def gui():
    import threading
    import tkinter as tk
    from tkinter import filedialog, scrolledtext, ttk

    root = tk.Tk()
    root.title("STL → STEP")
    root.geometry("820x420")

    merge = tk.BooleanVar(value=True)
    prismatic = tk.BooleanVar(value=True)
    angle = tk.DoubleVar(value=0.5)
    fmt = tk.StringVar(value=".step")

    bar = ttk.Frame(root, padding=8)
    bar.pack(fill="x")
    btn = ttk.Button(bar, text="Open STL(s)…")
    btn.pack(side="left")
    ttk.Checkbutton(bar, text="Merge coplanar faces", variable=merge).pack(side="left", padx=10)
    ttk.Checkbutton(bar, text="Recognise curved surfaces", variable=prismatic).pack(side="left")
    ttk.Label(bar, text="merge angle°").pack(side="left", padx=(12, 2))
    ttk.Spinbox(bar, from_=0.01, to=10, increment=0.25, width=5, textvariable=angle).pack(side="left")
    ttk.Combobox(bar, values=[".step", ".iges", ".brep"], textvariable=fmt, width=6,
                 state="readonly").pack(side="left", padx=10)

    out = scrolledtext.ScrolledText(root, font=("monospace", 9))
    out.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def say(msg):
        root.after(0, lambda: (out.insert("end", msg + "\n"), out.see("end")))

    def run(paths):
        for p in paths:
            try:
                convert(p, Path(p).with_suffix(fmt.get()), merge.get(), angle.get(), progress=say,
                        prismatic=prismatic.get())
            except Exception as e:  # keep going with the rest of the batch
                say(f"  ERROR: {e}")
        say("done.\n")
        root.after(0, lambda: btn.state(["!disabled"]))

    def pick():
        paths = filedialog.askopenfilenames(filetypes=[("STL mesh", "*.stl *.STL"), ("All", "*")])
        if paths:
            btn.state(["disabled"])
            threading.Thread(target=run, args=(paths,), daemon=True).start()

    btn.configure(command=pick)
    say("Pick one or more STL files; output is written next to each input.")
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="STL files (no args = open the GUI)")
    ap.add_argument("-o", "--output", help="output file (.step/.stp/.iges/.brep); single input only")
    ap.add_argument("--no-merge", action="store_true", help="keep every triangle as its own face")
    ap.add_argument("--angle", type=float, default=0.5,
                    help="max angle in degrees between triangles merged into one face (default 0.5)")
    ap.add_argument("--faceted", action="store_true",
                    help="skip cylinder/plane recognition; only merge coplanar triangles")
    ap.add_argument("--fit-tol", type=float,
                    help="max deviation in mm for a vertex to count as on a plane/cylinder (default: auto)")
    ap.add_argument("--weld", type=float, help="vertex weld tolerance in mm (default: auto)")
    args = ap.parse_args()

    if not args.inputs:
        return gui()
    if args.output and len(args.inputs) > 1:
        ap.error("-o only works with a single input")
    rc = 0
    for src in args.inputs:
        try:
            convert(src, args.output, not args.no_merge, args.angle, args.weld,
                    prismatic=not args.faceted, fit_tol=args.fit_tol)
        except Exception as e:
            log(f"{src}: ERROR: {e}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
