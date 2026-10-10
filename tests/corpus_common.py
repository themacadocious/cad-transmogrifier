"""Helpers shared by make_corpus.py, score.py and test_convert.py.

Kept free of converter imports so the corpus tooling never depends on the code under test.
"""

import json
import struct
from pathlib import Path

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_SurfaceType as ST
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.TopExp import TopExp
# OCP 8: TopTools_IndexedMapOfShape is gone; the NCollection template lives in OCP.collections
from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as TopTools_IndexedMapOfShape
from OCP.TopoDS import TopoDS

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
ROOT = HERE.parent

# surface type codes used in the .labels.npz files and the JSON "faces_by_type" keys
TYPE_NAMES = ["plane", "cylinder", "cone", "sphere", "torus", "bspline", "other"]
TYPE_CODE = {n: i for i, n in enumerate(TYPE_NAMES)}
RESOLUTIONS = ("coarse", "fine")


def surface_type_name(geomabs_type):
    if geomabs_type == ST.GeomAbs_Plane:
        return "plane"
    if geomabs_type == ST.GeomAbs_Cylinder:
        return "cylinder"
    if geomabs_type == ST.GeomAbs_Cone:
        return "cone"
    if geomabs_type == ST.GeomAbs_Sphere:
        return "sphere"
    if geomabs_type == ST.GeomAbs_Torus:
        return "torus"
    if geomabs_type in (ST.GeomAbs_BSplineSurface, ST.GeomAbs_BezierSurface):
        return "bspline"
    return "other"


def unique_faces(shape):
    """Faces of `shape`, each once, in a stable order (TopExp map order)."""
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, m)
    return [TopoDS.Face(m.FindKey(i)) for i in range(1, m.Extent() + 1)]


def solids(shape):
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_SOLID, m)
    return [TopoDS.Solid(m.FindKey(i)) for i in range(1, m.Extent() + 1)]


def canon_dir(d):
    """Unit vector with a deterministic sign (first significant component positive)."""
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    for c in d:
        if abs(c) > 1e-9:
            return d if c > 0 else -d
    return d


def axis_foot(origin, direction):
    """Point on the axis line closest to the world origin (canonical axis location)."""
    o, d = np.asarray(origin, float), canon_dir(direction)
    return o - np.dot(o, d) * d


def face_surface(face):
    """(type_name, params dict) for a face, from its underlying analytic surface."""
    s = BRepAdaptor_Surface(face, True)
    name = surface_type_name(s.GetType())
    p = {}
    if name == "plane":
        ax = s.Plane().Axis()
        p["origin"] = _xyz(ax.Location())
        p["normal"] = _xyz(ax.Direction())  # geometric, NOT orientation-corrected
    elif name == "cylinder":
        c = s.Cylinder()
        d = canon_dir(_xyz(c.Axis().Direction()))
        p["origin"] = axis_foot(_xyz(c.Axis().Location()), d).tolist()
        p["axis"] = d.tolist()
        p["radius"] = c.Radius()
    elif name == "cone":
        c = s.Cone()
        p["apex"] = _xyz(c.Apex())
        p["axis"] = _xyz(c.Axis().Direction())  # apex -> base direction depends on build
        p["semi_angle"] = c.SemiAngle()
        p["ref_radius"] = c.RefRadius()
        p["origin"] = _xyz(c.Location())
    elif name == "sphere":
        c = s.Sphere()
        p["center"] = _xyz(c.Location())
        p["radius"] = c.Radius()
    elif name == "torus":
        c = s.Torus()
        d = canon_dir(_xyz(c.Axis().Direction()))
        p["center"] = _xyz(c.Location())
        p["axis"] = d.tolist()
        p["major_radius"] = c.MajorRadius()
        p["minor_radius"] = c.MinorRadius()
    return name, p


def _xyz(v):
    return [v.X(), v.Y(), v.Z()]


# ------------------------------------------------------------------ cylinders

def dedupe_cylinders(cyls, rtol=0.01, ang_deg=1.0):
    """Collapse cylinder faces that are pieces of the same logical cylinder (same axis line
    and radius) — e.g. a hole split by a seam or by a slicing feature."""
    out = []
    for c in cyls:
        if not any(same_cylinder(c, o, rtol, ang_deg) for o in out):
            out.append(c)
    return out


def same_cylinder(a, b, rtol=0.01, ang_deg=1.0):
    ra, rb = a["radius"], b["radius"]
    if abs(ra - rb) > rtol * max(ra, rb):
        return False
    da, db = canon_dir(a["axis"]), canon_dir(b["axis"])
    if np.degrees(np.arccos(min(1.0, abs(float(np.dot(da, db)))))) > ang_deg:
        return False
    # distance between the (near-parallel) axis lines
    off = np.asarray(b["origin"], float) - np.asarray(a["origin"], float)
    perp = off - np.dot(off, da) * da
    return np.linalg.norm(perp) <= max(rtol * ra, 0.05)


def match_cylinders(truth, found, rtol=0.01, ang_deg=1.0):
    """One-to-one greedy match of logical cylinders. Returns (#truth matched, #truth)."""
    truth = dedupe_cylinders(truth, rtol, ang_deg)
    found = dedupe_cylinders(found, rtol, ang_deg)
    used = set()
    hits = 0
    for t in truth:
        for i, f in enumerate(found):
            if i not in used and same_cylinder(t, f, rtol, ang_deg):
                used.add(i)
                hits += 1
                break
    return hits, len(truth)


# ------------------------------------------------------------------ STL / mesh

STL_REC = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def write_stl(path, tris, header=b"stl2step test corpus"):
    tris = np.asarray(tris, np.float64)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1)
    n[ln > 0] /= ln[ln > 0, None]
    rec = np.zeros(len(tris), STL_REC)
    rec["n"] = n
    rec["v"] = tris
    with open(path, "wb") as f:
        f.write(header[:80].ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        f.write(rec.tobytes())


def read_stl(path):
    data = Path(path).read_bytes()
    (count,) = struct.unpack_from("<I", data, 80)
    if 84 + count * 50 != len(data):
        raise ValueError(f"{path}: not a binary STL")
    return np.frombuffer(data, STL_REC, count, 84)["v"].astype(np.float64)


def mesh_volume(tris):
    """Signed volume via the divergence theorem (positive for outward winding)."""
    return float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6.0)


def open_edges(tris):
    """Count directed edges with no opposite partner (0 for a closed, consistently wound mesh)."""
    flat = tris.reshape(-1, 3).astype(np.float32)
    _, inv = np.unique(flat, axis=0, return_inverse=True)
    f = inv.reshape(-1, 3)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    fwd = {tuple(x) for x in e.tolist()}
    return sum((b, a) not in fwd for a, b in fwd) + (len(e) - len(fwd))


# ------------------------------------------------------------------ truth lookup

def truth_for(path):
    """Ground-truth JSON for a corpus file (<name>_<res>.stl / .step -> <name>.json)."""
    p = Path(path)
    stem = p.stem
    for r in RESOLUTIONS:
        if stem.endswith("_" + r):
            stem = stem[: -len(r) - 1]
            break
    for d in (p.parent, CORPUS):
        j = d / f"{stem}.json"
        if j.exists():
            return json.loads(j.read_text())
    return None


def resolution_of(path):
    stem = Path(path).stem
    for r in RESOLUTIONS:
        if stem.endswith("_" + r):
            return r
    return None
