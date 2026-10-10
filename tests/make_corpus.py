#!/usr/bin/env python3
"""Generate the ground-truth test corpus in tests/corpus/.

For every part this writes:
  <name>.json                     true volume, bbox, faces by surface type, every cylinder
  <name>_<res>.stl                binary STL, res in {coarse, fine}
  <name>_<res>.labels.npz         per-triangle ground-truth segmentation, same order as the STL

The STL is written here (not with StlAPI_Writer) from the per-face triangulations, so
triangle i of the STL is exactly labels[i]. .labels.npz arrays (F = number of B-rep faces):
  labels        (T,)  int32   source face index of each STL triangle
  face_type     (F,)  int8    index into type_names
  type_names    (7,)  str     plane cylinder cone sphere torus bspline other
  face_body     (F,)  int32   which solid the face belongs to
  face_sense    (F,)  int8    curved faces: +1 convex (boss, normals point away from axis /
                              centre), -1 concave (hole); 0 for planes / freeform
  plane_origin, plane_normal         (F,3)  outward unit normal; NaN unless plane
  cyl_origin, cyl_axis, cyl_radius   (F,3),(F,3),(F,)  axis foot point nearest the world
                                     origin + canonical axis sign; NaN unless cylinder
  cone_apex, cone_axis, cone_semi_angle, sphere_center, sphere_radius,
  torus_center, torus_axis, torus_major, torus_minor   likewise for the other analytic types

Usage: .venv/bin/python tests/make_corpus.py [part names...]
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeSphere,
)
from OCP.BRepTools import BRepTools
from OCP.GProp import GProp_GProps
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopAbs import TopAbs_EDGE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Compound
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_common import (  # noqa: E402
    CORPUS, TYPE_CODE, TYPE_NAMES, face_surface, mesh_volume, open_edges, read_stl,
    solids, unique_faces, write_stl,
)

RES = {"coarse": (0.1, 0.5), "fine": (0.01, 0.1)}  # (linear deflection mm, angular rad)
Z = gp_Dir(0, 0, 1)


# ------------------------------------------------------------------ modelling helpers

def box(x, y, z, dx, dy, dz):
    return BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), dx, dy, dz).Shape()


def cyl(x, y, z, r, h, d=Z):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, z), d), r, h).Shape()


def cut(a, *tools):
    for t in tools:
        op = BRepAlgoAPI_Cut(a, t)
        assert op.IsDone()
        a = op.Shape()
    return a


def fuse(a, *tools):
    for t in tools:
        op = BRepAlgoAPI_Fuse(a, t)
        assert op.IsDone()
        a = op.Shape()
    return a


def through(x, y, r, z0=-1, h=200):
    return cyl(x, y, z0, r, h)


def edges_where(shape, pred):
    out, exp = [], TopExp_Explorer(shape, TopAbs_EDGE)
    seen = []
    while exp.More():
        e = TopoDS.Edge(exp.Current())
        if not any(e.IsSame(s) for s in seen):
            seen.append(e)
            b = Bnd_Box()
            BRepBndLib.Add_s(e, b)
            lo, hi = b.CornerMin(), b.CornerMax()
            if pred((lo.X(), lo.Y(), lo.Z()), (hi.X(), hi.Y(), hi.Z())):
                out.append(e)
        exp.Next()
    return out


def fillet(shape, edges, r):
    mk = BRepFilletAPI_MakeFillet(shape)
    for e in edges:
        mk.Add(r, e)
    mk.Build()
    assert mk.IsDone(), "fillet failed"
    return mk.Shape()


def near(a, b, tol=0.2):
    return abs(a - b) < tol


# ------------------------------------------------------------------ parts (mm)

def p_box():
    return box(0, 0, 0, 40, 30, 20)


def p_box_hole():
    return cut(box(0, 0, 0, 50, 40, 15), through(25, 20, 5))


def p_plate_holes_cbore():
    plate = box(0, 0, 0, 80, 60, 8)
    holes = [through(x, y, 3) for x in (8, 72) for y in (8, 52)]
    cbore = cyl(40, 30, 4, 8, 10)          # Ø16 x 4 deep from the top
    return cut(plate, *holes, through(40, 30, 4), cbore)


def p_l_bracket():
    base = box(0, 0, 0, 60, 40, 6)
    upright = box(0, 0, 0, 6, 40, 50)
    body = fuse(base, upright)
    # the inner concave edge runs along Y at x=6, z=6
    inner = edges_where(body, lambda lo, hi: near(lo[0], 6) and near(hi[0], 6)
                        and near(lo[2], 6) and near(hi[2], 6) and hi[1] - lo[1] > 30)
    assert len(inner) == 1, inner
    body = fillet(body, inner, 5)
    return cut(body, through(35, 12, 3), through(35, 28, 3))


def p_boss_plate():
    return fuse(box(0, 0, 0, 60, 60, 6), cyl(30, 30, 6, 10, 20))


def p_slot_plate():
    plate = box(0, 0, 0, 80, 40, 6)
    # stadium 30 long x 10 wide (centres 15 apart), built as one tool
    slot = fuse(box(32.5, 15, -1, 15, 10, 10), cyl(32.5, 20, -1, 5, 10), cyl(47.5, 20, -1, 5, 10))
    return cut(plate, slot)


def p_tube():
    return cut(cyl(0, 0, 0, 15, 50), cyl(0, 0, -1, 10, 52))


def p_countersink_dome():
    plate = box(0, 0, 0, 70, 50, 10)
    # 90° countersink: cone from Ø14 at the top face down to Ø6, then a Ø6 through hole
    sink = BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(18, 25, 6), Z), 3, 7, 4).Shape()
    cap = cyl(18, 25, 10, 7, 2)            # carry the cone wall cleanly out of the top face
    body = cut(plate, through(18, 25, 3), sink, cap)
    # hemispherical dome r=12 sitting on the top face
    dome = BRepPrimAPI_MakeSphere(gp_Ax2(gp_Pnt(48, 25, 10), Z), 12, 0, math.pi / 2).Shape()
    return fuse(body, dome)


def p_flanged_shaft():
    flange = cyl(0, 0, 0, 20, 8)
    shaft = cyl(0, 0, 8, 8, 30)
    body = fuse(flange, shaft)
    # circular junction edge at z=8, r=8 -> torus fillet
    ring = edges_where(body, lambda lo, hi: near(lo[2], 8) and near(hi[2], 8)
                       and near(hi[0] - lo[0], 16, 0.5))
    assert len(ring) == 1, ring
    body = fillet(body, ring, 3)
    return cut(body, through(0, 0, 4))


def p_rounded_block():
    b = box(0, 0, 0, 50, 40, 20)
    allr = edges_where(b, lambda lo, hi: True)
    return fillet(b, allr, 4)


def p_two_bodies():
    comp = TopoDS_Compound()
    bld = BRep_Builder()
    bld.MakeCompound(comp)
    bld.Add(comp, box(0, 0, 0, 30, 30, 10))
    bld.Add(comp, cyl(60, 15, 0, 10, 20))
    return comp


PARTS = {
    "box": (p_box, "plain 40x30x20 box"),
    "box_hole": (p_box_hole, "50x40x15 block, one Ø10 through hole"),
    "plate_holes_cbore": (p_plate_holes_cbore, "80x60x8 plate, 4x Ø6 holes, Ø8 hole with Ø16x4 counterbore"),
    "l_bracket": (p_l_bracket, "L-bracket, R5 inner fillet, 2x Ø6 holes"),
    "boss_plate": (p_boss_plate, "60x60x6 plate with Ø20x20 boss"),
    "slot_plate": (p_slot_plate, "80x40x6 plate with a 30x10 stadium slot"),
    "tube": (p_tube, "Ø30/Ø20 tube, 50 long"),
    "countersink_dome": (p_countersink_dome, "plate with a 90° countersunk hole (cone) and R12 hemisphere dome"),
    "flanged_shaft": (p_flanged_shaft, "Ø40 flange + Ø16 shaft, R3 torus fillet, Ø8 bore"),
    "rounded_block": (p_rounded_block, "50x40x20 block, all edges R4 (cylinders + corner blends)"),
    "two_bodies": (p_two_bodies, "two disjoint solids: box and cylinder"),
}


# ------------------------------------------------------------------ ground truth

def clean(shape):
    """Merge same-domain faces so face counts are canonical (one face per logical surface
    patch), as a CAD user would see them."""
    u = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
    u.Build()
    return u.Shape()


def volume(shape):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def truth(name, desc, shape):
    faces = unique_faces(shape)
    counts = {t: 0 for t in TYPE_NAMES}
    cylinders = []
    for f in faces:
        t, p = face_surface(f)
        counts[t] += 1
        if t == "cylinder":
            cylinders.append({k: p[k] for k in ("origin", "axis", "radius")})
    b = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, b, False, False)
    lo, hi = b.CornerMin(), b.CornerMax()
    return {
        "name": name,
        "description": desc,
        "units": "mm",
        "volume": volume(shape),
        "bbox": {"min": [lo.X(), lo.Y(), lo.Z()], "max": [hi.X(), hi.Y(), hi.Z()]},
        "solids": len(solids(shape)),
        "faces": len(faces),
        "faces_by_type": counts,
        "cylinders": cylinders,
    }


# ------------------------------------------------------------------ meshing + labels

def nan(*shape):
    return np.full(shape, np.nan)


def mesh_with_labels(shape, lin, ang):
    """Mesh `shape` and return (tris (T,3,3), arrays dict for the .npz)."""
    BRepTools.Clean_s(shape)
    mesher = BRepMesh_IncrementalMesh(shape, lin, False, ang, True)
    mesher.Perform()
    assert mesher.IsDone()

    faces = unique_faces(shape)
    F = len(faces)
    a = {
        "face_type": np.zeros(F, np.int8),
        "face_body": np.zeros(F, np.int32),
        "face_sense": np.zeros(F, np.int8),
        "plane_origin": nan(F, 3), "plane_normal": nan(F, 3),
        "cyl_origin": nan(F, 3), "cyl_axis": nan(F, 3), "cyl_radius": nan(F),
        "cone_apex": nan(F, 3), "cone_axis": nan(F, 3), "cone_semi_angle": nan(F),
        "sphere_center": nan(F, 3), "sphere_radius": nan(F),
        "torus_center": nan(F, 3), "torus_axis": nan(F, 3), "torus_major": nan(F), "torus_minor": nan(F),
    }
    body_faces = [unique_faces(s) for s in solids(shape)]
    all_tris, labels = [], []
    for fi, face in enumerate(faces):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        assert tri is not None, f"face {fi} has no triangulation"
        trsf = loc.Transformation()
        nodes = np.empty((tri.NbNodes(), 3))
        for i in range(1, tri.NbNodes() + 1):
            p = tri.Node(i).Transformed(trsf)
            nodes[i - 1] = (p.X(), p.Y(), p.Z())
        idx = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)]) - 1
        if face.Orientation() == TopAbs_REVERSED:  # REVERSED face: flip winding to stay outward
            idx = idx[:, [0, 2, 1]]
        t = nodes[idx]
        all_tris.append(t)
        labels.append(np.full(len(t), fi, np.int32))

        # per-face params
        for bi, bf in enumerate(body_faces):
            if any(face.IsSame(g) for g in bf):
                a["face_body"][fi] = bi
                break
        name, p = face_surface(face)
        a["face_type"][fi] = TYPE_CODE[name]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])  # area-weighted outward normals
        cen = t.mean(axis=1)
        if name == "plane":
            nn = n.sum(0)
            nn /= np.linalg.norm(nn)
            geo = np.array(p["normal"])
            assert abs(abs(nn @ geo) - 1) < 1e-6, (fi, nn, geo)
            a["plane_normal"][fi] = geo if nn @ geo > 0 else -geo
            a["plane_origin"][fi] = p["origin"]
        elif name == "cylinder":
            a["cyl_origin"][fi], a["cyl_axis"][fi], a["cyl_radius"][fi] = p["origin"], p["axis"], p["radius"]
            o, d = np.array(p["origin"]), np.array(p["axis"])
            radial = (cen - o) - np.outer((cen - o) @ d, d)
            a["face_sense"][fi] = np.sign(np.einsum("ij,ij->", n, radial))
        elif name == "cone":
            a["cone_apex"][fi], a["cone_axis"][fi], a["cone_semi_angle"][fi] = p["apex"], p["axis"], p["semi_angle"]
            o, d = np.array(p["apex"]), np.array(p["axis"])
            radial = (cen - o) - np.outer((cen - o) @ d, d)
            a["face_sense"][fi] = np.sign(np.einsum("ij,ij->", n, radial))
        elif name == "sphere":
            a["sphere_center"][fi], a["sphere_radius"][fi] = p["center"], p["radius"]
            a["face_sense"][fi] = np.sign(np.einsum("ij,ij->", n, cen - np.array(p["center"])))
        elif name == "torus":
            a["torus_center"][fi], a["torus_axis"][fi] = p["center"], p["axis"]
            a["torus_major"][fi], a["torus_minor"][fi] = p["major_radius"], p["minor_radius"]
            # sense relative to the tube centre circle
            c, d, R = np.array(p["center"]), np.array(p["axis"]), p["major_radius"]
            v = cen - c
            rad = v - np.outer(v @ d, d)
            ring = c + rad / np.linalg.norm(rad, axis=1)[:, None] * R
            a["face_sense"][fi] = np.sign(np.einsum("ij,ij->", n, cen - ring))

    a["labels"] = np.concatenate(labels)
    a["type_names"] = np.array(TYPE_NAMES)
    return np.concatenate(all_tris), a


# ------------------------------------------------------------------ driver

def build(name):
    fn, desc = PARTS[name]
    shape = clean(fn())
    assert BRepCheck_Analyzer(shape).IsValid(), f"{name}: generated shape invalid"
    gt = truth(name, desc, shape)
    gt["meshes"] = {}
    for res, (lin, ang) in RES.items():
        tris, arrays = mesh_with_labels(shape, lin, ang)
        # BRepMesh emits zero-area slivers at sphere poles (two nodes at the singular point);
        # they create a duplicated edge, so drop any triangle that collapses in float32
        f32 = tris.astype(np.float32)
        keep = ~((f32[:, 0] == f32[:, 1]).all(1) | (f32[:, 1] == f32[:, 2]).all(1)
                 | (f32[:, 0] == f32[:, 2]).all(1))
        tris, arrays["labels"] = tris[keep], arrays["labels"][keep]
        stl = CORPUS / f"{name}_{res}.stl"
        write_stl(stl, tris, f"stl2step corpus {name} {res}".encode())
        np.savez_compressed(CORPUS / f"{name}_{res}.labels.npz", **arrays)

        # round-trip checks on what actually landed on disk (float32!)
        back = read_stl(stl)
        assert len(back) == len(arrays["labels"]) == len(tris), name
        mv = mesh_volume(back)
        err = (mv - gt["volume"]) / gt["volume"] * 100
        oe = open_edges(back)
        assert abs(err) < 1.0, f"{name}_{res}: mesh volume off by {err:.3f}%"
        assert oe == 0, f"{name}_{res}: {oe} open/non-manifold edges"
        gt["meshes"][res] = {
            "file": stl.name, "linear_deflection": lin, "angular_deflection": ang,
            "triangles": len(back), "dropped_degenerate": int((~keep).sum()), "mesh_volume": mv, "mesh_volume_error_pct": err,
        }
        print(f"  {name}_{res}: {len(back):7d} tris  mesh vol err {err:+.3f}%")
    (CORPUS / f"{name}.json").write_text(json.dumps(gt, indent=1) + "\n")
    c = {k: v for k, v in gt["faces_by_type"].items() if v}
    print(f"{name}: vol {gt['volume']:.2f}  {gt['faces']} faces {c}  {gt['solids']} solid(s)")


def main(names):
    CORPUS.mkdir(exist_ok=True)
    for n in names or PARTS:
        build(n)


if __name__ == "__main__":
    main(sys.argv[1:])
