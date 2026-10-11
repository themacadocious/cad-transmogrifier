"""repair.py: synthetic broken meshes must come back closed/manifold/outward with the
right volume, and clean meshes (the whole corpus) must come back untouched."""

import numpy as np
import pytest

import repair
import stl2step
from conftest import STLS

TOL = 1e-6

# unit cube, outward-wound
BOX_V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
BOX_F = np.array([[0, 2, 1], [0, 3, 2],      # z=0
                  [4, 5, 6], [4, 6, 7],      # z=1
                  [0, 1, 5], [0, 5, 4],      # y=0
                  [2, 3, 7], [2, 7, 6],      # y=1
                  [1, 2, 6], [1, 6, 5],      # x=1
                  [3, 0, 4], [3, 4, 7]])     # x=0


def box(origin=(0, 0, 0), size=1.0):
    return BOX_V * size + np.asarray(origin, float), BOX_F.copy()


def merge(*meshes):
    pts, fcs, off = [], [], 0
    for p, f in meshes:
        pts.append(p)
        fcs.append(f + off)
        off += len(p)
    return np.vstack(pts), np.vstack(fcs)


def weld_points(p, f):
    """Merge coincident vertices (like stl2step.weld, without dropping triangles)."""
    u, inv = np.unique(np.round(p / 1e-9).astype(np.int64), axis=0, return_inverse=True)
    out = np.zeros((len(u), 3))
    out[inv.ravel()] = p
    return out, inv.ravel()[f]


def check(p, f, volume, shells=None):
    d = repair.diagnose(p, f)
    assert d["closed_manifold"], d
    assert d["volume"] == pytest.approx(volume, rel=1e-9, abs=1e-9)
    if shells is not None:
        assert d["components"] == shells
    return d


def assert_solid(p, f, volume):
    """stl2step's own faceted path must now produce valid closed solid(s)."""
    from OCP.BRepCheck import BRepCheck_Analyzer
    shell, failed = stl2step.build_shell(p, f)
    assert failed == 0
    shape, is_solid = stl2step.make_solid(shell)
    assert is_solid
    assert BRepCheck_Analyzer(shape).IsValid()
    assert stl2step.volume(shape) == pytest.approx(volume, rel=1e-6)


# ---------------------------------------------------------------- no-op

def test_clean_box_is_noop():
    p, f = box()
    p2, f2, notes = repair.repair(p, f, TOL)
    assert p2 is p and f2 is f and notes == []


@pytest.mark.parametrize("stl", STLS, ids=[s.stem for s in STLS])
def test_corpus_noop(stl):
    tris = stl2step.read_stl(stl)
    diag = float(np.linalg.norm(np.ptp(tris.reshape(-1, 3), axis=0)))
    tol = max(diag * 1e-7, 1e-6)
    p, f, _ = stl2step.weld(tris, tol)
    p2, f2, notes = repair.repair(p, f, tol)
    assert notes == []
    assert p2 is p and f2 is f
    assert p2.tobytes() == p.tobytes() and f2.tobytes() == f.tobytes()


# ---------------------------------------------------------------- defects

def test_missing_triangle():
    p, f = box()
    f = np.delete(f, 3, axis=0)
    assert repair.diagnose(p, f)["boundary_edges"] == 3
    p2, f2, notes = repair.repair(p, f, TOL)
    check(p2, f2, 1.0, shells=1)
    assert any("filled 1 small hole" in n for n in notes)
    assert_solid(p2, f2, 1.0)


def test_missing_face_quad_hole():
    p, f = box()
    f = np.delete(f, [2, 3], axis=0)          # whole top face gone: a 4-edge hole
    p2, f2, notes = repair.repair(p, f, TOL)
    check(p2, f2, 1.0, shells=1)


def test_flipped_triangle():
    p, f = box()
    f[5] = f[5][::-1]
    assert repair.diagnose(p, f)["inconsistent_edges"] == 3
    p2, f2, notes = repair.repair(p, f, TOL)
    check(p2, f2, 1.0, shells=1)
    assert any("flipped 1 triangles" in n for n in notes)


def test_inside_out_box():
    p, f = box()
    p2, f2, notes = repair.repair(p, f[:, ::-1].copy(), TOL)
    check(p2, f2, 1.0)
    assert any("inside-out" in n for n in notes)


def test_two_boxes_sharing_an_edge():
    # second box touches the first only along the edge x=1, y=1 (z 0..1)
    p, f = weld_points(*merge(box(), box((1, 1, 0))))
    d = repair.diagnose(p, f)
    assert d["nonmanifold_edges"] == 1
    p2, f2, notes = repair.repair(p, f, TOL)
    check(p2, f2, 2.0, shells=2)
    assert_solid(p2, f2, 2.0)


def box_b():
    """Unit box at x=1 whose x=0 face is triangulated exactly like box()'s x=1 face."""
    bp, bf = box((1, 0, 0))
    bf[10:12] = [[3, 0, 7], [0, 4, 7]]
    return bp, bf


def test_two_boxes_sharing_a_face_fuse():
    # back-to-back identical triangles on the shared face (an exporter that didn't union)
    p, f = weld_points(*merge(box(), box_b()))
    p2, f2, notes = repair.repair(p, f, TOL)
    check(p2, f2, 2.0, shells=1)
    assert any("internal walls" in n for n in notes)
    assert_solid(p2, f2, 2.0)


def test_two_boxes_sharing_a_face_different_triangulation():
    # box()'s x=1 face and box((1,0,0))'s x=0 face use different diagonals
    p, f = weld_points(*merge(box(), box((1, 0, 0))))
    assert repair.diagnose(p, f)["nonmanifold_edges"] == 4
    p2, f2, notes = repair.repair(p, f, TOL)
    check(p2, f2, 2.0, shells=1)
    assert any("back-to-back patches" in n for n in notes)


def test_two_boxes_one_copy_of_shared_face():
    # what stl2step.weld leaves of the case above: it drops one of each back-to-back pair,
    # so the shared face survives once as an internal zero-thickness wall
    tris = merge(box(), box_b())
    p, f, dropped = stl2step.weld(tris[0][tris[1]], 1e-7)
    assert dropped == 2
    assert repair.diagnose(p, f)["nonmanifold_edges"] == 4
    p2, f2, notes = repair.repair(p, f, TOL)
    check(p2, f2, 2.0, shells=1)
    assert any("internal wall" in n for n in notes)


def test_duplicated_faces():
    p, f = box()
    f = np.vstack([f, f[[0, 4, 7]], f[[0]]])  # same-wound duplicates (one of them twice)
    p2, f2, notes = repair.repair(p, f, TOL)
    check(p2, f2, 1.0, shells=1)
    assert any("dropped 4 duplicate" in n for n in notes)


def test_degenerate_triangles():
    p, f = box()
    f = np.vstack([f, [[0, 0, 1], [0, 1, 2]]])   # repeated index; ... (0,1,2) is a real tri
    f = f[:-1]
    p = np.vstack([p, [[0.5, 0, 0]]])            # collinear with 0-1
    f = np.vstack([f, [[0, 1, 8]]])
    p2, f2, notes = repair.repair(p, f, TOL)
    check(p2, f2, 1.0, shells=1)
    assert any("degenerate" in n for n in notes)


def test_t_junction_crack():
    # the front face (y=0) has an extra vertex m on its top edge; the top face doesn't:
    # a zero-area crack m-(0,0,1)-(1,0,1) -> "not watertight"
    p, f = box()
    p = np.vstack([p, [[0.5, 0, 1]]])         # m = 8
    f = np.delete(f, [4, 5], axis=0)
    f = np.vstack([f, [[0, 1, 8], [1, 5, 8], [0, 8, 4]]])
    d = repair.diagnose(p, f)
    assert d["boundary_edges"] == 3
    p2, f2, notes = repair.repair(p, f, TOL)
    check(p2, f2, 1.0, shells=1)
    assert any("T-junction" in n for n in notes)
    assert_solid(p2, f2, 1.0)


def test_t_junction_after_weld_drops_needle():
    # a proper STL closes the T-junction with a zero-area needle triangle; stl2step.weld
    # drops it as degenerate, re-opening the crack
    p, f = box()
    p = np.vstack([p, [[0.5, 0, 1]]])
    f = np.delete(f, [4, 5], axis=0)
    f = np.vstack([f, [[0, 1, 8], [1, 5, 8], [0, 8, 4], [4, 8, 5]]])   # needle 4-8-5
    pw, fw, dropped = stl2step.weld(p[f], 1e-7)
    assert dropped == 1
    p2, f2, notes = repair.repair(pw, fw, 1e-7)
    check(p2, f2, 1.0, shells=1)


def test_near_coincident_crack():
    # two vertices that should be one, 1e-5 apart (above the weld tolerance)
    p, f = box()
    p = np.vstack([p, p[6] + [1e-5, 0, 0]])   # 8 ~ 6
    f[3] = [4, 8, 7]                          # top face triangle uses the copy
    assert repair.diagnose(p, f)["boundary_edges"] > 0
    p2, f2, notes = repair.repair(p, f, 1e-6)
    d = check(p2, f2, repair.diagnose(p2, f2)["volume"], shells=1)
    assert d["volume"] == pytest.approx(1.0, rel=1e-4)


def test_internal_void_points_inward():
    outer = box((0, 0, 0), 3.0)
    inner = box((1, 1, 1), 1.0)               # wound outward like the outer: wrong for a void
    p, f = merge(outer, inner)
    p2, f2, notes = repair.repair(p, f, TOL)
    check(p2, f2, 27.0 - 1.0, shells=2)
    assert any("internal void" in n for n in notes)
    # already-correct void is left alone
    p3, f3, notes3 = repair.repair(p2, f2, TOL)
    assert notes3 == [] and f3 is f2


def test_overlapping_bodies_not_treated_as_void():
    # second box overlaps the first but sticks out: both stay outward
    p, f = merge(box((0, 0, 0), 2.0), box((1, 1, 1), 2.0))
    p2, f2, notes = repair.repair(p, f, TOL)
    assert notes == [] and f2 is f


def test_body_inside_void_stays_outward():
    p, f = merge(box((0, 0, 0), 5.0), box((1, 1, 1), 3.0), box((2, 2, 2), 1.0))
    f[12:24] = f[12:24, ::-1]                 # middle shell is a correct (inward) void
    p2, f2, notes = repair.repair(p, f, TOL)
    assert notes == [] and f2 is f            # 3 levels: solid / void / solid is consistent


def test_sliver_component_dropped():
    p, f = box((0, 0, 0), 10.0)
    sp, sf = box((20, 0, 0), 1.0)
    sp[:, 2] *= 1e-6                          # flat sliver: tiny volume and area
    sp[:, :2] *= 1e-2
    p, f = merge((p, f), (sp, sf))
    p2, f2, notes = repair.repair(p, f, 1e-9)
    check(p2, f2, 1000.0, shells=1)
    assert any("sliver" in n for n in notes)


def test_large_open_hole_reported_not_filled():
    # a big non-planar hole (a box with two adjacent faces missing) is not guessed at
    p, f = box()
    f = np.delete(f, [2, 3, 4, 5], axis=0)
    p2, f2, notes = repair.repair(p, f, TOL)
    assert any("WARNING" in n and "open" in n for n in notes)
