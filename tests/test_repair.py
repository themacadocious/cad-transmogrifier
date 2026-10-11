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


# ---------------------------------------------------------------- cap triangles

def cap_box(eps):
    """Unit box whose top face has a cap: vertex c sits eps off the 4-6 diagonal, so
    triangle (4,c,6) has its apex almost on its longest edge."""
    p, f = box()
    p = np.vstack([p, [[0.5 + eps, 0.5 - eps, 1]]])          # c = 8
    top = [i for i, t in enumerate(f.tolist()) if t == [4, 5, 6]][0]
    f = np.vstack([np.delete(f, top, axis=0), [[4, 5, 8], [5, 6, 8], [4, 8, 6]]])
    return p, f


def test_cap_flipped_coplanar():
    p, f = cap_box(1e-7)
    d = repair.diagnose(p, f, TOL)
    assert d["closed_manifold"] and d["caps"] == 1
    p2, f2, notes = repair.repair(p, f, TOL)
    d2 = check(p2, f2, 1.0, shells=1)
    assert d2["caps"] == 0
    assert any("cap" in n and "1 -> 0" in n for n in notes)
    assert np.array_equal(p2, p)               # a flip moves no vertex
    assert_solid(p2, f2, 1.0)


def test_cap_across_crease_reported():
    # front face (y=0) has c just below the top edge 4-5: the cap's neighbour across 4-5
    # is on the top face, so the flip bends the surface by eps (reported)
    eps = 5e-6
    p, f = box()
    p = np.vstack([p, [[0.5, 0, 1 - eps]]])                  # c = 8
    f = np.vstack([np.delete(f, [5], axis=0), [[0, 5, 8], [0, 8, 4], [8, 5, 4]]])
    assert repair.diagnose(p, f, TOL)["caps"] == 1
    p2, f2, notes = repair.repair(p, f, TOL)
    d = check(p2, f2, repair.diagnose(p2, f2)["volume"], shells=1)
    assert d["caps"] == 0 and d["volume"] == pytest.approx(1.0, abs=1e-5)
    assert any("crease" in n for n in notes)


def test_cap_tol_zero_disables():
    p, f = cap_box(1e-7)
    p2, f2, notes = repair.repair(p, f, TOL, cap_tol=0)
    assert notes == [] and f2 is f


def test_fat_triangle_is_not_a_cap():
    p, f = cap_box(1e-3)                       # 1.4e-3 off the diagonal: a real triangle
    assert repair.diagnose(p, f, TOL)["caps"] == 0
    p2, f2, notes = repair.repair(p, f, TOL)
    assert notes == [] and f2 is f


# ---------------------------------------------------------------- needles

def test_needle_collapsed():
    # w sits 5e-6 from corner 6 on the 6-7 edge: (4,6,w) and (2,w,6) are needles
    p, f = box()
    p = np.vstack([p, p[6] + [-5e-6, 0, 0]])                 # w = 8
    top = [i for i, t in enumerate(f.tolist()) if t == [4, 6, 7]][0]
    back = [i for i, t in enumerate(f.tolist()) if t == [2, 7, 6]][0]
    f = np.vstack([np.delete(f, [top, back], axis=0),
                   [[4, 6, 8], [4, 8, 7], [2, 7, 8], [2, 8, 6]]])
    d = repair.diagnose(p, f, TOL)
    assert d["closed_manifold"] and d["needles"] == 2
    p2, f2, notes = repair.repair(p, f, TOL)
    d2 = check(p2, f2, 1.0, shells=1)
    assert d2["needles"] == 0 and len(f2) == 12
    assert any("needle" in n for n in notes)
    assert_solid(p2, f2, 1.0)


def test_needle_collapse_respects_link_condition():
    # a tetrahedron-like tiny closed shell: collapsing its short edge would leave two
    # back-to-back triangles, so it must be refused (the shell is a sliver and dropped)
    p, f = box((0, 0, 0), 10.0)
    q = np.array([[20, 0, 0], [20 + 5e-6, 0, 0], [20, 1, 0], [20, 0, 1]], float)
    tf = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    P, F = merge((p, f), (q, tf))
    p2, f2, notes = repair.repair(P, F, TOL)
    d = repair.diagnose(p2, f2, TOL)
    assert d["closed_manifold"] and d["duplicate"] == 0


# ---------------------------------------------------------------- self-intersections

def grid_box(size=(20.0, 20.0, 1.0), step=0.5):
    """Closed box whose faces are regular triangle grids (welded, outward-wound)."""
    sx, sy, sz = size
    tris = []

    def face(origin, u, v, nu, nv):
        o, u, v = (np.asarray(x, float) for x in (origin, u, v))
        for i in range(nu):
            for j in range(nv):
                a = o + u * i / nu + v * j / nv
                b = o + u * (i + 1) / nu + v * j / nv
                c = o + u * (i + 1) / nu + v * (j + 1) / nv
                d = o + u * i / nu + v * (j + 1) / nv
                tris.extend([[a, b, c], [a, c, d]])
    n = lambda length: max(1, int(round(length / step)))  # noqa: E731
    face((0, 0, 0), (0, sy, 0), (sx, 0, 0), n(sy), n(sx))          # z=0, normal -z
    face((0, 0, sz), (sx, 0, 0), (0, sy, 0), n(sx), n(sy))         # z=sz, +z
    face((0, 0, 0), (sx, 0, 0), (0, 0, sz), n(sx), n(sz))          # y=0, -y
    face((0, sy, 0), (0, 0, sz), (sx, 0, 0), n(sz), n(sx))         # y=sy, +y
    face((0, 0, 0), (0, 0, sz), (0, sy, 0), n(sz), n(sy))          # x=0, -x
    face((sx, 0, 0), (0, sy, 0), (0, 0, sz), n(sy), n(sz))         # x=sx, +x
    T = np.array(tris)
    return weld_points(T.reshape(-1, 3), np.arange(len(T) * 3).reshape(-1, 3))


def test_grid_box_is_clean():
    p, f = grid_box()
    d = repair.diagnose(p, f)
    assert d["closed_manifold"] and d["self_intersections"] == 0
    assert d["volume"] == pytest.approx(400.0)
    p2, f2, notes = repair.repair(p, f, TOL)
    assert notes == [] and f2 is f


def test_sat_basic_cases():
    A = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float)
    cross = np.array([[[0.2, 0.2, -1], [0.2, 0.2, 1], [0.3, 0.25, 0]]], float)   # pierces A
    apart = A + [0, 0, 1]
    coplanar = A + [0.2, 0.2, 0]                                                # overlaps in-plane
    touching = np.array([[[1, 0, 0], [2, 0, 0], [1, 1, 0]]], float)             # one shared point
    assert repair._tri_tri_sat(A, cross, 1e-9)[0]
    assert not repair._tri_tri_sat(A, apart, 1e-9)[0]
    assert repair._tri_tri_sat(A, coplanar, 1e-9)[0]
    assert not repair._tri_tri_sat(A, touching, 1e-9)[0]


def test_folded_flap_self_intersection_repaired():
    # push one interior top vertex down through the 1 mm thick plate: its fan of
    # triangles pierces the bottom face
    p, f = grid_box()
    top = np.flatnonzero((np.abs(p[:, 2] - 1) < 1e-9) & (np.abs(p[:, 0] - 10) < 1e-9)
                         & (np.abs(p[:, 1] - 10) < 1e-9))[0]
    p = p.copy()
    p[top, 2] = -0.5
    d = repair.diagnose(p, f, TOL)
    assert d["closed_manifold"] and d["self_intersections"] > 0
    p2, f2, notes = repair.repair(p, f, TOL, self_intersections=True)
    d2 = repair.diagnose(p2, f2, TOL)
    assert d2["closed_manifold"] and d2["self_intersections"] == 0, notes
    assert d2["volume"] == pytest.approx(400.0, rel=1e-6)
    assert any("self-intersections" in n for n in notes)
    assert_solid(p2, f2, 400.0)


def test_overlapping_separate_shells_untouched():
    p, f = merge(box((0, 0, 0), 2.0), box((1, 1, 1), 2.0))
    d = repair.diagnose(p, f, TOL)
    assert d["self_intersections"] == 0       # crossings between shells don't count
    p2, f2, notes = repair.repair(p, f, TOL)
    assert notes == [] and f2 is f and p2 is p


def test_large_self_intersection_left_with_warning():
    # the whole inside of the top face pushed through the bottom: far more than 1% of
    # the shell, so it is reported, not cut out
    p, f = grid_box((4.0, 4.0, 1.0), 1.0)
    p = p.copy()
    inner = (np.abs(p[:, 2] - 1) < 1e-9) & (p[:, 0] > 0.5) & (p[:, 0] < 3.5) \
        & (p[:, 1] > 0.5) & (p[:, 1] < 3.5)
    p[inner, 2] = -0.5
    assert repair.diagnose(p, f, TOL)["self_intersections"] > 0
    p2, f2, notes = repair.repair(p, f, TOL)
    assert any("WARNING" in n and "self-intersect" in n for n in notes)
    assert f2 is f                            # nothing changed, only reported


def test_self_intersections_warned_not_fixed_by_default():
    p, f = grid_box()
    top = np.flatnonzero((np.abs(p[:, 2] - 1) < 1e-9) & (np.abs(p[:, 0] - 10) < 1e-9)
                         & (np.abs(p[:, 1] - 10) < 1e-9))[0]
    p = p.copy()
    p[top, 2] = -0.5
    p2, f2, notes = repair.repair(p, f, TOL)
    assert p2 is p and f2 is f
    assert any(n.startswith("WARNING") and "self-intersecting" in n for n in notes)
