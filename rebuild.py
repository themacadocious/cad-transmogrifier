"""Rebuild a B-rep solid from a segmented mesh ("prismatic" reconstruction).

Input is the welded mesh plus segment.segment()'s output: a label per triangle and a
Plane/Cylinder per label (-1 = freeform triangle, kept as its own flat facet).

Outline:
  1. every mesh edge between two different regions is a *boundary edge*; boundary edges
     between the same pair of regions are chained, split at corners (vertices touching
     three or more regions)
  2. each chain becomes ONE shared B-rep edge: a straight line, a circular arc on the
     cylinder, or (between flat faces only) a polyline
  3. a cylinder whose boundary can't be expressed as lines along its axis and circles
     around it (oblique cuts, freeform neighbours) is demoted back to facets, and step 1
     repeats until stable
  4. corners are snapped onto every analytic surface they touch, faces are built from
     their boundary loops, and the faces are closed into a solid

Raises RebuildError when the result is not a valid closed solid; the caller falls back
to the faceted conversion.
"""

from collections import defaultdict

import numpy as np
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_Copy,
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeWire,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepLib import BRepLib
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.Geom import (
    Geom_Circle,
    Geom_ConicalSurface,
    Geom_CylindricalSurface,
    Geom_Plane,
    Geom_SphericalSurface,
    Geom_ToroidalSurface,
    Geom_TrimmedCurve,
)
from OCP.Geom2d import Geom2d_Line
from OCP.GeomAbs import GeomAbs_Plane
from OCP.GeomAPI import GeomAPI_IntSS, GeomAPI_ProjectPointOnCurve
from OCP.ShapeAnalysis import ShapeAnalysis_Surface
from OCP.ShapeBuild import ShapeBuild_ReShape
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.collections import Map_TopoDS_Shape_TopTools_ShapeMapHasher as ShapeMap
from OCP.ShapeFix import ShapeFix_Face, ShapeFix_Shape
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Face, TopoDS_Shell, TopoDS_Solid
from OCP.gp import gp_Ax2, gp_Ax3, gp_Circ, gp_Dir, gp_Dir2d, gp_Pln, gp_Pnt, gp_Pnt2d


class RebuildError(Exception):
    pass


class _ChainError(RebuildError):
    """Something couldn't be built: demote `regions`' curved surfaces, and rebuild the
    boundaries of `poly` regions as plain polylines (no line simplification)."""

    def __init__(self, msg, regions, poly=()):
        super().__init__(msg)
        self.regions = regions
        self.poly = set(poly)


def _kind(s):
    """Surface kind from its fields: segment.py's Plane/Cylinder/Cone/Sphere/Torus."""
    if s is None:
        return None
    if hasattr(s, "half_angle"):
        return "cone"
    if hasattr(s, "major"):
        return "torus"
    if hasattr(s, "radius"):
        return "cylinder" if hasattr(s, "axis") else "sphere"
    return "plane"


def _is_cone(s):
    return _kind(s) == "cone"


def _is_cyl(s):
    return _kind(s) == "cylinder"


def _is_curved(s):
    return _kind(s) not in (None, "plane")


def _radius_at(s, h):
    """Radius of a cylinder/cone at height h along its axis (cone: measured from the apex)."""
    return s.radius if _is_cyl(s) else h * np.tan(s.half_angle)


def _surface_dist(pts, s):
    """Distance of points from an analytic surface (cone: radial, ~exact for small offsets)."""
    k = _kind(s)
    if k == "sphere":
        return np.abs(np.linalg.norm(np.atleast_2d(pts) - s.center, axis=1) - s.radius)
    if k == "torus":
        q = np.atleast_2d(pts) - s.center
        h = q @ s.axis
        rho = np.linalg.norm(q - np.outer(h, s.axis), axis=1)
        return np.abs(np.hypot(rho - s.major, h) - s.minor)
    off = np.atleast_2d(pts) - s.origin
    if k == "plane":
        return np.abs(off @ s.normal)
    h = off @ s.axis
    rho = np.linalg.norm(off - np.outer(h, s.axis), axis=1)
    d = np.abs(rho - _radius_at(s, h))
    return d * np.cos(s.half_angle) if _is_cone(s) else d


def _pnt(p):
    return gp_Pnt(float(p[0]), float(p[1]), float(p[2]))


def _dir(v):
    return gp_Dir(float(v[0]), float(v[1]), float(v[2]))


def mesh_volume(points, faces):
    p = points[faces]
    return float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0)


# ---------------------------------------------------------------- mesh topology

def _half_edges(faces):
    """Map directed edge (a, b) -> (triangle, slot). Raises if the mesh isn't manifold."""
    he = {}
    for t, tri in enumerate(faces.tolist()):
        for k in range(3):
            key = (tri[k], tri[(k + 1) % 3])
            if key in he:
                raise RebuildError("non-manifold or inconsistently wound mesh")
            he[key] = (t, k)
    for a, b in he:
        if (b, a) not in he:
            raise RebuildError("mesh is not watertight")
    return he


def _region_index(labels):
    """{region: triangle indices}, built in one pass (per-region scans are O(n^2))."""
    order = np.argsort(labels, kind="stable")
    regions, starts = np.unique(labels[order], return_index=True)
    return dict(zip(regions.tolist(), np.split(order, starts[1:])))


def _boundary_loops(faces, region, tris, he, labels):
    """Directed boundary half-edges of one region, linked into closed loops (vertex lists)."""
    out = {}
    for t in tris.tolist():
        tri = faces[t]
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            if labels[he[(b, a)][0]] != region:
                out[(a, b)] = t
    loops, used = [], set()
    for start in out:
        if start in used:
            continue
        loop, cur = [], start
        while cur not in used:
            used.add(cur)
            loop.append(cur)
            a, b = cur
            # rotate around b inside the region to find the next boundary half-edge
            t = out[cur]
            for _ in range(10_000):
                tri = faces[t].tolist()
                k = tri.index(b)
                c = tri[(k + 1) % 3]
                if (b, c) in out:
                    cur = (b, c)
                    break
                t = he[(c, b)][0]
            else:
                raise RebuildError("could not walk region boundary")
        if cur != start:
            raise RebuildError("region boundary loops are tangled")
        loops.append([e[0] for e in loop])
    return loops


# ---------------------------------------------------------------- chain geometry

def _fit_line(pts, tol):
    d = pts[-1] - pts[0]
    n = np.linalg.norm(d)
    if n < tol:
        return None
    d /= n
    off = pts - pts[0]
    dev = np.linalg.norm(off - np.outer(off @ d, d), axis=1)
    return d if dev.max() <= tol else None


def _circle_for(pts, s, tol):
    """Chain is a circle around the axis of cylinder/cone s? Returns (centre, radius) or None."""
    h = (pts - s.origin) @ s.axis
    if np.ptp(h) > tol or _surface_dist(pts, s).max() > tol:
        return None
    return s.origin + s.axis * h.mean(), float(_radius_at(s, h.mean()))


def _fit_circle(pts, tol):
    """Best-fit circle through 3D points: (centre, normal, radius) or None if they aren't one."""
    if len(pts) < 3:
        return None
    c = pts.mean(axis=0)
    _, sv, vt = np.linalg.svd(pts - c)
    n = vt[2]
    if np.abs((pts - c) @ n).max() > tol:
        return None
    x, y = vt[0], vt[1]
    u, v = (pts - c) @ x, (pts - c) @ y
    A = np.column_stack([u, v, np.ones_like(u)])
    sol, *_ = np.linalg.lstsq(A, u * u + v * v, rcond=None)
    cu, cv = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cu * cu + cv * cv
    if r2 <= 0:
        return None
    r = np.sqrt(r2)
    if np.abs(np.hypot(u - cu, v - cv) - r).max() > tol:
        return None
    return c + cu * x + cv * y, n, float(r)


def _geom_surface(s):
    if _kind(s) == "sphere":
        return Geom_SphericalSurface(gp_Ax3(_pnt(s.center), gp_Dir(0, 0, 1)), float(s.radius))
    if _kind(s) == "torus":
        return Geom_ToroidalSurface(gp_Ax3(_pnt(s.center), _dir(s.axis)), float(s.major), float(s.minor))
    if _is_cone(s):
        return Geom_ConicalSurface(gp_Ax3(_pnt(s.origin), _dir(s.axis)), float(s.half_angle), 0.0)
    if _is_cyl(s):
        return Geom_CylindricalSurface(gp_Ax3(_pnt(s.origin), _dir(s.axis)), float(s.radius))
    return Geom_Plane(gp_Pln(_pnt(s.origin), _dir(s.normal)))


def _intersection_curve(pts, sa, sb, tol):
    """Exact intersection curve of two analytic surfaces that the chain lies on, or None.
    Handles what the fast paths can't: ellipses from oblique cuts, plane/cone conics."""
    try:
        inter = GeomAPI_IntSS(_geom_surface(sa), _geom_surface(sb), tol)
    except Exception:
        return None
    if not inter.IsDone():
        return None
    for i in range(1, inter.NbLines() + 1):
        c = inter.Line(i)
        # IntSS hands back trimmed copies; trimming/reversing those misbehaves across the
        # trim boundary, so work on the underlying (periodic) ellipse/circle/line
        if isinstance(c, Geom_TrimmedCurve):  # OCP returns handles as their concrete type
            c = c.BasisCurve()
        ok = True
        for q in pts if len(pts) <= 64 else pts[:: max(1, len(pts) // 64)]:
            pr = GeomAPI_ProjectPointOnCurve(_pnt(q), c)
            if pr.NbPoints() == 0 or pr.LowerDistance() > tol:
                ok = False
                break
        if ok:
            return c
    return None


def _chain_geom(pts, sa, sb, tol, closed):
    """Decide the curve for a chain between surfaces sa and sb.
    Returns ('line',) | ('circle', centre, axis, radius) | ('curve', Geom_Curve) | ('poly',)
    or None when the chain can't lie on a curved neighbour (that neighbour gets demoted)."""
    if sa is sb and _is_curved(sa):
        # the cut between the two halves of a split region: any curve on the surface
        # joining the corners will do, so it needn't follow the mesh's zigzag
        return ("uvline", sa)
    curved = [s for s in (sa, sb) if _is_curved(s)]
    on_both = lambda: all(_surface_dist(pts, x).max() <= tol for x in (sa, sb) if x is not None)
    if not closed:
        d = _fit_line(pts, tol)
        if d is not None and all(_surface_dist(pts, c).max() <= tol for c in curved) \
                and all(abs(abs(d @ c.axis) - 1) < 1e-3 for c in curved if _is_cyl(c)):
            return ("line",)
    if not curved:
        return ("poly",)
    # from here the curve comes from the surfaces, so even a single mesh edge (2 points)
    # is enough; only the free circle fit below needs real support
    c0 = curved[0]
    circ = _circle_for(pts, c0, tol) if _kind(c0) in ("cylinder", "cone") else None
    if circ is not None and on_both() and all(
            _kind(c) not in ("cylinder", "cone") or abs(abs(c.axis @ c0.axis) - 1) < 1e-3
            for c in curved[1:]):
        return ("circle", circ[0], c0.axis, circ[1])
    # tangent blends (sphere/torus fillets) meet their neighbours in circles too, and
    # exact surface intersection is unreliable exactly there, so fit the circle directly
    gen = _fit_circle(pts, tol)
    if gen is not None and len(pts) >= 4 and on_both():
        return ("circle", gen[0], gen[1], gen[2])
    if sa is not None and sb is not None:
        c = _intersection_curve(pts, sa, sb, tol)
        if c is not None:
            return ("curve", c)
    return None


# ---------------------------------------------------------------- main entry

def rebuild(points, faces, labels, surfaces, tol, max_rounds=20):
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces)
    labels = np.asarray(labels).copy()
    surfaces = list(surfaces)
    he = _half_edges(faces)

    # every freeform triangle becomes its own region with no analytic surface
    free = np.nonzero(labels < 0)[0]
    labels[free] = len(surfaces) + np.arange(len(free))
    surfaces += [None] * len(free)

    def demote(regions):
        for r in regions:
            tris = np.nonzero(labels == r)[0]
            labels[tris] = len(surfaces) + np.arange(len(tris))
            surfaces.extend([None] * len(tris))

    n_split = len(_split_wrapping(points, faces, labels, surfaces, range(len(surfaces))))
    poly_regions = set()
    for _ in range(max_rounds):
        chains, bad = _build_chains(points, faces, labels, surfaces, he, tol, poly_regions)
        if bad:
            demote(bad)
            continue
        try:
            shape, verified = _assemble(points, faces, labels, surfaces, he, chains, tol)
            break
        except _ChainError as e:
            curved = [r for r in e.regions if _is_curved(surfaces[r])]
            fresh_poly = e.poly - poly_regions
            if not curved and not fresh_poly:
                raise
            demote(curved)
            poly_regions |= fresh_poly
    else:
        raise RebuildError("chain classification did not converge")

    if not verified and not is_valid(shape):
        raise RebuildError("rebuilt solid fails BRepCheck")
    # a valid compound of open shells still integrates to the right volume, so check
    # explicitly that every face ended up inside a solid
    if count_solid_faces(shape) != count_all_faces(shape):
        raise RebuildError("rebuild produced open shells, not solids")
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    want = mesh_volume(points, faces)
    # flat facets cut inside convex curves (and outside concave ones), so a correct
    # rebuild of a coarse mesh legitimately differs from it: allow the volume between
    # each curved region's facets and its true surface, on top of 2%
    allowance = 0.02 * abs(want) + 1.5 * _chord_volume(points, faces, labels, surfaces)
    if want <= 0 or abs(props.Mass() - want) > allowance:
        raise RebuildError(f"volume mismatch: rebuilt {props.Mass():.3f} vs mesh {want:.3f}")
    if n_split or count_all_faces(shape) <= REJOIN_FULL_LIMIT:
        shape = _rejoin(shape, tol)
    return shape


REJOIN_FULL_LIMIT = 5000  # faces; above this the final merge leaves planes alone


def _rejoin(shape, tol, angle_deg=0.5, fix_limit=REJOIN_FULL_LIMIT):
    """Merge split curved halves back into single faces (and, on small shapes, any
    near-coplanar facets the segmenter left apart), best effort: the input is a
    valid solid, so any merge that throws or comes out invalid is simply dropped.
    On big shapes planes are kept out of the merge (the segmenter already grouped them,
    give or take a stray coplanar facet) and no ShapeFix is attempted, which together
    cost seconds per 10k faces; small shapes get the full merge + ShapeFix rescue."""
    small = count_all_faces(shape) <= fix_limit
    keep = ShapeMap()
    exp = TopExp_Explorer(shape if not small else TopoDS_Compound(), TopAbs_FACE)
    while exp.More():
        f = exp.Current()
        if BRepAdaptor_Surface(TopoDS.Face(f)).GetType() == GeomAbs_Plane:
            keep.Add(f)
        exp.Next()
    # each attempt costs a full validity check, so big shapes get one try
    for unify_edges in (True, False) if small else (True,):
        try:
            # on a copy: a merge that throws can leave its input modified in place
            u = ShapeUpgrade_UnifySameDomain(BRepBuilderAPI_Copy(shape).Shape(), unify_edges, True, False)
            u.SetLinearTolerance(tol)
            u.SetAngularTolerance(np.radians(angle_deg))
            u.AllowInternalEdges(False)
            u.KeepShapes(keep)
            u.Build()
            merged = u.Shape()
        except Exception:
            continue
        candidates = [merged]
        if small:
            fix = ShapeFix_Shape(merged)
            fix.Perform()
            candidates.append(fix.Shape())
        for m in candidates:
            if is_valid(m) and count_solid_faces(m) == count_all_faces(m):
                return m
    return shape


def _chord_volume(points, faces, labels, surfaces):
    """Rough volume between the facets of curved regions and their true surfaces:
    facet area x distance of the facet centre from the surface. Each region's distances
    are capped at 4x the part's median, so a bogus fit (vertices of flat geometry that
    happen to lie on a cylinder, facets far inside it) can't widen its own allowance."""
    regions = []
    for r, tris in _region_index(labels).items():
        s = surfaces[r]
        if not _is_curved(s):
            continue
        p = points[faces[tris]]
        area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) / 2
        regions.append((area, _surface_dist(p.mean(axis=1), s)))
    if not regions:
        return 0.0
    typical = float(np.median([np.average(d, weights=a) for a, d in regions]))
    cap = 4 * typical if typical > 0 else np.inf
    return float(sum((a * np.minimum(d, cap)).sum() for a, d in regions))


def _split_wrapping(points, faces, labels, surfaces, regions):
    """Split those of `regions` that wrap all the way round their axis into two halves.
    Returns the set of regions that were split (each gains a twin at the end of surfaces).

    A face whose boundary loops each circle the axis needs a seam edge joining them, and
    ShapeFix_Face only manages that for plain circular ends (not ellipses, notches...).
    Two half-shells need no seam at all; the final same-domain merge rejoins them into
    one face. Triangles are split by centroid angle, which for the usual column-wise
    tessellation puts the cut along axial mesh edges; where it zigzags instead (cones,
    tori) the twin boundary is built as a straight line in UV, see _uv_edge."""
    done = set()
    for r in regions:
        s = surfaces[r]
        if _kind(s) == "sphere":
            if _split_sphere(points, faces, labels, surfaces, r):
                done.add(r)
            continue
        if _kind(s) not in ("cylinder", "cone", "torus"):
            continue
        tris = np.nonzero(labels == r)[0]
        if len(tris) < 4:
            continue
        origin = s.center if _kind(s) == "torus" else s.origin
        c = points[faces[tris]].mean(axis=1) - origin
        x = np.cross(s.axis, [1.0, 0, 0] if abs(s.axis[0]) < 0.9 else [0, 1.0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(s.axis, x)
        ang = np.sort(np.arctan2(c @ y, c @ x))
        gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * np.pi]))
        if gaps.max() > np.pi / 2:  # an open arc (fillet, partial wall): no seam trouble
            continue
        # cut in the gaps between triangle centroids — between facet columns, i.e. along
        # axial mesh edges — once at the widest gap and once at the gap nearest opposite
        mids = (ang + gaps / 2) % (2 * np.pi)
        cut1 = mids[np.argmax(gaps)]
        # only gaps between columns qualify: a quad split into two triangles leaves a
        # small gap *inside* its column, and cutting there would slice a facet diagonally
        between = gaps >= 0.6 * gaps.max()
        dist = np.abs(((mids - cut1) % (2 * np.pi)) - np.pi)
        cut2 = mids[np.argmin(np.where(between, dist, np.inf))]
        theta = np.arctan2(c @ y, c @ x)
        half = (theta - cut1) % (2 * np.pi) < (cut2 - cut1) % (2 * np.pi)
        if half.all() or not half.any():
            continue
        labels[tris[half]] = len(surfaces)
        surfaces.append(s)
        done.add(r)
    return done


def _split_sphere(points, faces, labels, surfaces, r, max_angle=np.radians(70), depth=3):
    """Bisect a large sphere patch until every piece stays within max_angle of its own
    centre direction. A sphere frame then always has both poles (90 degrees from the piece
    centre, see _make_face) outside the piece; a hemisphere dome otherwise has a pole on or
    inside it, which needs degenerate edges. Pieces meet along great-circle arcs."""
    s = surfaces[r]
    tris = np.nonzero(labels == r)[0]
    if depth == 0 or len(tris) < 8:
        return False
    d = points[faces[tris]].reshape(-1, 3) - s.center
    d /= np.linalg.norm(d, axis=1)[:, None]
    m = d.mean(axis=0)
    m /= np.linalg.norm(m)
    if np.arccos(np.clip(d @ m, -1, 1)).max() <= max_angle:
        return False
    # cut with the plane through the centre and m, across the widest spread
    c = points[faces[tris]].mean(axis=1) - s.center
    perp = c - np.outer(c @ m, m)
    _, _, vt = np.linalg.svd(perp, full_matrices=False)
    side = perp @ vt[0] > 0
    if side.all() or not side.any():
        return False
    twin = len(surfaces)
    labels[tris[side]] = twin
    surfaces.append(s)
    _split_sphere(points, faces, labels, surfaces, r, max_angle, depth - 1)
    _split_sphere(points, faces, labels, surfaces, twin, max_angle, depth - 1)
    return True


def _build_chains(points, faces, labels, surfaces, he, tol, poly_regions=frozenset()):
    """Group boundary edges into chains and classify them. Returns (chains, regions_to_demote)."""
    # every undirected edge once (a < b), with the regions on either side — vectorised,
    # since a freeform mesh has a region boundary on nearly every edge
    F = np.asarray(faces)
    a = F.ravel()
    b = np.roll(F, -1, axis=1).ravel()
    tri = np.repeat(np.arange(len(F)), 3)
    fwd = a < b
    # the twin half-edge (b, a) belongs to the triangle on the other side
    key_f = a[fwd].astype(np.int64) * len(points) + b[fwd]
    key_r = b[~fwd].astype(np.int64) * len(points) + a[~fwd]
    of, orv = np.argsort(key_f, kind="stable"), np.argsort(key_r, kind="stable")
    if len(of) != len(orv) or not np.array_equal(key_f[of], key_r[orv]):
        raise RebuildError("mesh is not watertight")
    ea, eb = a[fwd][of], b[fwd][of]
    ra = labels[tri[fwd][of]]
    rb = labels[tri[~fwd][orv]]
    bnd = ra != rb
    ea, eb, ra, rb = ea[bnd], eb[bnd], ra[bnd], rb[bnd]
    lo, hi = np.minimum(ra, rb), np.maximum(ra, rb)

    # corners: vertices touching three or more regions
    vr = np.concatenate([np.c_[ea, ra], np.c_[ea, rb], np.c_[eb, ra], np.c_[eb, rb]])
    vr = vr[np.lexsort((vr[:, 1], vr[:, 0]))]
    first = np.r_[True, (np.diff(vr, axis=0) != 0).any(axis=1)]
    v_ids = vr[first, 0]
    starts = np.r_[True, v_ids[1:] != v_ids[:-1]]
    counts = np.diff(np.r_[np.flatnonzero(starts), len(v_ids)])
    corners = set(v_ids[starts][counts >= 3].tolist())

    # group edges by region pair; a pair sharing one edge between two non-curved regions
    # (two facets, a facet and a plane...) is a one-edge straight chain: no walk, no fit
    order = np.lexsort((hi, lo))
    lo, hi, ea, eb = lo[order], hi[order], ea[order], eb[order]
    gstart = np.flatnonzero(np.r_[True, (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])])
    gend = np.r_[gstart[1:], len(lo)]
    curved = np.array([_is_curved(x) for x in surfaces], bool)
    chains = []
    pair_edges = {}
    for s0, s1 in zip(gstart.tolist(), gend.tolist()):
        pair = (int(lo[s0]), int(hi[s0]))
        if s1 - s0 == 1 and not curved[pair[0]] and not curved[pair[1]]:
            chains.append([pair, [int(ea[s0]), int(eb[s0])], ("line",)])
        else:
            pair_edges[pair] = list(zip(ea[s0:s1].tolist(), eb[s0:s1].tolist()))
    n_simple = len(chains)

    demote = set()
    for pair, edges in pair_edges.items():
        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        stops = corners | {v for v, n in adj.items() if len(n) != 2}
        seen = set()

        def walk(v0, v1):
            path = [v0, v1]
            seen.add((min(v0, v1), max(v0, v1)))
            while path[-1] not in stops and path[-1] != path[0]:
                prev, cur = path[-2], path[-1]
                nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
                seen.add((min(cur, nxt), max(cur, nxt)))
                path.append(nxt)
            return path

        for v in sorted(adj):
            if v in stops:
                for w in adj[v]:
                    if (min(v, w), max(v, w)) not in seen:
                        chains.append([pair, walk(v, w)])
        for a, b in edges:  # closed loops with no corner on them (a full hole edge)
            if (min(a, b), max(a, b)) not in seen:
                chains.append([pair, walk(a, b)])

    for ch in chains[n_simple:]:
        (ra, rb), path = ch
        closed = path[0] == path[-1]
        geom = _chain_geom(points[path], surfaces[ra], surfaces[rb], tol, closed)
        if geom is not None and geom[0] == "line" and (ra in poly_regions or rb in poly_regions):
            # a near-collinear chain straightened into a line can collapse a thin face
            geom = ("poly",)
        if geom is None:
            demote.update(r for r in (ra, rb) if _is_curved(surfaces[r]))
        ch.append(geom)
    return chains, demote


def _snap(p, surfs, iters=10):
    """Alternating projection of a corner onto all analytic surfaces it touches."""
    if not surfs:
        return p
    if len(surfs) == 1:  # a single projection is already exact
        iters = 1
    for _ in range(iters):
        for s in surfs:
            k = _kind(s)
            if k == "sphere":
                d = p - s.center
                n = np.linalg.norm(d)
                if n > 1e-12:
                    p = s.center + d * (s.radius / n)
            elif k == "torus":
                q = p - s.center
                h = q @ s.axis
                radial = q - h * s.axis
                rho = np.linalg.norm(radial)
                if rho > 1e-12:
                    m = s.center + radial * (s.major / rho)
                    d = p - m
                    n = np.linalg.norm(d)
                    if n > 1e-12:
                        p = m + d * (s.minor / n)
            elif k == "cone":
                off = p - s.origin
                h = off @ s.axis
                radial = off - h * s.axis
                rho = np.linalg.norm(radial)
                if rho > 1e-12:
                    g = np.array([np.cos(s.half_angle), np.sin(s.half_angle)])  # generator in (h, rho)
                    t = max(h * g[0] + rho * g[1], 0.0)
                    p = s.origin + t * g[0] * s.axis + radial * (t * g[1] / rho)
            elif _is_cyl(s):
                off = p - s.origin
                h = off @ s.axis
                radial = off - h * s.axis
                n = np.linalg.norm(radial)
                if n > 1e-12:
                    p = s.origin + h * s.axis + radial * (s.radius / n)
            else:
                p = p - ((p - s.origin) @ s.normal) * s.normal
    return p


def _assemble(points, faces, labels, surfaces, he, chains, tol):
    # snap chain end points (corners) onto their analytic surfaces
    touching = defaultdict(set)
    for (ra, rb), path, _ in chains:
        for v in (path[0], path[-1]):
            touching[v].update((ra, rb))
    pos = {}
    for v, rs in touching.items():
        surfs = [surfaces[r] for r in rs if surfaces[r] is not None]
        pos[v] = _snap(points[v].copy(), surfs) if surfs else points[v]

    vtx = {v: BRepBuilderAPI_MakeVertex(_pnt(p)).Vertex() for v, p in pos.items()}

    def vertex(v):
        if v not in vtx:
            vtx[v] = BRepBuilderAPI_MakeVertex(_pnt(points[v])).Vertex()
        return vtx[v]

    def P(v):
        return pos.get(v, points[v])

    # build one TopoDS edge list per chain, oriented along the chain path
    half_to_chain = {}  # directed mesh edge -> (chain index, forward?)
    chain_edges = []
    for ci, ((ra, rb), path, geom) in enumerate(chains):
        try:
            es = _chain_edges(geom, path, points, vertex, P)
        except RebuildError as e:
            raise _ChainError(str(e), (ra, rb)) from e
        chain_edges.append(es)
        for a, b in zip(path, path[1:]):
            half_to_chain[(a, b)] = (ci, True)
            half_to_chain[(b, a)] = (ci, False)
    return _faces_and_solid(points, faces, labels, surfaces, he, chains, chain_edges,
                            half_to_chain, tol)


def _chain_edges(geom, path, points, vertex, P):
    """The TopoDS edges for one chain, oriented along its path."""
    kind = geom[0]
    if kind == "line":
        es = [BRepBuilderAPI_MakeEdge(vertex(path[0]), vertex(path[-1])).Edge()]
    elif kind == "circle":
        _, centre, axis, radius = geom
        x = P(path[0]) - centre
        x -= (x @ axis) * axis
        x /= np.linalg.norm(x)
        circ = Geom_Circle(gp_Circ(gp_Ax2(_pnt(centre), _dir(axis), _dir(x)), float(radius)))
        es = [_trimmed_edge(circ, path, points, vertex, P)]
    elif kind == "curve":
        es = [_trimmed_edge(geom[1], path, points, vertex, P)]
    elif kind == "uvline" and _kind(geom[1]) == "sphere":
        es = [_great_arc(geom[1], vertex(path[0]), vertex(path[-1]), P(path[0]), P(path[-1]))]
    elif kind == "uvline":
        es = [_uv_edge(geom[1], vertex(path[0]), vertex(path[-1]), P(path[0]), P(path[-1]))]
    else:
        es = [BRepBuilderAPI_MakeEdge(vertex(a), vertex(b)).Edge() for a, b in zip(path, path[1:])]
    return es


def _faces_and_solid(points, faces, labels, surfaces, he, chains, chain_edges, half_to_chain, tol):
    # one shell per connected body; the mesh winding already orients every face
    # consistently, so the shells are assembled directly (ShapeFix_Shell's orientation
    # pass flips full-turn hole cylinders, whose UV boundary looks ambiguous to it)
    body = _bodies(faces, he)
    reshape = ShapeBuild_ReShape()
    builder = BRep_Builder()
    shells = {}
    made = []
    region_body = {}

    # curved regions and their neighbours are the only faces this construction can
    # spoil, so build and check them first: a failure then costs a retry before the
    # (often tens of thousands of) plain facets have been built for nothing
    neighbours = defaultdict(set)
    for (ra, rb), _, _ in chains:
        neighbours[ra].add(rb)
        neighbours[rb].add(ra)
    index = _region_index(labels)
    curvedish = {r for r in index if _is_curved(surfaces[r])}
    curvedish |= {x for r in curvedish for x in neighbours[r]}
    order = sorted(curvedish) + [r for r in index if r not in curvedish]
    blame, poly = set(), set()

    deferred = set()  # invalid planes, left to the whole-solid fix below

    def check_curvedish():
        for r, f in made:
            if not BRepCheck_Analyzer(reshape.Value(f)).IsValid():
                # a bad curved face is its own fault. A bad plane is left to the whole-solid
                # fix: blaming its curved neighbours right away would throw out every hole
                # in a wall over one fixable edge; they only get blamed if that fix fails
                (blame if _is_curved(surfaces[r]) else deferred).add(r)
        if blame:
            raise _ChainError(f"{len(blame)} curved faces came out invalid", tuple(blame))

    for k, r in enumerate(order):
        if k == len(curvedish):
            check_curvedish()
        tris = index[r]
        surf = surfaces[r]
        loops = _boundary_loops(faces, r, tris, he, labels)
        wires = []
        for loop in loops:
            mw = BRepBuilderAPI_MakeWire()
            n = len(loop)
            # rotate the loop so it starts at a chain boundary
            starts = [i for i in range(n) if _chain_starts_here(loop, i, half_to_chain, chains)]
            if not starts:
                raise RebuildError("loop without chain start")
            i0 = starts[0]
            seq = loop[i0:] + loop[:i0]
            i = 0
            while i < n:
                a, b = seq[i], seq[(i + 1) % n]
                ci, fwd = half_to_chain[(a, b)]
                path = chains[ci][1]
                es = chain_edges[ci]
                if fwd:
                    for e in es:
                        mw.Add(e)
                else:
                    for e in reversed(es):
                        mw.Add(TopoDS.Edge(e.Reversed()))
                i += len(path) - 1
            if not mw.IsDone():
                raise _ChainError(f"wire failed for region {r} ({mw.Error()})", (r,))
            wires.append((mw.Wire(), loop))

        face = _make_face(surf, wires, points, faces, tris, tol, reshape)
        made.append((r, face))
        b = int(body[tris[0]])
        region_body[r] = b
        if b not in shells:
            shells[b] = TopoDS_Shell()
            builder.MakeShell(shells[b])
        builder.Add(shells[b], face)

    # inward-wound bodies are internal voids: they go into the solid that contains them
    # as an extra shell, instead of becoming (negative) solids of their own
    tet = np.einsum("ij,ij->i", points[faces[:, 0]], np.cross(points[faces[:, 1]], points[faces[:, 2]]))
    body_vol = {b: float(tet[body == b].sum() / 6) for b in shells}
    outer = [b for b in shells if body_vol[b] > 0]
    solid_of = {}
    solids = []
    for b in outer:
        shells[b].Closed(True)
        so = TopoDS_Solid()
        builder.MakeSolid(so)
        builder.Add(so, shells[b])
        solid_of[b] = len(solids)
        solids.append(so)
    for b in shells:
        if b in solid_of:
            continue
        probe = points[faces[np.nonzero(body == b)[0][0], 0]]
        holders = [o for o in outer if _point_in_mesh(probe, points[faces[body == o]])]
        if not holders:
            raise RebuildError("inward-facing shell that isn't inside any body")
        # innermost container = the smallest one holding it
        o = min(holders, key=lambda x: body_vol[x])
        shells[b].Closed(True)
        builder.Add(solids[solid_of[o]], shells[b])
        solid_of[b] = solid_of[o]
    if len(solids) == 1:
        solid = solids[0]
    else:
        solid = TopoDS_Compound()
        builder.MakeCompound(solid)
        for so in solids:
            builder.Add(solid, so)
    solid = reshape.Apply(solid)

    # a face that came out invalid is almost always a curved surface whose boundary
    # didn't sit well on it (or a plane next to one): demote those and let rebuild() retry
    if len(order) == len(curvedish):
        check_curvedish()

    # each body is checked once; the whole-shape ShapeFix (seconds on big meshes) only
    # runs if a body actually needs it
    out = solid
    bad_solids = _bad_solids(out, len(solids))
    if bad_solids:
        # usually edges whose parametrisation/tolerances no longer match after corners
        # were snapped: re-syncing them is ~7x cheaper than a whole-shape ShapeFix
        BRepLib.SameParameter_s(out, tol, True)
        BRepLib.UpdateTolerances_s(out, False)
        bad_solids = _bad_solids(out, len(solids))
    if bad_solids:
        final = ShapeFix_Shape(solid)
        final.Perform()
        out = final.Shape()
        bad_solids = _bad_solids(out, len(solids))
    if deferred and bad_solids != set():
        # the solid couldn't be fixed with those planes as they are: now blame their
        # curved neighbours (the planes themselves get plain polyline edges)
        raise _ChainError(f"{len(deferred)} planar faces stayed invalid",
                          tuple(x for r in deferred for x in neighbours[r] if _is_curved(surfaces[x])),
                          deferred)
    if bad_solids is None:  # ShapeFix changed the body count: let rebuild() check it all
        return out, False
    # a body can come out invalid as a solid even with every face valid (closure,
    # orientation); demote that body's curved regions rather than lose every body
    bad_bodies = {b for b, i in solid_of.items() if i in bad_solids}
    if bad_bodies:
        # usually a thin planar face whose straightened edges collapsed: find the bad
        # faces; planes get polyline edges, curved faces (and a bad plane's curved
        # neighbours) get demoted
        for r, f in made:
            if region_body[r] in bad_bodies and not BRepCheck_Analyzer(reshape.Value(f)).IsValid():
                if _is_curved(surfaces[r]):
                    blame.add(r)
                else:
                    poly.add(r)
                    blame.update(x for x in neighbours[r] if _is_curved(surfaces[x]))
        if not blame and not poly:  # nothing face-level to blame: drop its curves
            blame = {r for r, b in region_body.items() if b in bad_bodies and _is_curved(surfaces[r])}
        raise _ChainError(f"{len(bad_bodies)} bodies came out invalid", tuple(blame), poly)
    return out, True


def _bad_solids(shape, expected):
    """Indices of invalid solids, or None if the shape doesn't have `expected` solids."""
    built = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        built.append(exp.Current())
        exp.Next()
    if len(built) != expected:
        return None
    return {i for i, so in enumerate(built) if not is_valid(so)}


def is_valid(shape):
    """BRepCheck of a whole shape, multithreaded (it dominates on 10k+ face shapes)."""
    return BRepCheck_Analyzer(shape, True, True).IsValid()


def _count(shape, kind):
    n, exp = 0, TopExp_Explorer(shape, kind)
    while exp.More():
        n += 1
        exp.Next()
    return n


def count_all_faces(shape):
    return _count(shape, TopAbs_FACE)


def count_solid_faces(shape):
    n, exp = 0, TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        n += _count(exp.Current(), TopAbs_FACE)
        exp.Next()
    return n


def _point_in_mesh(p, tris):
    """Ray-parity test of point p against a closed triangle mesh (T, 3, 3)."""
    d = np.array([0.5773, 0.5774, 0.5775])  # skewed so the ray avoids edges/vertices
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    h = np.cross(d, e2)
    a = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(a) > 1e-14
    f = np.where(ok, 1.0 / np.where(ok, a, 1), 0)
    s = p - tris[:, 0]
    u = f * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = f * (q @ d)
    t = f * np.einsum("ij,ij->i", e2, q)
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return bool(hit.sum() % 2)


def _bodies(faces, he):
    """Connected-component id per triangle (triangles joined across shared edges)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    t1 = np.fromiter((t for (a, b), (t, _) in he.items() if a < b), np.int64)
    t2 = np.fromiter((he[(b, a)][0] for (a, b), _ in he.items() if a < b), np.int64)
    graph = coo_matrix((np.ones(len(t1)), (t1, t2)), shape=(len(faces), len(faces)))
    return connected_components(graph, directed=False)[1]


def _polyline_dist(q, poly):
    """Distance from each point in q to the polyline poly."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(np.einsum("qij,ij->qi", q[:, None] - a, ab) / np.maximum((ab * ab).sum(1), 1e-30), 0, 1)
    closest = a + t[..., None] * ab
    return np.linalg.norm(q[:, None] - closest, axis=2).min(axis=1)


def _trimmed_edge(curve, path, points, vertex, P):
    """Edge along `curve` between the chain's end vertices, running the way the mesh chain
    runs. Both senses are tried and checked against the chain polyline, so a wrong guess
    (the long way round a circle, the wrong branch of an ellipse) can't slip through."""
    closed = path[0] == path[-1]
    poly = np.array([P(v) for v in path])
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    limit = 0.5 * seg.max() + 1e-6
    p0, p1 = P(path[0]), P(path[-1])
    best = None
    for c in (curve, curve.Reversed()):
        u0 = GeomAPI_ProjectPointOnCurve(_pnt(p0), c).LowerDistanceParameter()
        u1 = GeomAPI_ProjectPointOnCurve(_pnt(p1), c).LowerDistanceParameter()
        if c.IsPeriodic():
            T = c.Period()
            u1 = u0 + T if closed else u0 + (u1 - u0) % T
        if u1 <= u0:
            continue
        samples = np.array([c.Value(float(u)).Coord() for u in np.linspace(u0, u1, 9)[1:-1]])
        err = _polyline_dist(samples, poly).max()
        if closed:
            # both senses trace the same full circle, so the shape test can't tell them
            # apart: compare how far along each sense the chain's quarter point sits
            k = max(1, len(path) // 4)
            frac = np.cumsum(seg)[k - 1] / seg.sum()
            uq = GeomAPI_ProjectPointOnCurve(_pnt(poly[k]), c).LowerDistanceParameter()
            err += abs((uq - u0) % (u1 - u0) / (u1 - u0) - frac) * seg.sum()
        if best is None or err < best[0]:
            best = (err, c, u0, u1)
    if best is None or best[0] > limit:
        raise RebuildError("curve does not follow the mesh boundary")
    _, c, u0, u1 = best
    v0, v1 = vertex(path[0]), vertex(path[-1])
    # corners were snapped onto the surfaces, not onto this curve: let the vertices
    # absorb the small gap instead of failing point projection
    for v, u in ((v0, u0), (v1, u1)):
        gap = c.Value(float(u)).Distance(BRep_Tool.Pnt_s(v))
        if gap > BRep_Tool.Tolerance_s(v):
            BRep_Builder().UpdateVertex(v, gap * 1.01)
    mk = BRepBuilderAPI_MakeEdge(c, v0, v1, float(u0), float(u1))
    if not mk.IsDone():
        raise RebuildError(f"curve edge failed ({mk.Error()})")
    return mk.Edge()


def _great_arc(sph, v0, v1, p0, p1):
    """Shorter great-circle arc from p0 to p1: exactly on the sphere, and unlike a UV line
    it doesn't care where the parametrisation's poles are."""
    a, b = p0 - sph.center, p1 - sph.center
    n = np.cross(a, b)
    if np.linalg.norm(n) < 1e-12:
        raise RebuildError("degenerate great arc")
    ang = float(np.arctan2(np.linalg.norm(n), a @ b))
    circ = Geom_Circle(gp_Circ(gp_Ax2(_pnt(sph.center), _dir(n), _dir(a)), float(sph.radius)))
    for v, u in ((v0, 0.0), (v1, ang)):
        gap = circ.Value(u).Distance(BRep_Tool.Pnt_s(v))
        if gap > BRep_Tool.Tolerance_s(v):
            BRep_Builder().UpdateVertex(v, gap * 1.01)
    mk = BRepBuilderAPI_MakeEdge(circ, v0, v1, 0.0, ang)
    if not mk.IsDone():
        raise RebuildError(f"great arc failed ({mk.Error()})")
    return mk.Edge()


def _uv_edge(surf, v0, v1, p0, p1):
    """Edge from p0 to p1 that is a straight line in the surface's (u, v) space — exactly
    on the surface whatever its type, taking the short way across any periodic seam."""
    S = _geom_surface(surf)
    sas = ShapeAnalysis_Surface(S)
    a = sas.ValueOfUV(_pnt(p0), 1e-7)
    b = sas.ValueOfUV(_pnt(p1), 1e-7)
    du, dv = b.X() - a.X(), b.Y() - a.Y()
    if S.IsUPeriodic():
        T = S.UPeriod()
        du = (du + T / 2) % T - T / 2
    if S.IsVPeriodic():
        T = S.VPeriod()
        dv = (dv + T / 2) % T - T / 2
    length = float(np.hypot(du, dv))
    if length < 1e-12:
        raise RebuildError("degenerate UV edge")
    line = Geom2d_Line(gp_Pnt2d(a.X(), a.Y()), gp_Dir2d(du, dv))
    for v, (u, w) in ((v0, (a.X(), a.Y())), (v1, (a.X() + du, a.Y() + dv))):
        gap = S.Value(u, w).Distance(BRep_Tool.Pnt_s(v))
        if gap > BRep_Tool.Tolerance_s(v):
            BRep_Builder().UpdateVertex(v, gap * 1.01)
    mk = BRepBuilderAPI_MakeEdge(line, S, v0, v1, 0.0, length)
    if not mk.IsDone():
        raise RebuildError(f"UV edge failed ({mk.Error()})")
    edge = mk.Edge()
    BRepLib.BuildCurve3d_s(edge, 1e-6)
    return edge


def _seam_dir(pts, origin, axis):
    """X direction for a surface of revolution: opposite the patch, so a partial patch
    (fillet, arc) never straddles the parametric seam."""
    off = pts - origin
    radial = (off - np.outer(off @ axis, axis)).mean(axis=0)
    if np.linalg.norm(radial) < 1e-9:  # full revolution: any direction will do
        radial = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
    return -radial / np.linalg.norm(radial)


def _chain_starts_here(loop, i, half_to_chain, chains):
    a, b = loop[i], loop[(i + 1) % len(loop)]
    ci, fwd = half_to_chain[(a, b)]
    path = chains[ci][1]
    return a == (path[0] if fwd else path[-1])


def _make_face(surf, wires, points, faces, tris, tol, reshape):
    if surf is None:
        # freeform triangle: its own plane
        # plain floats: a numpy call per facet costs more than the facet itself
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = points[faces[tris[0]]].tolist()
        ux, uy, uz, vx, vy, vz = x1 - x0, y1 - y0, z1 - z0, x2 - x0, y2 - y0, z2 - z0
        surf_pln = gp_Pln(gp_Pnt(x0, y0, z0),
                          gp_Dir(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx))
        mk = BRepBuilderAPI_MakeFace(surf_pln, wires[0][0], True)
        if not mk.IsDone():
            raise RebuildError("facet face failed")
        return mk.Face()

    if not _is_curved(surf):
        pln = gp_Pln(_pnt(surf.origin), _dir(surf.normal))
        # outer loop = largest positive signed area about the outward normal
        def area(loop):
            q = points[loop]
            return np.cross(q, np.roll(q, -1, axis=0)).sum(axis=0) @ surf.normal
        order = sorted(range(len(wires)), key=lambda i: -area(wires[i][1]))
        mk = BRepBuilderAPI_MakeFace(pln, wires[order[0]][0], True)
        for i in order[1:]:
            mk.Add(wires[i][0])
        if not mk.IsDone():
            raise RebuildError("planar face failed")
        return _fix_face(mk.Face(), tol, reshape)

    # cylinder/cone: build on a surface whose natural normal matches the material normal,
    # so the mesh-derived wire orientation is already right. A left-handed frame
    # (YReverse) flips the natural normal to point at the axis, for holes/countersinks.
    region_pts = points[faces[tris]].reshape(-1, 3)
    k = _kind(surf)
    if k == "sphere":
        # put the poles and the seam away from the patch: axis across it, seam behind it
        mean_dir = region_pts.mean(axis=0) - surf.center
        mean_dir /= max(np.linalg.norm(mean_dir), 1e-12)
        zaxis = np.cross(mean_dir, [1.0, 0, 0] if abs(mean_dir[0]) < 0.9 else [0, 1.0, 0])
        ax3 = gp_Ax3(_pnt(surf.center), _dir(zaxis), _dir(-mean_dir))
    else:
        origin = surf.center if k == "torus" else surf.origin
        if k == "cone":
            # reference circle at the region's mean height keeps the apex out of the patch
            h = float(np.mean((region_pts - surf.origin) @ surf.axis))
            origin = surf.origin + h * surf.axis
        ax3 = gp_Ax3(_pnt(origin), _dir(surf.axis), _dir(_seam_dir(region_pts, origin, surf.axis)))
    if not surf.outward:
        ax3.YReverse()
    if k == "sphere":
        geom = Geom_SphericalSurface(ax3, float(surf.radius))
    elif k == "torus":
        geom = Geom_ToroidalSurface(ax3, float(surf.major), float(surf.minor))
    elif k == "cone":
        geom = Geom_ConicalSurface(ax3, float(surf.half_angle), float(h * np.tan(surf.half_angle)))
    else:
        geom = Geom_CylindricalSurface(ax3, float(surf.radius))
    b = BRep_Builder()
    face = TopoDS_Face()
    b.MakeFace(face, geom, tol)
    for w, _ in wires:
        b.Add(face, w)
    return _fix_face(face, tol, reshape)


def _fix_face(face, tol, reshape):
    """ShapeFix_Face (pcurves, seams) — but it replaces the face's edges with copies, which
    would disconnect the face from its neighbours. Recording the replacements in a shared
    ReShape context and applying that to the whole shell afterwards keeps edges shared.
    Returns the ORIGINAL face; the context maps it to the fixed one."""
    fx = ShapeFix_Face(face)
    fx.SetContext(reshape)
    fx.SetPrecision(tol)
    fx.Perform()
    return face
