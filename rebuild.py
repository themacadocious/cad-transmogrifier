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
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeWire,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.Geom import (
    Geom_Circle,
    Geom_ConicalSurface,
    Geom_CylindricalSurface,
    Geom_Plane,
    Geom_SphericalSurface,
    Geom_ToroidalSurface,
)
from OCP.GeomAPI import GeomAPI_IntSS, GeomAPI_ProjectPointOnCurve
from OCP.ShapeBuild import ShapeBuild_ReShape
from OCP.ShapeFix import ShapeFix_Face, ShapeFix_Shape
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Face, TopoDS_Shell, TopoDS_Solid
from OCP.gp import gp_Ax2, gp_Ax3, gp_Circ, gp_Dir, gp_Pln, gp_Pnt


class RebuildError(Exception):
    pass


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


def _boundary_loops(faces, region, he, labels):
    """Directed boundary half-edges of one region, linked into closed loops (vertex lists)."""
    tris = np.nonzero(labels == region)[0]
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
    probe = pts[[0, len(pts) // 2, -1]] if len(pts) > 2 else pts
    for i in range(1, inter.NbLines() + 1):
        c = inter.Line(i)
        ok = True
        for q in pts if len(pts) <= 64 else pts[:: max(1, len(pts) // 64)]:
            pr = GeomAPI_ProjectPointOnCurve(_pnt(q), c)
            if pr.NbPoints() == 0 or pr.LowerDistance() > tol:
                ok = False
                break
        if ok and len(probe):
            return c
    return None


def _chain_geom(pts, sa, sb, tol, closed):
    """Decide the curve for a chain between surfaces sa and sb.
    Returns ('line',) | ('circle', centre, axis, radius) | ('curve', Geom_Curve) | ('poly',)
    or None when the chain can't lie on a curved neighbour (that neighbour gets demoted)."""
    curved = [s for s in (sa, sb) if _is_curved(s)]
    on_both = lambda: all(_surface_dist(pts, x).max() <= tol for x in (sa, sb) if x is not None)
    if not closed:
        d = _fit_line(pts, tol)
        if d is not None and all(_surface_dist(pts, c).max() <= tol for c in curved) \
                and all(abs(abs(d @ c.axis) - 1) < 1e-3 for c in curved if _is_cyl(c)):
            return ("line",)
    if not curved:
        return ("poly",)
    if len(pts) < 3:
        return None
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

    for _ in range(max_rounds):
        chains, demote = _build_chains(points, faces, labels, surfaces, he, tol)
        if not demote:
            break
        for r in demote:
            tris = np.nonzero(labels == r)[0]
            labels[tris] = len(surfaces) + np.arange(len(tris))
            surfaces += [None] * len(tris)
    else:
        raise RebuildError("chain classification did not converge")

    shape = _assemble(points, faces, labels, surfaces, he, chains, tol)

    if not BRepCheck_Analyzer(shape).IsValid():
        raise RebuildError("rebuilt solid fails BRepCheck")
    # a valid compound of open shells still integrates to the right volume, so check
    # explicitly that every face ended up inside a solid
    if count_solid_faces(shape) != count_all_faces(shape):
        raise RebuildError("rebuild produced open shells, not solids")
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    want = mesh_volume(points, faces)
    if want <= 0 or abs(props.Mass() - want) / want > 0.02:
        raise RebuildError(f"volume mismatch: rebuilt {props.Mass():.3f} vs mesh {want:.3f}")
    return shape


def _build_chains(points, faces, labels, surfaces, he, tol):
    """Group boundary edges into chains and classify them. Returns (chains, regions_to_demote)."""
    # undirected boundary edges keyed by the (sorted) region pair
    pair_edges = defaultdict(list)
    vert_regions = defaultdict(set)
    for (a, b), (t, _) in he.items():
        if a > b:
            continue
        ra, rb = int(labels[t]), int(labels[he[(b, a)][0]])
        if ra != rb:
            pair_edges[(min(ra, rb), max(ra, rb))].append((a, b))
            vert_regions[a].update((ra, rb))
            vert_regions[b].update((ra, rb))
    corners = {v for v, rs in vert_regions.items() if len(rs) >= 3}

    chains, demote = [], set()
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

    for ch in chains:
        (ra, rb), path = ch
        closed = path[0] == path[-1]
        geom = _chain_geom(points[path], surfaces[ra], surfaces[rb], tol, closed)
        if geom is None:
            demote.update(r for r in (ra, rb) if _is_curved(surfaces[r]))
        ch.append(geom)
    return chains, demote


def _snap(p, surfs, iters=10):
    """Alternating projection of a corner onto all analytic surfaces it touches."""
    if not surfs:
        return p
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
        pos[v] = _snap(points[v].copy(), [surfaces[r] for r in rs if surfaces[r] is not None])

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
        kind = geom[0]
        if kind == "line":
            es = [BRepBuilderAPI_MakeEdge(vertex(path[0]), vertex(path[-1])).Edge()]
        elif kind == "circle":
            _, centre, axis, radius = geom
            p0 = P(path[0])
            x = p0 - centre
            x -= (x @ axis) * axis
            x /= np.linalg.norm(x)
            # pick the circle sense so the arc passes through the chain's middle point
            mid = points[path[len(path) // 2]] - centre
            y = np.cross(axis, x)
            ang_mid = np.arctan2(mid @ y, mid @ x) % (2 * np.pi)
            if path[0] == path[-1]:
                second = points[path[1]] - centre
                forward = (np.arctan2(second @ y, second @ x) % (2 * np.pi)) < np.pi
            else:
                end = P(path[-1]) - centre
                ang_end = np.arctan2(end @ y, end @ x) % (2 * np.pi)
                forward = ang_mid < ang_end
            ax = axis if forward else -axis
            circ = Geom_Circle(gp_Circ(gp_Ax2(_pnt(centre), _dir(ax), _dir(x)), float(radius)))
            if path[0] == path[-1]:
                u1 = 2 * np.pi
            else:
                u1 = ang_end if forward else 2 * np.pi - ang_end
            # the corners were snapped onto the surfaces, not onto this fitted circle, so
            # let the vertices absorb the small gap instead of failing point projection
            v0, v1 = vertex(path[0]), vertex(path[-1])
            for v, u in ((v0, 0.0), (v1, u1)):
                gap = circ.Value(float(u)).Distance(BRep_Tool.Pnt_s(v))
                if gap > BRep_Tool.Tolerance_s(v):
                    BRep_Builder().UpdateVertex(v, gap * 1.01)
            mk = BRepBuilderAPI_MakeEdge(circ, v0, v1, 0.0, float(u1))
            if not mk.IsDone():
                raise RebuildError(f"arc edge failed ({mk.Error()})")
            es = [mk.Edge()]
        elif kind == "curve":
            es = [_curve_edge(geom[1], vertex(path[0]), vertex(path[-1]), P(path[0]), P(path[-1]),
                              points[path[len(path) // 2]], path[0] == path[-1])]
        else:
            es = [BRepBuilderAPI_MakeEdge(vertex(a), vertex(b)).Edge() for a, b in zip(path, path[1:])]
        chain_edges.append(es)
        for a, b in zip(path, path[1:]):
            half_to_chain[(a, b)] = (ci, True)
            half_to_chain[(b, a)] = (ci, False)

    # one shell per connected body; the mesh winding already orients every face
    # consistently, so the shells are assembled directly (ShapeFix_Shell's orientation
    # pass flips full-turn hole cylinders, whose UV boundary looks ambiguous to it)
    body = _bodies(faces, he)
    reshape = ShapeBuild_ReShape()
    builder = BRep_Builder()
    shells = {}

    for r in np.unique(labels).tolist():
        surf = surfaces[r]
        loops = _boundary_loops(faces, r, he, labels)
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
                raise RebuildError(f"wire failed for region {r} ({mw.Error()})")
            wires.append((mw.Wire(), loop))

        face = _make_face(surf, wires, points, faces, labels, r, tol, reshape)
        b = int(body[np.nonzero(labels == r)[0][0]])
        if b not in shells:
            shells[b] = TopoDS_Shell()
            builder.MakeShell(shells[b])
        builder.Add(shells[b], face)

    solids = []
    for sh in shells.values():
        sh.Closed(True)
        so = TopoDS_Solid()
        builder.MakeSolid(so)
        builder.Add(so, sh)
        solids.append(so)
    if len(solids) == 1:
        solid = solids[0]
    else:
        solid = TopoDS_Compound()
        builder.MakeCompound(solid)
        for so in solids:
            builder.Add(solid, so)
    solid = reshape.Apply(solid)
    final = ShapeFix_Shape(solid)
    final.Perform()
    return final.Shape()


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


def _bodies(faces, he):
    """Connected-component id per triangle."""
    parent = np.arange(len(faces))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), (t, _) in he.items():
        u, v = find(t), find(he[(b, a)][0])
        if u != v:
            parent[u] = v
    return np.array([find(t) for t in range(len(faces))])


def _curve_edge(curve, v0, v1, p0, p1, pmid, closed):
    """Edge along `curve` from p0 to p1 passing through pmid (reversing the curve if needed)."""
    for c in (curve, curve.Reversed()):
        u0 = GeomAPI_ProjectPointOnCurve(_pnt(p0), c).LowerDistanceParameter()
        um = GeomAPI_ProjectPointOnCurve(_pnt(pmid), c).LowerDistanceParameter()
        u1 = GeomAPI_ProjectPointOnCurve(_pnt(p1), c).LowerDistanceParameter()
        if c.IsPeriodic():
            T = c.Period()
            um = u0 + (um - u0) % T
            u1 = u0 + T if closed else u0 + (u1 - u0) % T
        if u0 < um < u1:
            mk = BRepBuilderAPI_MakeEdge(c, v0, v1, u0, u1)
            if mk.IsDone():
                return mk.Edge()
    raise RebuildError("could not trim intersection curve")


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


def _make_face(surf, wires, points, faces, labels, r, tol, reshape):
    if surf is None:
        # freeform triangle: its own plane
        t = faces[np.nonzero(labels == r)[0][0]]
        p = points[t]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        surf_pln = gp_Pln(_pnt(p[0]), _dir(n / np.linalg.norm(n)))
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
            raise RebuildError(f"planar face failed for region {r}")
        return _fix_face(mk.Face(), tol, reshape)

    # cylinder/cone: build on a surface whose natural normal matches the material normal,
    # so the mesh-derived wire orientation is already right. A left-handed frame
    # (YReverse) flips the natural normal to point at the axis, for holes/countersinks.
    region_pts = points[faces[labels == r]].reshape(-1, 3)
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
