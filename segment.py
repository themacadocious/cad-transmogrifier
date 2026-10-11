"""Mesh segmentation into analytic surfaces (planes, cylinders, cones).

This is the recognition half of "prismatic" mesh -> B-rep conversion: every triangle of a
welded, consistently wound mesh is assigned to a fitted surface so that holes, bosses,
fillets and countersinks can be rebuilt as real cylinders / cones instead of facet strips.

Pipeline
  1. planar units: edge-connected groups of triangles that lie on one plane within `tol`
     (normals within `angle_deg`).  A coarse cylinder facet strip is one small unit, a
     real flat face is one big unit.
  2. cylinders: region growing over *units* (a unit is accepted whole or not at all, so a
     flat face tangent to a fillet is never swallowed: its far vertices are off the
     cylinder).  Seeds are units whose two nearest-normal neighbours give the same
     axis (n_u x n_v), so triples of facets around a common axis.
  3. cones: same growth, seeded from triples of facets whose normals lie on a small circle
     of the Gauss sphere and whose planes meet in a common apex.
     In both passes a region whose ends run smoothly on into more curved surface may be a
     band of a torus / sphere (finely tessellated, several facet rows of those fit a
     cylinder or cone within tol): a torus / sphere is tried from it at once, and failing
     that it is set aside until after steps 4-5 (accept_or_defer, doubly_from).
  4. tori, seeded from 2-ring patches (closed form screened for all seeds at once).
  5. spheres, seeded from facet triples on one sphere that more facets around support.
     Cones, spheres and tori must fit tightly on meshes that are exact tessellations
     (_reject_loose): threads and organic shapes only fit them approximately.
  6. set-aside regions nobody claimed are accepted after all; merge of same surfaces;
     full-turn polygonal 'cylinders' are dropped when the mesh proves finer tessellation.
  7. whatever planar units are left (>= 2 triangles) become planes; single triangles that
     match nothing get label -1.

Every accepted region is re-grown from its seed with the final fitted parameters held
fixed, so all its vertices are within `tol` of the reported surface and it is
edge-connected by construction.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class Plane:
    origin: np.ndarray   # (3,) a point on the plane
    normal: np.ndarray   # (3,) unit, pointing OUT of the solid (agrees with triangle winding normals)


@dataclass
class Cylinder:
    origin: np.ndarray   # (3,) a point on the axis
    axis: np.ndarray     # (3,) unit
    radius: float
    outward: bool        # True if surface normals point away from the axis (boss/outer wall), False for holes


@dataclass
class Cone:
    origin: np.ndarray   # (3,) the APEX
    axis: np.ndarray     # (3,) unit, pointing from the apex toward the OPENING (radius grows along +axis)
    half_angle: float    # radians, 0 < half_angle < pi/2; radius at height h along axis = h*tan(half_angle)
    outward: bool        # True if facet normals point away from the axis (e.g. a cone boss), False for a countersink/chamfered hole


@dataclass
class Sphere:
    center: np.ndarray   # (3,)
    radius: float
    outward: bool        # normals point away from center (convex corner blend) vs toward (concave)


@dataclass
class Torus:
    center: np.ndarray   # (3,)
    axis: np.ndarray     # (3,) unit
    major: float         # distance from axis to the tube centre circle
    minor: float         # tube radius
    outward: bool        # normals point away from the tube centre circle


# curved regions are grown with a tighter tolerance than `tol`: the vertices of a
# tangent neighbour (flat face next to a fillet, sphere corner after a fillet) leave the
# surface quadratically, so a tighter band stops growth closer to the true boundary
GROW_TOL_FRAC = 0.5
UNIT_TOL_FRAC = 0.05
UNIT_ANGLE_DEG = 0.02
MIN_SPAN_DOUBLE_DEG = 40.0
# a sphere seed triple needs this many more facets around it on the same sphere
MIN_SPHERE_SUPPORT = 3
# Facet normal directions closer than MIN_DIR_SEP_DEG count as one (coplanar facets agree
# to ~1e-4 deg; genuine tessellation steps in the 65-mesh survey are >= 1 deg). A curved
# region needs at least MIN_FACET_DIRS of them: one direction is a flat patch. (Two -- a
# 2-facet arc -- are only taken for a fillet when a neighbour continues it tangentially,
# see _CylModel._tangent_neighbour.)
MIN_FACET_DIRS = 2
MIN_DIR_SEP_DEG = 0.5
# Misfit rejection (_Seg.reject_misfits): facet-centre deviation from the fitted surface
MISFIT_UNIT_RATIO = 4.0   # one facet vs the region's median facet (cylinders, cones)
MISFIT_PART_RATIO = 20.0  # region vs the part's typical curved region
MISFIT_MIN = 0.1          # mm, and at least this
MISFIT_ABS = 0.25         # mm, when there is nothing to compare with
# Polygonal prism vs coarse cylinder (reject_polygon_prisms). From a survey of 233
# full-turn cylinders in 65 real-world STLs (Fusion / Onshape / slicer exports): the
# coarsest had 20 facets per turn (18 degree steps) and the largest sagitta 0.05 mm;
# within a part, curved faces' sagittas agree to ~2-3x.
POLY_MAX_FACETS = 18       # only full turns this coarse (facets per turn) are ever in doubt
POLY_SAG_RATIO = 4.0       # x the part's typical sagitta
POLY_ABS_SAG = 0.25        # mm, no reference curvature: 5x the largest seen in the survey
POLY_ABS_FACETS = 12       # ... and at most this many facets per turn (none seen below 20)


# ------------------------------------------------------------------ small helpers

def _unit(v):
    v = np.asarray(v, float)
    return v / np.sqrt(v @ v)


def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _basis(a):
    """Two unit vectors completing `a` to a right-handed orthonormal frame."""
    h = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = _unit(_cross(a, h))
    return e1, _cross(a, e1)


def _canon(d):
    """Deterministic sign for an axis direction (largest |component| positive)."""
    return d if d[np.argmax(np.abs(d))] > 0 else -d


def _ranges(ptr, sel):
    """Concatenate CSR rows `sel`: returns (positions into the CSR index array, row lengths)."""
    starts = ptr[sel]
    lens = ptr[sel + 1] - starts
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, np.int64), lens
    shift = np.repeat(starts - (np.cumsum(lens) - lens), lens)
    return np.arange(total) + shift, lens


def _seg_starts(lens):
    return np.concatenate(([0], np.cumsum(lens)[:-1]))


def _uniq(a):
    """Sorted unique values of an integer array. Same result as np.unique, but numpy 2.x
    np.unique uses a hash table for integers that is up to ~100x slower than sorting
    on the large index arrays built here."""
    a = np.sort(np.asarray(a).ravel())
    if len(a) < 2:
        return a
    return a[np.concatenate(([True], a[1:] != a[:-1]))]


def _uniq_first(a):
    """(_uniq(a), index of the first occurrence of each value)."""
    a = np.asarray(a).ravel()
    o = np.argsort(a, kind="stable")
    b = a[o]
    m = np.concatenate(([True], b[1:] != b[:-1])) if len(b) else np.zeros(0, bool)
    return b[m], o[m]


def _span(theta):
    """Angular coverage (rad) of a set of azimuths: 2*pi minus the largest circular gap."""
    if len(theta) < 2:
        return 0.0
    t = np.sort(np.mod(theta, 2 * np.pi))
    gaps = np.diff(np.concatenate((t, [t[0] + 2 * np.pi])))
    return float(2 * np.pi - gaps.max())


def _n_dirs(theta, gap):
    """Number of distinct azimuth clusters (separated by more than `gap` rad)."""
    if len(theta) == 0:
        return 0
    t = np.sort(np.mod(theta, 2 * np.pi))
    d = np.diff(np.concatenate((t, [t[0] + 2 * np.pi])))
    k = int((d > gap).sum())
    return max(k, 1)


def _solve3(A, b):
    """Solution of the symmetric 3x3 system A x = b by cofactors in plain floats (called
    a few hundred thousand times on freeform meshes, where numpy's per-call overhead
    dominates); None if (nearly) singular."""
    a00, a01, a02 = float(A[0][0]), float(A[0][1]), float(A[0][2])
    a11, a12, a22 = float(A[1][1]), float(A[1][2]), float(A[2][2])
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    c00 = a11 * a22 - a12 * a12
    c01 = a02 * a12 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    det = a00 * c00 + a01 * c01 + a02 * c02
    scale = max(abs(a00), abs(a11), abs(a22), 1e-300) ** 3
    if not abs(det) > 1e-14 * scale:
        return None
    c11 = a00 * a22 - a02 * a02
    c12 = a01 * a02 - a00 * a12
    c22 = a00 * a11 - a01 * a01
    x = np.array([c00 * b0 + c01 * b1 + c02 * b2,
                  c01 * b0 + c11 * b1 + c12 * b2,
                  c02 * b0 + c12 * b1 + c22 * b2]) / det
    return x if np.all(np.isfinite(x)) else None


def _circle_fit(xy):
    """Algebraic (Kasa) circle fit followed by a few geometric Gauss-Newton steps."""
    m = xy.mean(0)
    q = xy - m
    n = len(q)
    x, y = q[:, 0], q[:, 1]
    M = np.empty((3, n))
    M[0], M[1], M[2] = x, y, 1.0
    sol = _solve3(M @ M.T, -(M @ (x * x + y * y)))
    if sol is None:
        sol, *_ = np.linalg.lstsq(M.T, -(x * x + y * y), rcond=None)
    c = -sol[:2] / 2
    r2 = c @ c - sol[2]
    if not np.isfinite(r2) or r2 <= 0:
        return None
    r = np.sqrt(r2)
    for _ in range(3):
        d = q - c
        rho = np.sqrt(np.einsum("ij,ij->i", d, d))
        if np.any(rho < 1e-12):
            break
        # J = -[u_x, u_y, 1]: solve J^T J s = -J^T res
        M[0], M[1] = d[:, 0] / rho, d[:, 1] / rho
        step = _solve3(M @ M.T, M @ (rho - r))
        if step is None:
            step, *_ = np.linalg.lstsq(-M.T, -(rho - r), rcond=None)
        c = c + step[:2]
        r = r + step[2]
        if abs(step[0]) + abs(step[1]) + abs(step[2]) <= 1e-13 * abs(r):
            break   # converged to rounding
    if not np.isfinite(r) or r <= 0:
        return None
    return c + m, float(r)


def _lm(resfun, nparam, iters=10, jac=None, done=0.0):
    """Tiny Levenberg-Marquardt on a residual function of a parameter offset vector.
    `jac(x)` gives the Jacobian analytically; without it, forward differences. Stops
    once the squared residual sum is <= `done`."""
    x = np.zeros(nparam)
    r = resfun(x)
    cost = r @ r
    lam = 1e-6
    h = 1e-7
    for _ in range(iters):
        if cost <= done:
            break
        if jac is not None:
            J = jac(x)
        else:
            J = np.empty((len(r), nparam))
            for j in range(nparam):
                dx = np.zeros(nparam)
                dx[j] = h
                J[:, j] = (resfun(x + dx) - r) / h
        JtJ = J.T @ J
        g = J.T @ r
        improved = False
        for _ in range(6):
            try:
                step = -np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-12), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            rn = resfun(x + step)
            cn = rn @ rn
            if np.isfinite(cn) and cn < cost:
                # converged once a step no longer buys anything measurable
                improved = cn < cost * (1 - 1e-6)
                x, r, cost = x + step, rn, cn
                lam = max(lam / 10, 1e-9)
                break
            lam *= 10
        if not improved or np.linalg.norm(step) < 1e-12:
            break
    return x


def _group_sphere_fit(X, own, n):
    """Algebraic sphere fit per group (rows of X with the same `own` in 0..n-1).
    Returns (ok, centre, radius, max |residual|) per group."""
    cnt = np.bincount(own, minlength=n).astype(float)
    mean = np.stack([np.bincount(own, X[:, k], n) for k in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    Q = X - mean[own]
    z = (Q * Q).sum(1)
    cols = (Q[:, 0], Q[:, 1], Q[:, 2], np.ones(len(Q)))
    A = np.empty((n, 4, 4))
    b = np.empty((n, 4))
    for i in range(4):
        b[:, i] = -np.bincount(own, cols[i] * z, n)
        for j in range(i, 4):
            A[:, i, j] = A[:, j, i] = np.bincount(own, cols[i] * cols[j], n)
    scale = np.abs(A).max(axis=(1, 2)) + 1e-300
    ok = np.abs(np.linalg.det(A / scale[:, None, None])) > 1e-14
    A[~ok] = np.eye(4)
    sol = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    c = -sol[:, :3] / 2
    r2 = (c * c).sum(1) - sol[:, 3]
    ok &= r2 > 0
    r = np.sqrt(np.maximum(r2, 0))
    res = np.abs(np.linalg.norm(Q - c[own], axis=1) - r[own])
    mx = np.zeros(n)
    np.maximum.at(mx, own, res)
    return ok, c + mean, r, mx


# A tessellation exported from CAD has its vertices ON the surfaces (residual a tiny
# fraction of tol); threads and organic shapes only ever fit a cone / sphere / torus
# approximately, using up a good part of the tolerance band. How much that says depends
# on the mesh, and its cylinders (found first, and reliably) tell: exact tessellation
# (at least EXACT_MESH_SHARE of them within EXACT_MESH_FRAC tol) -> a loose fit is
# rejected outright; plenty of loose cylinders too (mixed or lossy mesh: some exporters
# approximate fillets) -> never; too few cylinders to judge by -> only a partial patch
# (small angular extent) is.
LOOSE_FIT_FRAC = 0.2
EXACT_MESH_FRAC = 0.05
EXACT_MESH_SHARE = 0.75
TIGHT_REGROW_FRAC = 0.1   # band for re-growing a loose fit on an exact mesh


def _exact_mesh(s):
    """True / False when the cylinders found so far say the mesh is / isn't an exact
    tessellation; None when there are too few to tell."""
    n = len(s.cyl_res)
    if n >= 2 and float(np.mean(np.array(s.cyl_res) <= EXACT_MESH_FRAC)) >= EXACT_MESH_SHARE:
        return True
    return False if n >= 6 else None


def _reject_loose(model, X, partial):
    if float(np.abs(model.residual(X)).max()) <= LOOSE_FIT_FRAC * model.s.tol:
        return False
    exact = _exact_mesh(model.s)
    return partial if exact is None else exact


def _spread_at_least(nrm, cos_thr):
    """min over pairs of nrm[i] @ nrm[j] <= cos_thr (i.e. some two unit normals are at least
    arccos(cos_thr) apart), without forming the full Gram matrix for big sets."""
    n = len(nrm)
    if n < 2:
        return False
    if n <= 256:
        return bool(np.min(nrm @ nrm.T) <= cos_thr)
    m = nrm.sum(0)
    ln = np.sqrt(m @ m)
    if ln > 1e-12:
        c = m / ln
        cmin = float(np.min(nrm @ c))
        # every normal is within arccos(cmin) of c: the diameter is at most twice that
        if cmin > -1 and 2 * np.arccos(min(cmin, 1.0)) < np.arccos(np.clip(cos_thr, -1, 1)) - 1e-9:
            return False
    i0 = int(np.argmin(nrm @ (c if ln > 1e-12 else nrm[0])))
    d = nrm @ nrm[i0]
    if d.min() <= cos_thr:
        return True
    i1 = int(np.argmin(d))
    if np.min(nrm @ nrm[i1]) <= cos_thr:
        return True
    for k in range(0, n, 1024):
        if np.min(nrm[k:k + 1024] @ nrm.T) <= cos_thr:
            return True
    return False


# ------------------------------------------------------------------ adjacency

def edge_adjacency(faces):
    """(F,3) index of the triangle across each edge, edge k = (faces[i,k], faces[i,(k+1)%3]).
    -1 where the edge is a boundary or is shared by more than two triangles."""
    faces = np.asarray(faces, np.int64)
    F = len(faces)
    if F == 0:
        return np.zeros((0, 3), np.int64)
    a = faces
    b = np.roll(faces, -1, axis=1)
    lo = np.minimum(a, b).ravel()
    hi = np.maximum(a, b).ravel()
    key = lo * (int(faces.max()) + 1) + hi
    order = np.argsort(key, kind="stable")
    ks = key[order]
    start = np.concatenate(([True], ks[1:] != ks[:-1]))
    gid = np.cumsum(start) - 1
    cnt = np.bincount(gid)[gid]
    nbr = np.full(F * 3, -1, np.int64)
    first = np.nonzero(start & (cnt == 2))[0]
    e1, e2 = order[first], order[first + 1]
    nbr[e1] = e2 // 3
    nbr[e2] = e1 // 3
    return nbr.reshape(F, 3)


# ------------------------------------------------------------------ planar units

def _planar_units(P, n, area, nbr, tol, angle_deg):
    """Partition triangles into edge-connected planar groups. Returns unit id per triangle."""
    F = len(n)
    cosang = np.cos(np.radians(angle_deg))
    rows = np.repeat(np.arange(F), 3)
    cols = nbr.ravel()
    ok = cols >= 0
    rows, cols = rows[ok], cols[ok]
    ok = np.einsum("ij,ij->i", n[rows], n[cols]) >= cosang
    g = coo_matrix((np.ones(ok.sum(), np.int8), (rows[ok], cols[ok])), shape=(F, F)).tocsr()
    ncomp, comp = connected_components(g, directed=False)

    # fast path: a whole component that is flat (area-weighted normal + centroid fit)
    w = area[:, None]
    msum = np.zeros((ncomp, 3))
    np.add.at(msum, comp, n * w)
    csum = np.zeros((ncomp, 3))
    np.add.at(csum, comp, P.mean(1) * w)
    asum = np.bincount(comp, area, ncomp)
    mn = msum / np.maximum(np.linalg.norm(msum, axis=1), 1e-300)[:, None]
    cen = csum / np.maximum(asum, 1e-300)[:, None]
    dist = np.abs(np.einsum("ikj,ij->ik", P - cen[comp][:, None, :], mn[comp])).max(1)
    nok = np.einsum("ij,ij->i", n, mn[comp]) >= cosang
    good = np.ones(ncomp, bool)
    np.logical_and.at(good, comp, (dist <= tol) & nok)

    unit = np.full(F, -1, np.int64)
    unit[good[comp]] = comp[good[comp]]
    next_id = ncomp

    bad = np.nonzero(~good)[0]
    if len(bad):
        order = np.argsort(comp, kind="stable")
        cptr = np.searchsorted(comp[order], np.arange(ncomp + 1))
        for c in bad:
            tris = order[cptr[c]:cptr[c + 1]]
            # seeds: biggest triangles first (deterministic tie-break by index)
            for s in tris[np.lexsort((tris, -area[tris]))]:
                if unit[s] >= 0:
                    continue
                reg = _grow_plane(s, P, n, nbr, comp, unit, tol, cosang)
                unit[reg] = next_id
                next_id += 1
    # compact ids
    _, unit = np.unique(unit, return_inverse=True)
    return unit


def _grow_plane(seed, P, n, nbr, comp, unit, tol, cosang, fixed=None):
    c = comp[seed]
    inreg = {int(seed)}
    reg = [int(seed)]
    if fixed is None:
        pts = P[seed]
        S1 = pts.sum(0)
        S2 = pts.T @ pts
        N = 3
        nrm, org = n[seed].copy(), pts.mean(0)
    else:
        nrm, org = fixed
    new = np.array([seed])
    while True:
        cand = nbr[new].ravel()
        cand = _uniq(cand[cand >= 0])
        cand = cand[(comp[cand] == c) & (unit[cand] < 0)]
        cand = np.array([t for t in cand.tolist() if t not in inreg], np.int64)
        if len(cand) == 0:
            break
        ok = (n[cand] @ nrm >= cosang) & (np.abs((P[cand] - org) @ nrm).max(1) <= tol)
        for t in cand.tolist():
            inreg.add(t)  # rejected ones too: don't retest
        acc = cand[ok]
        if len(acc) == 0:
            break
        reg.extend(acc.tolist())
        new = acc
        if fixed is None:
            pts = P[acc].reshape(-1, 3)
            S1 = S1 + pts.sum(0)
            S2 = S2 + pts.T @ pts
            N += len(pts)
            org = S1 / N
            C = S2 / N - np.outer(org, org)
            v = np.linalg.eigh(C)[1][:, 0]
            nrm = v if v @ n[seed] > 0 else -v
    reg = np.array(reg)
    if fixed is None:
        pts = P[reg].reshape(-1, 3)
        if np.abs((pts - org) @ nrm).max() > tol:
            # refits drifted: re-grow with the final plane held fixed
            return _grow_plane(seed, P, n, nbr, comp, unit, tol, cosang, fixed=(nrm, org))
    return reg


# ------------------------------------------------------------------ surface models

class _Model:
    """Base: fitted analytic surface tested against whole planar units."""

    def __init__(self, seg):
        self.s = seg

    def unit_points(self, units):
        pos, lens = _ranges(self.s.vptr, units)
        return self.s.points[self.s.vidx[pos]], lens

    def azimuth(self, X, base=None):
        v = X - (base if base is not None else self.c)
        return np.arctan2(v @ self.e2, v @ self.e1)

    def test(self, units):
        """Bool mask: unit lies on the surface (normals, vertices, facet width)."""
        s = self.s
        nrm = s.un[units]
        ok = self.normal_ok(nrm, s.ucen[units])
        if not ok.any():
            return ok
        sub = units[ok]
        X, lens = self.unit_points(sub)
        res = np.abs(self.residual(X))
        st = _seg_starts(lens)
        good = np.maximum.reduceat(res, st) <= s.grow_tol
        # facet angular width around the axis must be at most max_step (and not much more
        # than the region's own facets, once that is known: see try_region)
        good &= self._widths(sub, X, lens) <= min(s.max_step, self.wmax)
        ok[np.nonzero(ok)[0][~good]] = False
        return ok

    wmax = np.inf

    def _widths(self, units, X=None, lens=None):
        """Angular width (rad) of each unit around the axis."""
        if X is None:
            X, lens = self.unit_points(units)
        st = _seg_starts(lens)
        az = self.azimuth(X)
        az0 = np.repeat(self.azimuth(self.s.ucen[units]), lens)
        d = np.mod(az - az0 + np.pi, 2 * np.pi) - np.pi
        return np.maximum.reduceat(d, st) - np.minimum.reduceat(d, st)

    def limit_width(self, units):
        """A shallow D-flat on a shaft spans less than max_step and has all its vertices on
        the cylinder, so it passes as one more (wide) facet. On a cylinder that is evenly
        tessellated almost all round -- at least 8 facets of a common width covering
        180+ degrees -- one or two facets over twice that width are taken for flats.
        (Arcs are often tessellated unevenly elsewhere, e.g. thin facets at both tangent
        edges of a fillet, organic exports, so nothing else is touched.)"""
        if self.kind != "cyl":
            return
        X, lens = self.unit_points(units)
        w = self._widths(units, X, lens)
        typ = np.median(w)
        regular = w <= 1.5 * typ
        wide = w > max(2 * typ, typ + np.radians(3))
        if regular.sum() < 8 or regular.sum() * typ < np.radians(180) or not 1 <= wide.sum() <= 2:
            return
        self.wmax = max(2 * typ, typ + np.radians(3))

    def quick_ok(self, units):
        """Cheap necessary conditions before the expensive polish: angular coverage, and
        not a band of a doubly-curved surface (those units are then never seeds again)."""
        X, _ = self.unit_points(units)
        span = _span(self.azimuth(X))
        if span < self.s.min_span * 0.8:
            return False
        if self.doubly_curved_band(units, X, 0, span):
            self.s.noseed[units] = True
            return False
        return True

    def smooth_ends(self, units):
        """Fraction (by length) of the region's *end* boundary (edges across the generators)
        that continues smoothly (facet turn about as large as inside the region) into a
        neighbouring triangle. Returns (fraction, region triangles)."""
        s = self.s
        tris = s.tri_order[_ranges(s.tptr, units)[0]]
        inreg = np.zeros(len(s.faces), bool)
        inreg[tris] = True
        nb = s.nbr[tris]
        k = np.nonzero((nb >= 0) & ~inreg[np.maximum(nb, 0)] | (nb < 0))
        t, e = tris[k[0]], k[1]
        if len(t) == 0:
            return 0.0, tris
        p0 = s.points[s.faces[t, e]]
        p1 = s.points[s.faces[t, (e + 1) % 3]]
        d = p1 - p0
        ln = np.linalg.norm(d, axis=1)
        g = self.gen_dir((p0 + p1) / 2)
        end = np.abs(np.einsum("ij,ij->i", d, g)) < 0.5 * ln
        if not end.any():
            return 0.0, tris
        # typical facet-to-facet turn inside the region; a smooth continuation turns by
        # about as much, a real edge (cap, chamfer, plane) turns by more
        inside = inreg[np.maximum(nb, 0)] & (nb >= 0)
        inner = nb[inside]
        tin = np.repeat(tris, 3).reshape(-1, 3)[inside]
        cin = np.einsum("ij,ij->i", s.tn[tin], s.tn[inner])
        turns = np.arccos(np.clip(cin[cin < np.cos(np.radians(0.3))], -1, 1))
        step = np.median(turns) if len(turns) else s.max_step
        lim = np.cos(min(max(1.3 * step, np.radians(15)), s.max_step))
        other = nb[k]
        smooth = (other >= 0) & (np.einsum("ij,ij->i", s.tn[t], s.tn[np.maximum(other, 0)]) >= lim)
        return float(ln[end & smooth].sum() / ln[end].sum()), tris

    def doubly_curved_band(self, units, X, nd, span):
        """True if the region is a short band cut out of a doubly-curved surface (a row of
        a sphere / torus / revolved surface: two vertex rings always fit a cylinder or cone
        exactly). Signature: at least half of its *end* boundary (edges across the
        generators) continues smoothly (facet angle <= max_step) into the neighbouring
        surface, and it is short: length < 8 facet widths. Real cylinders end at sharp
        edges, or are long compared with their facets (fillet running into a corner blend)."""
        s = self.s
        frac, tris = self.smooth_ends(units)
        if frac < 0.5:
            return False
        # facet width: azimuthal extent of the region's triangles
        P = s.points[s.faces[tris]]
        az = self.azimuth(P.reshape(-1, 3)).reshape(-1, 3)
        daz = np.mod(az - az[:, :1] + np.pi, 2 * np.pi) - np.pi
        wid = np.ptp(daz, axis=1)
        h = (X - self.c) @ self.a
        rad = np.linalg.norm((X - self.c) - np.outer(h, self.a), axis=1).mean()
        width = np.percentile(wid[wid > 1e-9], 90) * rad if (wid > 1e-9).any() else 0.0
        length = np.ptp(h) / max(self.slant_cos(), 1e-6)
        return length < 8 * width

    def unit_devs(self, units):
        """Per unit: the largest distance of one of its triangles' centres from the
        surface (~2/3 of that facet's sagitta)."""
        s = self.s
        tris = s.tri_order[_ranges(s.tptr, units)[0]]
        own = np.repeat(np.arange(len(units)), s.ntri[units])
        dev = np.zeros(len(units))
        np.maximum.at(dev, own, np.abs(self.residual(s.P[tris].mean(1))))
        return dev

    def uneven_facets(self, units):
        """A tessellator steps evenly around an arc, so in a region of 3+ facets one facet
        cutting in over MISFIT_UNIT_RATIO x as deep as the median facet, and by more than
        MISFIT_MIN mm, is a flat face whose corners happen to lie on the surface -- e.g. a
        wide face flanked by two equal narrow chamfers (every isosceles trapezoid profile
        has its four corners on a circle). (Thin facets along the tangent edges of a real
        fillet are uneven too, but cut in by hundredths of a mm, not tenths.)"""
        if len(units) < 3:
            return False
        dev = self.unit_devs(units)
        return bool(dev.max() > max(MISFIT_UNIT_RATIO * np.median(dev), MISFIT_MIN))

    def gen_dir(self, X):
        return np.broadcast_to(self.a, X.shape)

    def slant_cos(self):
        return 1.0

    def radial(self, X):
        v = X - self.c
        r = v - np.outer(v @ self.a, self.a)
        return r / np.maximum(np.linalg.norm(r, axis=1), 1e-300)[:, None]


class _CylModel(_Model):
    kind = "cyl"

    def set(self, a, c, R):
        self.a = _unit(a)
        self.e1, self.e2 = _basis(self.a)
        self.c, self.R = np.asarray(c, float), float(R)

    def residual(self, X):
        v = X - self.c
        r = v - np.outer(v @ self.a, self.a)
        return np.linalg.norm(r, axis=1) - self.R

    def normal_ok(self, nrm, cen):
        s = self.s
        ok = np.abs(nrm @ self.a) <= s.sin_tilt_cyl
        rad = np.einsum("ij,ij->i", nrm, self.radial(cen)) * self.sign
        return ok & (rad >= s.cos_half_step)

    def fit(self, units, axis=None, polish=False):
        """Fit to the vertices of `units`. Axis from the normals unless given."""
        s = self.s
        if axis is None:
            w = s.uarea[units][:, None] * s.un[units]
            M = s.un[units].T @ w
            axis = np.linalg.eigh(M)[1][:, 0]
        a = _unit(axis)
        X, _ = self.unit_points(units)
        e1, e2 = _basis(a)
        cf = _circle_fit(np.column_stack((X @ e1, X @ e2)))
        if cf is None:
            return False
        (cx, cy), R = cf
        h0 = float(np.mean(X @ a))
        self.set(a, cx * e1 + cy * e2 + h0 * a, R)
        if polish and len(X) >= 6:
            self._polish(X)
        if not np.isfinite(self.R) or self.R <= 0:
            return False
        nrm = s.un[units]
        self.sign = 1.0 if (s.uarea[units] * np.einsum("ij,ij->i", nrm, self.radial(s.ucen[units]))).sum() >= 0 else -1.0
        return True

    def _polish(self, X):
        """Levenberg-Marquardt over (axis tilt 2, centre shift 2, radius), same steps as
        _lm, with residual and Jacobian computed together (this runs tens of thousands
        of times on freeform meshes, on a handful of points each time)."""
        a0, c0, R0 = self.a, self.c, self.R
        e1, e2 = self.e1, self.e2
        E = np.array([e1, e2])
        n = len(X)

        def ev(x):
            q = a0 + x[0] * e1 + x[1] * e2
            nq = np.sqrt(q @ q)
            a = q / nq
            v = X - (c0 + x[2] * e1 + x[3] * e2)
            h = v @ a
            w = v - h[:, None] * a
            rho = np.sqrt(np.einsum("ij,ij->i", w, w))
            return rho - (R0 + x[4]), (a, nq, h, w, rho)

        def jac(st):
            a, nq, h, w, rho = st
            wh = w / np.maximum(rho, 1e-300)[:, None]
            J = np.empty((n, 5))
            D = (E - np.outer(E @ a, a)) / nq
            J[:, :2] = -h[:, None] * (wh @ D.T)
            J[:, 2:4] = -(wh @ E.T)
            J[:, 4] = -1.0
            return J

        x = np.zeros(5)
        r, st = ev(x)
        r0max = np.abs(r).max()
        cost = r @ r
        # residuals a millionth of tol are as good as zero: few-point fits (most of the
        # calls) otherwise iterate on down to rounding noise
        done = n * (1e-6 * self.s.tol) ** 2
        lam = 1e-6
        for _ in range(10):
            if cost <= done:
                break
            J = jac(st)
            JtJ = J.T @ J
            g = J.T @ r
            dJ = np.diag(np.diag(JtJ) + 1e-12)
            improved = False
            step = np.zeros(5)
            for _ in range(6):
                try:
                    step = -np.linalg.solve(JtJ + lam * dJ, g)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                rn, stn = ev(x + step)
                cn = rn @ rn
                if np.isfinite(cn) and cn < cost:
                    improved = cn < cost * (1 - 1e-6)
                    x, r, st, cost = x + step, rn, stn, cn
                    lam = max(lam / 10, 1e-9)
                    break
                lam *= 10
            if not improved or np.sqrt(step @ step) < 1e-12:
                break
        a = _unit(a0 + x[0] * e1 + x[1] * e2)
        c = c0 + x[2] * e1 + x[3] * e2
        if np.abs(r).max() <= max(r0max, self.s.tol):
            c = c - (c - c0) @ a * a  # keep the origin near the region centre
            self.set(a, c, R0 + x[4])

    def valid(self, units):
        s = self.s
        if not (s.tol * 10 < self.R < 10 * s.diag):
            return False
        X, _ = self.unit_points(units)
        if _span(self.azimuth(X)) < s.min_span:
            return False
        span = _span(self.azimuth(X))
        nd = _n_dirs(np.arctan2(s.un[units] @ self.e2, s.un[units] @ self.e1), np.radians(MIN_DIR_SEP_DEG))
        if nd < 2 or (nd == 2 and not self._tangent_neighbour(units)):
            return False
        return not self.doubly_curved_band(units, X, nd, span)

    def _tangent_neighbour(self, units):
        """A 2-facet arc is accepted only if a neighbouring face continues it tangentially
        (fillet between two flat faces): its normal equals the radial direction at the
        shared vertices."""
        s = self.s
        nb = _uniq(s.uadj_idx[_ranges(s.uadj_ptr, units)[0]])
        nb = nb[~np.isin(nb, units)]
        if len(nb) == 0:
            return False
        regv = _uniq(s.vidx[_ranges(s.vptr, units)[0]])
        cos_t = np.cos(np.radians(3.0))
        for u in nb.tolist():
            nrm = s.un[u]
            if abs(nrm @ self.a) > s.sin_tilt:
                continue
            vs = s.vidx[s.vptr[u]:s.vptr[u + 1]]
            shared = vs[np.isin(vs, regv)]
            if len(shared) < 2:
                continue
            rad = self.radial(s.points[shared].mean(0)[None])[0] * self.sign
            if nrm @ rad >= cos_t:
                return True
        return False

    def surface(self):
        return Cylinder(self.c.copy(), _canon(self.a.copy()), float(self.R), bool(self.sign > 0))


class _ConeModel(_Model):
    kind = "cone"

    def set(self, apex, a, alpha):
        self.c = np.asarray(apex, float)
        self.a = _unit(a)
        self.e1, self.e2 = _basis(self.a)
        self.alpha = float(alpha)
        self.sa, self.ca = np.sin(self.alpha), np.cos(self.alpha)

    def residual(self, X):
        v = X - self.c
        h = v @ self.a
        rho = np.linalg.norm(v - np.outer(h, self.a), axis=1)
        d = rho * self.ca - h * self.sa
        # the other nappe is not part of the surface
        return np.where(h >= -self.s.tol, d, np.inf)

    def normal_ok(self, nrm, cen):
        s = self.s
        ok = np.abs(nrm @ self.a - self.k) <= s.sin_tilt_cone
        rad = np.einsum("ij,ij->i", nrm, self.radial(cen)) * self.sign
        return ok & (rad >= self.ca * s.cos_half_step)

    def fit(self, units, polish=False):
        s = self.s
        nrm, w = s.un[units], s.uarea[units]
        if hasattr(self, "alpha"):
            # refit: warm-start from the current cone (the closed-form initialisation
            # below is only exact when facets contain the apex, i.e. one-row meshes)
            X, _ = self.unit_points(units)
            self._polish(X)
            self.k = float((w * (nrm @ self.a)).sum() / w.sum())
            self.sign = 1.0 if (w * np.einsum("ij,ij->i", nrm, self.radial(s.ucen[units]))).sum() >= 0 else -1.0
            return True
        m = (w[:, None] * nrm).sum(0) / w.sum()
        d = nrm - m
        C = d.T @ (w[:, None] * d)
        a = np.linalg.eigh(C)[1][:, 0]
        # apex: least-squares intersection of the facet planes
        A = nrm.T @ (w[:, None] * nrm)
        b = (w * np.einsum("ij,ij->i", nrm, s.ucen[units])) @ nrm
        if np.linalg.cond(A) > 1e10:
            return False
        apex = np.linalg.solve(A, b)
        X, _ = self.unit_points(units)
        v = X - apex
        h = v @ a
        if h.mean() < 0:
            a, h = -a, -h
        rho = np.linalg.norm(v - np.outer(h, a), axis=1)
        alpha = float(np.median(np.arctan2(rho, h)))
        if not (0 < alpha < np.pi / 2):
            return False
        self.set(apex, a, alpha)
        if polish and len(X) >= 6:
            self._polish(X)
        self.k = float((w * (nrm @ self.a)).sum() / w.sum())
        self.sign = 1.0 if (w * np.einsum("ij,ij->i", nrm, self.radial(s.ucen[units]))).sum() >= 0 else -1.0
        return True

    def _polish(self, X):
        c0, a0, al0 = self.c, self.a, self.alpha
        e1, e2 = self.e1, self.e2

        def res(x):
            a = _unit(a0 + x[0] * e1 + x[1] * e2)
            c = c0 + x[2] * e1 + x[3] * e2 + x[4] * a0
            al = al0 + x[5]
            v = X - c
            h = v @ a
            rho = np.linalg.norm(v - np.outer(h, a), axis=1)
            return rho * np.cos(al) - h * np.sin(al)

        def jac(x):
            q = a0 + x[0] * e1 + x[1] * e2
            nq = np.sqrt(q @ q)
            a = q / nq
            v = X - (c0 + x[2] * e1 + x[3] * e2 + x[4] * a0)
            al = al0 + x[5]
            ca, sa = np.cos(al), np.sin(al)
            h = v @ a
            w = v - np.outer(h, a)
            rho = np.linalg.norm(w, axis=1)
            wh = w / np.maximum(rho, 1e-300)[:, None]
            J = np.empty((len(X), 6))
            for k, e in enumerate((e1, e2)):
                da = (e - (e @ a) * a) / nq
                J[:, k] = -ca * h * (wh @ da) - sa * (v @ da)
            for k, e in enumerate((e1, e2, a0)):
                J[:, 2 + k] = -ca * (wh @ e) + sa * (a @ e)
            J[:, 5] = -rho * sa - h * ca
            return J

        x = _lm(res, 6, jac=jac, done=len(X) * (1e-6 * self.s.tol) ** 2)
        before = np.abs(res(np.zeros(6))).max()
        if np.abs(res(x)).max() <= max(before, self.s.tol) and 0 < al0 + x[5] < np.pi / 2:
            self.set(c0 + x[2] * e1 + x[3] * e2 + x[4] * a0, a0 + x[0] * e1 + x[1] * e2, al0 + x[5])

    def valid(self, units):
        s = self.s
        if not (s.min_cone < self.alpha < s.max_cone):
            return False
        X, _ = self.unit_points(units)
        h = (X - self.c) @ self.a
        # a fan of triangles around a mesh vertex (sphere pole, dome top) fits a very flat
        # cone with its apex at that vertex; real cone tips are much sharper
        if self.alpha > np.radians(75) and np.linalg.norm(X - self.c, axis=1).min() <= s.tol * 10:
            return False
        if h.max() * np.tan(self.alpha) > 10 * s.diag or h.max() * np.tan(self.alpha) < 10 * s.tol:
            return False
        span = _span(self.azimuth(X))
        if span < s.min_span2:
            return False
        nrm = s.un[units]
        nd = _n_dirs(np.arctan2(nrm @ self.e2, nrm @ self.e1), np.radians(0.5))
        if nd < 3:
            return False
        if _reject_loose(self, X, self.partial(units, X)):
            return False
        return not self.doubly_curved_band(units, X, nd, span)

    def slant_cos(self):
        return self.ca

    def partial(self, units, X=None):
        if X is None:
            X, _ = self.unit_points(units)
        return _span(self.azimuth(X)) < np.radians(180)

    def gen_dir(self, X):
        v = X - self.c
        return v / np.maximum(np.linalg.norm(v, axis=1), 1e-300)[:, None]

    def surface(self):
        return Cone(self.c.copy(), self.a.copy(), float(self.alpha), bool(self.sign > 0))


class _SphereModel(_Model):
    kind = "sphere"

    def set(self, c, R):
        self.c, self.R = np.asarray(c, float), float(R)

    def residual(self, X):
        return np.linalg.norm(X - self.c, axis=1) - self.R

    def radial(self, X):
        v = X - self.c
        return v / np.maximum(np.linalg.norm(v, axis=1), 1e-300)[:, None]

    def normal_ok(self, nrm, cen):
        rad = np.einsum("ij,ij->i", nrm, self.radial(cen)) * self.sign
        return rad >= self.s.cos_half_step

    def test(self, units):
        s = self.s
        ok = self.normal_ok(s.un[units], s.ucen[units])
        if not ok.any():
            return ok
        sub = units[ok]
        X, lens = self.unit_points(sub)
        st = _seg_starts(lens)
        good = np.maximum.reduceat(np.abs(self.residual(X)), st) <= s.grow_tol
        # facet angular size seen from the centre
        d = self.radial(X)
        dc = np.repeat(self.radial(s.ucen[sub]), lens, axis=0)
        good &= np.minimum.reduceat(np.einsum("ij,ij->i", d, dc), st) >= np.cos(s.max_step / 2)
        ok[np.nonzero(ok)[0][~good]] = False
        return ok

    def fit(self, units, polish=False, **_):
        s = self.s
        X, _ = self.unit_points(units)
        if len(X) < 4:
            return False
        m = X.mean(0)
        Q = X - m
        A = np.column_stack((Q, np.ones(len(Q))))
        b = -(Q * Q).sum(1)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        c = -sol[:3] / 2
        r2 = c @ c - sol[3]
        if not np.isfinite(r2) or r2 <= 0:
            return False
        self.set(c + m, np.sqrt(r2))
        if polish:
            c0, R0 = self.c, self.R

            def res(x):
                return np.linalg.norm(X - (c0 + x[:3]), axis=1) - (R0 + x[3])

            def jac(x):
                d = X - (c0 + x[:3])
                J = np.empty((len(X), 4))
                J[:, :3] = -d / np.maximum(np.linalg.norm(d, axis=1), 1e-300)[:, None]
                J[:, 3] = -1.0
                return J

            x = _lm(res, 4, jac=jac, done=len(X) * (1e-6 * s.tol) ** 2)
            if np.abs(res(x)).max() <= max(np.abs(res(np.zeros(4))).max(), s.tol):
                self.set(c0 + x[:3], R0 + x[3])
        nrm, w = s.un[units], s.uarea[units]
        self.sign = 1.0 if (w * np.einsum("ij,ij->i", nrm, self.radial(s.ucen[units]))).sum() >= 0 else -1.0
        return True

    def quick_ok(self, units):
        nrm = self.s.un[units]
        return len(units) >= 4 and _spread_at_least(nrm, np.cos(self.s.min_span * 0.8))

    def valid(self, units):
        s = self.s
        if not (s.tol * 10 < self.R < 10 * s.diag) or len(units) < 4:
            return False
        nrm = s.un[units]
        # angular extent of the patch
        if not _spread_at_least(nrm, np.cos(s.min_span2)):
            return False
        # doubly curved: normals must NOT all lie on one circle of the Gauss sphere
        # (a single row of a cylinder / cone / revolved surface fits a sphere exactly)
        m = nrm.mean(0)
        ev, vec = np.linalg.eigh((nrm - m).T @ (nrm - m))
        dev = np.abs((nrm - m) @ vec[:, 0]).max()
        if dev < 0.01:
            return False
        X, _ = self.unit_points(units)
        return not _reject_loose(self, X, self.partial(units))

    def partial(self, units, X=None):
        """Well short of a hemisphere's worth of normals."""
        return not _spread_at_least(self.s.un[units], np.cos(np.radians(150)))

    def surface(self):
        return Sphere(self.c.copy(), float(self.R), bool(self.sign > 0))


class _TorusModel(_Model):
    kind = "torus"

    def set(self, c, a, R, r):
        self.c = np.asarray(c, float)
        self.a = _unit(a)
        self.e1, self.e2 = _basis(self.a)
        self.R, self.r = float(R), float(r)

    def _rh(self, X):
        v = X - self.c
        h = v @ self.a
        rho = np.linalg.norm(v - np.outer(h, self.a), axis=1)
        return rho, h

    def residual(self, X):
        rho, h = self._rh(X)
        return np.hypot(rho - self.R, h) - self.r

    def radial(self, X):
        """Unit vector from the nearest tube-centre point to X."""
        v = X - self.c
        h = v @ self.a
        p = v - np.outer(h, self.a)
        rho = np.maximum(np.linalg.norm(p, axis=1), 1e-300)
        q = p * (self.R / rho)[:, None]
        d = v - q
        return d / np.maximum(np.linalg.norm(d, axis=1), 1e-300)[:, None]

    def normal_ok(self, nrm, cen):
        rad = np.einsum("ij,ij->i", nrm, self.radial(cen)) * self.sign
        return rad >= self.s.cos_half_step

    def test(self, units):
        s = self.s
        ok = self.normal_ok(s.un[units], s.ucen[units])
        if not ok.any():
            return ok
        sub = units[ok]
        X, lens = self.unit_points(sub)
        st = _seg_starts(lens)
        good = np.maximum.reduceat(np.abs(self.residual(X)), st) <= s.grow_tol
        ok[np.nonzero(ok)[0][~good]] = False
        return ok

    def fit(self, units, polish=False, **_):
        s = self.s
        if not hasattr(self, "r"):
            # axis: the line meeting all facet normal lines (linear in Pluecker coordinates)
            n, p = s.un[units], s.ucen[units]
            if len(units) < 5:
                return False
            M = np.column_stack((np.cross(p, n), n))
            sv, V = np.linalg.svd(M, full_matrices=False)[1:]
            a, mL = V[-1, :3], V[-1, 3:]
            na = np.linalg.norm(a)
            # a sphere patch (all normal lines through one point) leaves the axis undetermined
            if na < 1e-9 or sv[-2] < 1e-9 * sv[0]:
                return False
            a, mL = a / na, mL / na
            c = np.cross(a, mL)
            X, _ = self.unit_points(units)
            v = X - c
            h = v @ a
            rho = np.linalg.norm(v - np.outer(h, a), axis=1)
            cf = _circle_fit(np.column_stack((rho, h)))
            if cf is None:
                return False
            (R, h0), r = cf
            self.set(c + h0 * a, a, R, r)
            # facet normals/centroids only approximate the surface; the closed form must
            # already be close, or this patch is not a torus
            if np.abs(self.residual(X)).max() > 20 * s.tol or R < 0.1 * r:
                return False  # not a torus, or sphere-like (axis undetermined)
            # a real torus converges in a couple of steps from here; patches of threads and
            # other freeform surfaces crawl (and then fail the tol check anyway)
            self._polish(X, iters=3)
            if not np.abs(self.residual(X)).max() <= 3 * s.tol:
                return False
            polish = True
        if polish:
            X, _ = self.unit_points(units)
            if len(X) > 3000:  # deterministic subsample keeps refits cheap on big regions
                X = X[:: len(X) // 3000 + 1]
            self._polish(X)
        if not (np.isfinite(self.R) and np.isfinite(self.r) and self.r > 0):
            return False
        nrm, w = s.un[units], s.uarea[units]
        self.sign = 1.0 if (w * np.einsum("ij,ij->i", nrm, self.radial(s.ucen[units]))).sum() >= 0 else -1.0
        return True

    def _polish(self, X, iters=8):
        """Levenberg-Marquardt with an analytic Jacobian (axis tilt 2, centre 3, R, r)."""
        a, c, R, r = self.a, self.c, self.R, self.r

        def resid(a, c, R, r):
            v = X - c
            h = v @ a
            w = v - np.outer(h, a)
            rho = np.sqrt((w * w).sum(1))
            D = np.sqrt((rho - R) ** 2 + h * h)
            return D - r, v, h, w, rho, D

        f, v, h, w, rho, D = resid(a, c, R, r)
        cost0 = cost = f @ f
        start = np.abs(f).max()
        lam = 1e-6
        done_cost = len(X) * (1e-6 * self.s.tol) ** 2   # as good as exact
        for _ in range(iters):
            if cost <= done_cost:
                break
            e1, e2 = _basis(a)
            rho_s = np.maximum(rho, 1e-300)
            D_s = np.maximum(D, 1e-300)
            what = w / rho_s[:, None]
            dR = rho - R
            J = np.empty((len(X), 7))
            for k, e in enumerate((e1, e2)):          # axis tilt
                J[:, k] = (dR * (-h * (what @ e)) + h * (v @ e)) / D_s
            for k, e in enumerate((e1, e2, a)):       # centre shift
                J[:, 2 + k] = (dR * (-(what @ e)) + h * (-(e @ a))) / D_s
            J[:, 5] = -dR / D_s
            J[:, 6] = -1.0
            JtJ = J.T @ J
            g = J.T @ f
            done = True
            for _ in range(6):
                try:
                    step = -np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-12), g)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                an = _unit(a + step[0] * e1 + step[1] * e2)
                cn = c + step[2] * e1 + step[3] * e2 + step[4] * a
                Rn, rn = R + step[5], r + step[6]
                out = resid(an, cn, Rn, rn)
                cn_cost = out[0] @ out[0]
                if np.isfinite(cn_cost) and cn_cost < cost:
                    a, c, R, r = an, cn, Rn, rn
                    f, v, h, w, rho, D = out
                    cost = cn_cost
                    lam = max(lam / 10, 1e-9)
                    done = np.linalg.norm(step) < 1e-12
                    break
                lam *= 10
            else:
                break
            if done:
                break
        if cost <= cost0 and np.abs(f).max() <= max(start, self.s.tol):
            self.set(c, a, R, r)

    def quick_ok(self, units):
        nrm = self.s.un[units]
        return len(units) >= 6 and self.R >= 0.1 * self.r and _spread_at_least(nrm, np.cos(self.s.min_span * 0.8))

    def valid(self, units):
        s = self.s
        if not (10 * s.tol < self.r < 10 * s.diag) or not (0.1 * self.r < self.R < 10 * s.diag):
            return False
        if len(units) < 6:
            return False
        X, _ = self.unit_points(units)
        rho, h = self._rh(X)
        minor = _span(np.arctan2(h, rho - self.R))
        v = X - self.c
        major = _span(np.arctan2(v @ self.e2, v @ self.e1))
        if minor < s.min_span2 or major < s.min_span2:
            return False
        # a ring segment fitted loosely is a thread / freeform patch (see _reject_loose)
        return not _reject_loose(self, X, major < np.radians(270))

    def partial(self, units, X=None):
        if X is None:
            X, _ = self.unit_points(units)
        v = X - self.c
        return _span(np.arctan2(v @ self.e2, v @ self.e1)) < np.radians(270)

    def surface(self):
        return Torus(self.c.copy(), _canon(self.a.copy()), float(self.R), float(self.r), bool(self.sign > 0))


# ------------------------------------------------------------------ the segmenter

class _Seg:
    def __init__(self, points, faces, tol, angle_deg, max_step_deg, max_tilt_deg, min_span_deg):
        self.points = np.asarray(points, np.float64)
        self.faces = np.asarray(faces, np.int64)
        self.tol = float(tol)
        self.grow_tol = self.tol
        self.max_step = np.radians(max_step_deg)
        self.cos_half_step = np.cos(self.max_step / 2 + np.radians(5))
        self.sin_tilt = np.sin(np.radians(max_tilt_deg))
        self.sin_tilt_cyl = np.sin(np.radians(min(max_tilt_deg, 1.5)))
        self.sin_tilt_cone = np.sin(np.radians(min(max_tilt_deg, 3.0)))
        self.min_span = np.radians(min_span_deg)
        # cones, spheres and tori need more coverage: small patches of helical / freeform
        # surfaces (threads, organic shapes) fit them within tolerance by accident
        self.min_span2 = np.radians(max(min_span_deg, MIN_SPAN_DOUBLE_DEG))
        self.min_cone, self.max_cone = np.radians(2.0), np.radians(88.0)
        P = self.points[self.faces]
        self.P = P
        cr = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        ln = np.linalg.norm(cr, axis=1)
        self.tarea = ln / 2
        self.tn = cr / np.maximum(ln, 1e-300)[:, None]
        self.nbr = edge_adjacency(self.faces)
        self.diag = float(np.linalg.norm(np.ptp(self.points, axis=0))) if len(self.points) else 0.0
        # a cylinder region must cover min_span of arc inside the part, so its chord
        # 2 R sin(min_span / 2) is at most the bounding-box diagonal
        self.max_radius = self.diag / (2 * np.sin(self.min_span / 2))
        self.angle_deg = angle_deg

    # -------------------------------------------------- unit tables
    def build_units(self):
        F, V = len(self.faces), len(self.points)
        # units are built with a much tighter tolerance than planes: only exactly flat
        # facets (both triangles of a strip quad, a real flat face) stay together, so a
        # sliver of a neighbouring curved surface never gets welded onto a facet strip
        noise = 5e-7 * float(np.abs(self.points).max()) if len(self.points) else 0.0
        unit = _planar_units(self.P, self.tn, self.tarea, self.nbr, max(UNIT_TOL_FRAC * self.tol, noise),
                             min(self.angle_deg, UNIT_ANGLE_DEG))
        U = int(unit.max()) + 1 if F else 0
        self.unit, self.U = unit, U
        a = self.tarea
        self.uarea = np.bincount(unit, a, U)
        ns = np.zeros((U, 3))
        np.add.at(ns, unit, self.tn * a[:, None])
        self.un = ns / np.maximum(np.linalg.norm(ns, axis=1), 1e-300)[:, None]
        cs = np.zeros((U, 3))
        np.add.at(cs, unit, self.P.mean(1) * a[:, None])
        self.ucen = cs / np.maximum(self.uarea, 1e-300)[:, None]
        self.ntri = np.bincount(unit, minlength=U)
        order = np.argsort(unit, kind="stable")
        self.tri_order = order
        self.tptr = np.searchsorted(unit[order], np.arange(U + 1))
        # unit -> unique vertices
        key = _uniq(np.repeat(unit, 3) * V + self.faces.ravel())
        self.vidx = key % V
        self.vptr = np.searchsorted(key // V, np.arange(U + 1))
        # unit adjacency
        r = np.repeat(unit, 3)
        c = self.nbr.ravel()
        ok = c >= 0
        r, c = r[ok], unit[c[ok]]
        ok = r != c
        key = _uniq(r[ok] * U + c[ok])
        self.uadj_idx = key % U
        self.uadj_ptr = np.searchsorted(key // U, np.arange(U + 1))
        self.usrc = np.repeat(np.arange(U), np.diff(self.uadj_ptr))
        self.label = np.full(U, -1, np.int64)   # unit -> surface index
        self.noseed = np.zeros(U, bool)
        self.deferred = np.zeros(U, bool)
        # units no region may grow into for now (set-aside regions, during the first
        # cylinder / cone passes: another seed would only find the same band again)
        self.blocked = np.zeros(U, bool)
        self.doubly_failed = np.zeros(U, bool)
        self.deferred_regions = []
        self.cyl_res = []   # max residual / tol of each accepted cylinder (mesh quality)
        self.surfaces = []
        self.kinds = []

    # -------------------------------------------------- seeds
    def seed_triples(self, only=None, k_near=8):
        """Candidate seed triples (u, v, w): v, w among the k_near nearest-normal neighbour
        units of u (normal angle within max_step, not parallel), v and w not parallel either.
        Sorted by u, then nearest pairs first."""
        U = self.U
        e = np.zeros(0, np.int64)
        src = np.repeat(np.arange(U), np.diff(self.uadj_ptr))
        dst = self.uadj_idx
        if only is not None:
            m = only[src]
            src, dst = src[m], dst[m]
        cosang = np.einsum("ij,ij->i", self.un[src], self.un[dst])
        ok = (cosang < np.cos(np.radians(0.3))) & (cosang >= np.cos(self.max_step))
        src, dst, cosang = src[ok], dst[ok], cosang[ok]
        if len(src) == 0:
            return e, e, e
        o = np.lexsort((dst, -cosang, src))
        src, dst = src[o], dst[o]
        first = np.concatenate(([True], src[1:] != src[:-1]))
        gstart = np.maximum.accumulate(np.where(first, np.arange(len(src)), 0))
        rank = np.arange(len(src)) - gstart
        keep = rank < k_near
        src, dst, rank = src[keep], dst[keep], rank[keep]
        us, vs, ws, key = [], [], [], []
        for d in range(1, k_near):
            i = np.arange(len(src) - d)
            m = src[i] == src[i + d]
            i = i[m]
            us.append(src[i]); vs.append(dst[i]); ws.append(dst[i + d])
            key.append(rank[i] + rank[i + d])
        if not us:
            return e, e, e
        u, v, w, key = map(np.concatenate, (us, vs, ws, key))
        ok = np.einsum("ij,ij->i", self.un[v], self.un[w]) < np.cos(np.radians(0.3))
        u, v, w, key = u[ok], v[ok], w[ok], key[ok]
        # prefer v, w on opposite sides of u (u in the middle of a strip of facets)
        same_side = np.einsum("ij,ij->i", self.un[v] - self.un[u], self.un[w] - self.un[u]) > 0
        o = np.lexsort((w, v, key, same_side, u))
        return u[o], v[o], w[o]

    def _pair_edge_dir(self, a, b):
        """Unit direction of one mesh edge shared by units a[i] and b[i] (zeros if none)."""
        if not hasattr(self, "_pe_key"):
            F = len(self.faces)
            t = np.repeat(np.arange(F), 3)
            e = np.tile(np.arange(3), F)
            o = self.nbr.ravel()
            m = o >= 0
            t, e, o = t[m], e[m], o[m]
            ua, ub = self.unit[t], self.unit[o]
            m = ua != ub
            t, e, ua, ub = t[m], e[m], ua[m], ub[m]
            d = self.points[self.faces[t, (e + 1) % 3]] - self.points[self.faces[t, e]]
            d /= np.linalg.norm(d, axis=1)[:, None]
            key = ua * self.U + ub
            self._pe_key, first = _uniq_first(key)
            self._pe_dir = d[first]
        key = a * self.U + b
        i = np.minimum(np.searchsorted(self._pe_key, key), len(self._pe_key) - 1)
        found = self._pe_key[i] == key
        return np.where(found[:, None], self._pe_dir[i], 0.0)

    def _first_ok_per_unit(self, u, check, n=2):
        """Indices of the first n rows per distinct u (rows sorted by u) for which
        check(rows) is True -- the same rows as filtering everything with check and then
        taking _first_per_unit, but check only runs on rows that could be chosen."""
        m = len(u)
        if m == 0:
            return np.zeros(0, np.int64)
        first = np.concatenate(([True], u[1:] != u[:-1]))
        grp = np.cumsum(first) - 1
        gstart = np.nonzero(first)[0]
        done = np.zeros(m, bool)
        good = np.zeros(m, bool)
        while True:
            have = np.bincount(grp[good], minlength=len(gstart))
            todo = ~done
            # position of each unchecked row among its group's unchecked rows (1-based)
            c = np.cumsum(todo)
            c = c - np.concatenate(([0], c))[gstart][grp]
            pick = np.nonzero(todo & (c <= n - have[grp]))[0]
            if len(pick) == 0:
                break
            good[pick] = check(pick)
            done[pick] = True
        return np.nonzero(good)[0][self._first_per_unit(u[good], n)]

    def _first_per_unit(self, u, n=2):
        """Indices of the first n rows for each distinct u (rows sorted by u)."""
        if len(u) == 0:
            return np.zeros(0, np.int64)
        first = np.concatenate(([True], u[1:] != u[:-1]))
        gstart = np.maximum.accumulate(np.where(first, np.arange(len(u)), 0))
        return np.nonzero(np.arange(len(u)) - gstart < n)[0]

    # -------------------------------------------------- region growing
    def grow(self, model, seed_units, refit):
        """Grow over available units from seed_units. Returns sorted unit array."""
        s = self
        inreg = np.zeros(self.U, bool)
        seen = np.zeros(self.U, bool)
        inreg[seed_units] = seen[seed_units] = True
        region = list(seed_units)
        new = np.asarray(seed_units)
        grown_at_fit = len(region)
        stalled_at = -1
        waves = 0
        while True:
            waves += 1
            if not refit and waves > 8 and len(region) >= 64:
                # with the model fixed, membership is a per-unit test, so the grown region
                # is simply the seed's connected component among passing units: on a big
                # surface one vectorised pass beats a long breadth-first front
                return self._component(model, np.asarray(seed_units))
            pos, _ = _ranges(s.uadj_ptr, new)
            cand = _uniq(s.uadj_idx[pos])
            cand = cand[~seen[cand] & (s.label[cand] < 0) & ~s.blocked[cand]]
            acc = cand[:0]
            if len(cand):
                seen[cand] = True
                acc = cand[model.test(cand)]
            if len(acc) == 0:
                # stalled: early fits on a small patch extrapolate poorly, so refit the
                # whole region properly and give the rejected neighbours another chance
                if not refit or stalled_at == len(region):
                    break
                stalled_at = len(region)
                reg = np.array(region)
                saved = model.__dict__.copy()
                if not model.fit(reg, polish=model.kind != "sphere"):
                    model.__dict__.update(saved)
                grown_at_fit = len(region)
                pos, _ = _ranges(s.uadj_ptr, reg)
                ring = _uniq(s.uadj_idx[pos])
                ring = ring[~inreg[ring]]
                seen[ring] = False
                new = reg
                continue
            inreg[acc] = True
            region.extend(acc.tolist())
            new = acc
            if refit and len(region) >= 1.2 * grown_at_fit:
                saved = model.__dict__.copy()
                if not model.fit(np.array(region), polish=model.kind == "cone" or (model.kind == "cyl" and len(region) < 12)):
                    model.__dict__.update(saved)
                grown_at_fit = len(region)
                if model.kind == "cone" and len(region) >= 50 and model.alpha < 0.5 * s.min_cone:
                    break   # it is a cylinder: no cone (alpha > min_cone) will come of it
        return np.array(sorted(region))

    def _component(self, model, seed_units):
        free = np.nonzero((self.label < 0) & ~self.blocked)[0]
        ok = np.zeros(self.U, bool)
        ok[free[model.test(free)]] = True
        ok[seed_units] = True
        src, dst = self.usrc, self.uadj_idx
        m = ok[src] & ok[dst]
        idx = np.full(self.U, -1, np.int64)
        nodes = np.nonzero(ok)[0]
        idx[nodes] = np.arange(len(nodes))
        g = coo_matrix((np.ones(int(m.sum()), np.int8), (idx[src[m]], idx[dst[m]])), shape=(len(nodes), len(nodes)))
        comp = connected_components(g, directed=False)[1]
        return nodes[comp == comp[idx[seed_units[0]]]]

    def _cyl_dead_end(self, init):
        """Cheap look-ahead for a cylinder start: fitted the way growth refits (axis from
        the facet normals, polished), the start covers less than the minimum arc and not
        one free neighbour of it lies on that cylinder -- so growth would stall at once
        and quick_ok fail. On threads, organic and patterned meshes that is nearly every
        start, and the look-ahead costs a fraction of the fit / grow / refit / stall
        cycle. (Checked against all 65 survey meshes: no cylinder is lost.)"""
        ring = _uniq(self.uadj_idx[_ranges(self.uadj_ptr, init)[0]])
        ring = ring[(self.label[ring] < 0) & ~self.blocked[ring] & ~np.isin(ring, init)]
        m = _CylModel(self)
        if not m.fit(init, polish=True):
            return False
        X, _ = m.unit_points(init)
        return not (_span(m.azimuth(X)) >= self.min_span * 0.8 or (len(ring) and m.test(ring).any()))

    def try_region(self, model, seed, init_units, fit_kw):
        """Initial fit on init_units, grow, polish, re-grow fixed. Region or None."""
        if model.kind == "cyl" and self._cyl_dead_end(np.asarray(init_units)):
            return None
        if not model.fit(np.asarray(init_units), **fit_kw):
            return None
        if model.kind == "cyl" and model.R > self.max_radius:
            return None
        X, _ = model.unit_points(np.asarray(init_units))
        if np.abs(model.residual(X)).max() > self.tol:
            return None
        seedarr = np.array([seed])
        if not model.test(seedarr).all():
            return None
        region = self.grow(model, seedarr, refit=True)
        self.last_grown = region
        self.hopeless = True   # failing before the polish: the patch can't be this surface
        if len(region) < 2:
            return None
        if model.kind == "cone" and model.alpha < 0.5 * self.min_cone:
            return None   # a cylinder (or a patch of one), see grow
        if not model.quick_ok(region):
            if (model.kind in ("cyl", "cone") and len(region) >= 6 and self.label[region].max() < 0
                    and model.smooth_ends(region)[0] >= 0.5):
                # a band of a torus / sphere: grow that surface from here right away
                # instead of trying band after band of it
                self.doubly_from(region)
            return None
        self.hopeless = False
        if model.kind == "cyl":
            # the polished re-grow uses a narrower band, so it does not widen the arc:
            # short of min_span now, valid() will reject it (in a survey of 65 real
            # meshes no accepted cylinder ever had less arc at this point)
            X, _ = model.unit_points(region)
            if _span(model.azimuth(X)) < self.min_span:
                return None
        if model.kind in ("cyl", "cone"):
            model.limit_width(region)
            if not model.test(seedarr).all():
                # the seed is the odd wide facet: let another seed start this surface
                self.last_grown = None
                return None
        for _ in range(2):
            saved = model.__dict__.copy()
            if not model.fit(region, polish=True):
                model.__dict__.update(saved)
            self.grow_tol = self.tol * GROW_TOL_FRAC
            try:
                if not model.test(seedarr).all():
                    return None
                new = self.grow(model, seedarr, refit=False)
            finally:
                self.grow_tol = self.tol
            if len(new) <= len(region) and np.array_equal(new, region):
                break
            region = new
        if model.kind != "cyl" and len(region) >= 2 and _exact_mesh(self) and not model.partial(region):
            # exact tessellation: a real surface sits well inside the band, so if this
            # (all-round) one doesn't, re-grow it with the band narrowed to what such a
            # fit needs. A real surface that picked up a row of a neighbour keeps nearly
            # all of its facets; an approximate fit (organic patch) falls apart
            X, _ = model.unit_points(region)
            if float(np.abs(model.residual(X)).max()) > LOOSE_FIT_FRAC * self.tol:
                loose_size = len(region)
                saved = model.__dict__.copy()
                self.grow_tol = self.tol * TIGHT_REGROW_FRAC
                try:
                    for _ in range(2):
                        if not model.test(seedarr).all():
                            return None
                        region = self.grow(model, seedarr, refit=False)
                        if len(region) < 2 or not model.fit(region, polish=True):
                            model.__dict__.update(saved)
                            return None
                    if not model.test(seedarr).all():
                        return None
                    region = self.grow(model, seedarr, refit=False)
                finally:
                    self.grow_tol = self.tol
                if len(region) < 0.8 * loose_size:
                    return None
        if len(region) < 2 or not model.valid(region):
            return None
        return region

    def accept(self, model, region):
        if model.kind == "cyl" and not self.deferred[region].any():
            X, _ = model.unit_points(region)
            self.cyl_res.append(float(np.abs(model.residual(X)).max()) / self.tol)
        self.label[region] = len(self.surfaces)
        self.surfaces.append(model.surface())
        self.kinds.append(model.kind)
        self.models.append(model)

    def accept_or_defer(self, model, region, defer):
        """Accept a cylinder / cone region, or (first pass) set it aside when its ends run
        smoothly into more curved surface: such a region may be a band of a torus or sphere
        (at fine tessellation several facet rows of those fit a cylinder or cone within
        tol), so the doubly-curved passes get the first go at it."""
        if defer and model.smooth_ends(region)[0] >= 0.5:
            if model.kind == "cyl":   # counts towards the mesh's fit quality all the same
                X, _ = model.unit_points(region)
                self.cyl_res.append(float(np.abs(model.residual(X)).max()) / self.tol)
            if self.doubly_from(region):
                return False
            self.deferred[region] = True
            self.noseed[region] = True
            self.blocked[region] = True
            self.deferred_regions.append((model, region))
            return False
        if model.uneven_facets(region):
            # checked here, after the smooth-ends test: a band of a torus / sphere has
            # uneven facets too and must still lead to that surface (doubly_from)
            self.noseed[region] = True
            return False
        self.accept(model, region)
        return True

    def accept_deferred(self):
        """Set-aside cylinder / cone regions that no torus or sphere claimed are accepted
        as found (their units never overlap: they were blocked for later growth)."""
        for model, region in self.deferred_regions:
            if (self.label[region] < 0).all() and not model.uneven_facets(region):
                self.accept(model, region)

    def doubly_from(self, region):
        """Try a torus, then a sphere, on `region` plus two rings of free neighbours; accept
        the first that grows into a valid region covering most of `region`."""
        if self.doubly_failed[region].mean() >= 0.5:
            return False   # been here: same band, same answer
        patch = np.asarray(region)
        # facets only: a big flat unit (a tangent plane) is never part of the curved surface
        amax = 4 * np.median(self.uarea[patch])
        for _ in range(2):
            ring = self.uadj_idx[_ranges(self.uadj_ptr, patch)[0]]
            patch = np.union1d(patch, ring[(self.label[ring] < 0) & (self.uarea[ring] <= amax)])
        seed = int(region[np.lexsort((region, -self.uarea[region]))[0]])
        saved = self.noseed, self.last_grown, self.blocked   # the calling pass's bookkeeping
        for M in (_TorusModel, _SphereModel):
            m = M(self)
            self.noseed = np.zeros(self.U, bool)
            self.blocked = np.zeros(self.U, bool)
            try:
                reg = self.try_region(m, seed, patch, {})
            finally:
                self.noseed, self.last_grown, self.blocked = saved
            if reg is not None and np.isin(region, reg).mean() >= 0.5:
                self.accept(m, reg)
                return True
        self.doubly_failed[region] = True
        return False

    def cylinders(self, defer=False, only=None):
        self.noseed = np.zeros(self.U, bool)   # per pass
        u, v, w = self.seed_triples(only)
        nu, nv, nw = self.un[u], self.un[v], self.un[w]
        ax1 = np.cross(nu, nv)
        ax2 = np.cross(nu, nw)
        ax1 /= np.linalg.norm(ax1, axis=1)[:, None]
        ax2 /= np.linalg.norm(ax2, axis=1)[:, None]
        dot = np.einsum("ij,ij->i", ax1, ax2)
        agree = np.abs(dot) >= np.cos(np.radians(2.0))
        u, v, w = u[agree], v[agree], w[agree]
        ax = ax1[agree] + ax2[agree] * np.sign(dot[agree])[:, None]
        # prefer triples whose shared edges run along the axis (facet strips of a
        # tessellated cylinder share generator lines; a row of a sphere/torus does not)
        along = (np.abs(np.einsum("ij,ij->i", self._pair_edge_dir(u, v), ax / np.linalg.norm(ax, axis=1)[:, None])) > 0.99) \
            & (np.abs(np.einsum("ij,ij->i", self._pair_edge_dir(u, w), ax / np.linalg.norm(ax, axis=1)[:, None])) > 0.99)
        o = np.lexsort((np.arange(len(u)), ~along, u))
        u, v, w, ax = u[o], v[o], w[o], ax[o]
        # the first two triples per seed that lie on a circle (checked lazily: the check
        # is costly with big flat units and most seeds have dozens of candidate triples)
        sel = self._first_ok_per_unit(u, lambda i: self._circle_triples(u[i], v[i], w[i], ax[i]), 2)
        u, v, w, ax = u[sel], v[sel], w[sel], ax[sel]
        noseed = self.noseed
        # biggest facets first: deterministic and tends to start in the middle of strips
        order = np.lexsort((np.arange(len(u)), -self.uarea[u]))
        for i in order.tolist():
            seed = int(u[i])
            if self.label[seed] >= 0 or noseed[seed]:
                continue
            axis = _unit(ax[i])
            for init in ((seed, v[i], w[i]), (seed, w[i]), (seed, v[i])):
                if any(self.label[x] >= 0 for x in init[1:]):
                    continue
                m = _CylModel(self)
                self.last_grown = None
                region = self.try_region(m, seed, list(init), {"axis": axis, "polish": True})
                if region is not None:
                    self.accept_or_defer(m, region, defer)
                    break
                # a sizeable patch that fits one cylinder but is not a valid cylinder
                # region (too narrow, freeform): don't restart from inside it
                if self.last_grown is not None and len(self.last_grown) >= 6:
                    noseed[self.last_grown] = True
                    break

    def _unit_moments(self):
        """Per unit: vertex count, centroid, and 2nd / 3rd central moments of its vertices
        (enough to set up an algebraic circle fit of any group of units in any plane)."""
        if hasattr(self, "_um"):
            return self._um
        U = self.U
        own = np.repeat(np.arange(U), np.diff(self.vptr))
        X = self.points[self.vidx]
        cnt = np.bincount(own, minlength=U).astype(float)
        cen = np.stack([np.bincount(own, X[:, k], U) for k in range(3)], 1) / np.maximum(cnt, 1)[:, None]
        d = X - cen[own]
        M2 = np.empty((U, 3, 3))
        M3 = np.empty((U, 3, 3, 3))
        for i in range(3):
            for j in range(i, 3):
                M2[:, i, j] = M2[:, j, i] = np.bincount(own, d[:, i] * d[:, j], U)
                for k in range(j, 3):
                    val = np.bincount(own, d[:, i] * d[:, j] * d[:, k], U)
                    for p in {(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)}:
                        M3[(slice(None),) + p] = val
        self._um = (cnt, cen, M2, M3)
        return self._um

    def _circle_triples(self, u, v, w, ax):
        """Vectorised check that the vertices of each triple lie on one circle around `ax`
        (algebraic circle fit in the plane perpendicular to the axis). The fit comes from
        per-unit moments; the residual check goes unit by unit, small units first, and
        only on triples still alive, so big flat units are rarely touched vertex by vertex."""
        n = len(u)
        if n == 0:
            return np.zeros(0, bool)
        ax = ax / np.linalg.norm(ax, axis=1)[:, None]
        h = np.where(np.abs(ax[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        e1 = np.cross(ax, h)
        e1 /= np.linalg.norm(e1, axis=1)[:, None]
        e2 = np.cross(ax, e1)
        E = np.stack((e1, e2), 1)                                  # (n, 2, 3)
        cnt, cen, M2, M3 = self._unit_moments()
        tri = np.stack((u, v, w), 1)
        nk = cnt[tri]                                              # (n, 3)
        N = nk.sum(1)
        mean = np.einsum("nk,nkj->nj", nk, cen[tri]) / N[:, None]
        S2 = np.zeros((n, 3, 3))
        S3 = np.zeros((n, 3, 3, 3))
        for k in range(3):
            dl = cen[tri[:, k]] - mean                             # shift to the triple mean
            m2 = M2[tri[:, k]]
            S2 += m2 + nk[:, k, None, None] * np.einsum("ni,nj->nij", dl, dl)
            S3 += (M3[tri[:, k]] + np.einsum("nij,nk->nijk", m2, dl) + np.einsum("nik,nj->nijk", m2, dl)
                   + np.einsum("njk,ni->nijk", m2, dl) + nk[:, k, None, None, None] * np.einsum("ni,nj,nk->nijk", dl, dl, dl))
        P2 = np.einsum("nai,nij,nbj->nab", E, S2, E)               # sums of x x, x y, y y
        P3 = np.einsum("nai,nbj,nck,nijk->nabc", E, E, E, S3)
        Szx = P3[:, 0, 0, 0] + P3[:, 1, 1, 0]                      # sum (x^2 + y^2) x
        Szy = P3[:, 0, 0, 1] + P3[:, 1, 1, 1]
        Sz = P2[:, 0, 0] + P2[:, 1, 1]
        A = np.zeros((n, 3, 3))
        A[:, :2, :2] = P2
        A[:, 2, 2] = N
        bvec = -np.stack((Szx, Szy, Sz), 1)
        ok = np.abs(np.linalg.det(A)) > 1e-18
        A[~ok] = np.eye(3)
        sol = np.linalg.solve(A, bvec[:, :, None])[:, :, 0]
        cx, cy = -sol[:, 0] / 2, -sol[:, 1] / 2
        r2 = cx * cx + cy * cy - sol[:, 2]
        ok &= r2 > 0
        r = np.sqrt(np.maximum(r2, 0))
        ok &= (r > 10 * self.tol) & (r < self.max_radius)
        # residuals, smallest unit of each triple first, survivors only
        order = np.argsort(nk, axis=1, kind="stable")
        for col in range(3):
            live = np.nonzero(ok)[0]
            if len(live) == 0:
                break
            x = tri[live, order[live, col]]
            pos, lens = _ranges(self.vptr, x)
            owner = np.repeat(np.arange(len(live)), lens)
            Q = self.points[self.vidx[pos]] - mean[live][owner]
            px = np.einsum("ij,ij->i", Q, e1[live][owner])
            py = np.einsum("ij,ij->i", Q, e2[live][owner])
            res = np.abs(np.hypot(px - cx[live][owner], py - cy[live][owner]) - r[live][owner])
            # Kasa is slightly biased on short arcs; allow a little slack, the real check follows
            ok[live[np.maximum.reduceat(res, _seg_starts(lens)) > 2 * self.tol]] = False
        return ok

    def _cone_triples(self, u, v, w):
        """Vectorised cone through three facet units: apex = intersection of the three facet
        planes, axis = normal of the plane through the three Gauss-sphere points. Returns a
        mask of triples whose vertices all lie on that cone within tol, plus alpha."""
        nu, nv, nw = self.un[u], self.un[v], self.un[w]
        ax = np.cross(nv - nu, nw - nu)
        ln = np.linalg.norm(ax, axis=1)
        ok = ln > 1e-9
        ax = ax / np.maximum(ln, 1e-300)[:, None]
        N = np.stack((nu, nv, nw), 1)
        rhs = np.stack([np.einsum("ij,ij->i", self.un[x], self.ucen[x]) for x in (u, v, w)], 1)
        det = np.linalg.det(N)
        ok &= np.abs(det) > 1e-9
        Nsafe = np.where(ok[:, None, None], N, np.eye(3))
        apex = np.linalg.solve(Nsafe, rhs[:, :, None])[:, :, 0]
        sgn = np.sign(np.einsum("ij,ij->i", self.ucen[u] - apex, ax))
        ax *= np.where(sgn == 0, 1, sgn)[:, None]
        alpha = np.zeros(len(u))
        # unit by unit (u first: it sets the opening angle), each stage only on the
        # triples still alive -- big flat units otherwise make this the slowest step
        live = np.nonzero(ok)[0]
        for x in (u, v, w):
            if len(live) == 0:
                break
            pos, lens = _ranges(self.vptr, x[live])
            owner = np.repeat(np.arange(len(live)), lens)
            X = self.points[self.vidx[pos]] - apex[live][owner]
            axo = ax[live][owner]
            h = np.einsum("ij,ij->i", X, axo)
            rho = np.linalg.norm(X - h[:, None] * axo, axis=1)
            if x is u:
                alpha[live] = np.bincount(owner, np.arctan2(rho, h), len(live)) / np.maximum(lens, 1)
            al = alpha[live][owner]
            d = rho * np.cos(al) - h * np.sin(al)
            d = np.where(h >= -self.tol, np.abs(d), np.inf)
            good = np.maximum.reduceat(d, _seg_starts(lens)) <= self.tol
            if x is u:
                good &= (alpha[live] > self.min_cone) & (alpha[live] < self.max_cone)
            ok[live[~good]] = False
            live = live[good]
        return ok

    def spheres(self):
        self.noseed = np.zeros(self.U, bool)   # per pass
        u, v, w = self.seed_triples()
        keep = (self.label[u] < 0) & (self.label[v] < 0) & (self.label[w] < 0)
        u, v, w = u[keep], v[keep], w[keep]
        if len(u) == 0:
            return
        def check(i):
            ok, c, r, mx = self._sphere_triples(u[i], v[i], w[i])
            res = np.zeros(len(i), bool)
            j = np.nonzero(ok)[0]
            if len(j):
                sup = self._sphere_support(u[i][j], v[i][j], w[i][j], c[j], r[j])
                # a single facet row (a one-fan cap, a strip next to a torus of the same
                # radius) only has two more facets around any three of its own; take
                # those when the three sit exactly on the sphere, as CAD tessellates
                res[j] = (sup >= MIN_SPHERE_SUPPORT) | ((sup >= 2) & (mx[j] <= 0.05 * self.tol))
            return res
        sel = self._first_ok_per_unit(u, check, 2)
        u, v, w = u[sel], v[sel], w[sel]
        order = np.lexsort((np.arange(len(u)), -self.uarea[u]))
        for i in order.tolist():
            if self.label[u[i]] >= 0 or self.label[v[i]] >= 0 or self.label[w[i]] >= 0:
                continue
            if self.noseed[u[i]]:
                continue
            m = _SphereModel(self)
            self.last_grown = None
            region = self.try_region(m, int(u[i]), [int(u[i]), int(v[i]), int(w[i])], {})
            if region is not None:
                self.accept(m, region)
            elif self.last_grown is not None:
                self.noseed[self.last_grown] = True

    def _patches(self, seeds):
        """Seed + 2-ring of free units (through free units) for every seed, as a CSR
        matrix (row i = patch of seeds[i], columns sorted)."""
        free = self.label < 0
        src = np.repeat(np.arange(self.U), np.diff(self.uadj_ptr))
        dst = self.uadj_idx
        m = free[src] & free[dst]
        B = coo_matrix((np.ones(int(m.sum()) + int(free.sum()), np.float32),
                        (np.concatenate((src[m], np.nonzero(free)[0])), np.concatenate((dst[m], np.nonzero(free)[0])))),
                       shape=(self.U, self.U)).tocsr()
        P = (B[seeds] @ B).tocsr()
        P.sort_indices()
        P.data[:] = 1   # membership, not path counts
        return P

    def _torus_screen(self, seeds, P):
        """Vectorised version of the closed-form torus initialisation in _TorusModel.fit on
        every seed's patch (rows of P). False where that fit certainly fails (patch
        vertices far off the torus), so the per-seed fit is not worth running. Looser
        than the real test (algebraic circle fit, 2x the residual limit)."""
        S = len(seeds)
        ok = np.zeros(S, bool)
        if S == 0:
            return ok
        cnt = np.diff(P.indptr)
        own = np.repeat(np.arange(S), cnt)
        cols = P.indices
        o = np.zeros(3)
        n, p = self.un, self.ucen - o
        mv = np.column_stack((np.cross(p, n), n))
        O = (mv[:, :, None] * mv[:, None, :]).reshape(-1, 36)
        G = np.asarray(P.astype(np.float64) @ O).reshape(S, 6, 6)
        sc = np.abs(G).max(axis=(1, 2)) + 1e-300
        ev, V = np.linalg.eigh(G / sc[:, None, None])
        # nearly degenerate null space: which line the closed form picks there is down to
        # rounding, so the screen cannot predict it -- let those through
        undecided = ev[:, 1] < 1.2 * np.maximum(ev[:, 0], 1e-300)
        V = V[:, :, 0]
        a, mL = V[:, :3], V[:, 3:]
        na = np.linalg.norm(a, axis=1)
        ok = (cnt >= 6) & (na > 1e-9)
        a = a / np.maximum(na, 1e-300)[:, None]
        mL = mL / np.maximum(na, 1e-300)[:, None]
        c = np.cross(a, mL)
        pos, lens = _ranges(self.vptr, cols)
        V = len(self.points)
        key = _uniq(np.repeat(own, lens) * V + self.vidx[pos])
        vo = key // V
        X = self.points[key % V] - o
        v = X - c[vo]
        h = np.einsum("ij,ij->i", v, a[vo])
        rho = np.linalg.norm(v - h[:, None] * a[vo], axis=1)
        nv = np.bincount(vo, minlength=S).astype(float)
        nv1 = np.maximum(nv, 1)
        mr = np.bincount(vo, rho, S) / nv1
        mh = np.bincount(vo, h, S) / nv1
        x, y = rho - mr[vo], h - mh[vo]
        z = x * x + y * y
        cols3 = (x, y, np.ones_like(x))
        A = np.empty((S, 3, 3))
        b = np.empty((S, 3))
        for i in range(3):
            b[:, i] = -np.bincount(vo, cols3[i] * z, S)
            for j in range(i, 3):
                A[:, i, j] = A[:, j, i] = np.bincount(vo, cols3[i] * cols3[j], S)
        scl = np.abs(A).max(axis=(1, 2)) + 1e-300
        good = np.abs(np.linalg.det(A / scl[:, None, None])) > 1e-14
        A[~good] = np.eye(3)
        sol = np.linalg.solve(A, b[:, :, None])[:, :, 0]
        cx, cy = -sol[:, 0] / 2, -sol[:, 1] / 2
        r2 = cx * cx + cy * cy - sol[:, 2]
        good &= r2 > 0
        r = np.sqrt(np.maximum(r2, 0))
        for _ in range(3):   # geometric Gauss-Newton steps, as _circle_fit
            dx, dy = x - cx[vo], y - cy[vo]
            d = np.maximum(np.hypot(dx, dy), 1e-300)
            J = (-dx / d, -dy / d, -np.ones_like(d))
            f = d - r[vo]
            A = np.empty((S, 3, 3))
            g = np.empty((S, 3))
            for i in range(3):
                g[:, i] = -np.bincount(vo, J[i] * f, S)
                for j in range(i, 3):
                    A[:, i, j] = A[:, j, i] = np.bincount(vo, J[i] * J[j], S)
            sing = np.abs(np.linalg.det(A / (np.abs(A).max(axis=(1, 2)) + 1e-300)[:, None, None])) <= 1e-14
            A[sing] = np.eye(3)
            st = np.linalg.solve(A, g[:, :, None])[:, :, 0]
            st[sing] = 0
            cx, cy, r = cx + st[:, 0], cy + st[:, 1], r + st[:, 2]
        res = np.abs(np.hypot(x - cx[vo], y - cy[vo]) - r[vo])
        mx = np.zeros(S)
        np.maximum.at(mx, vo, res)
        R = mr + cx
        # a patch that is a piece of a sphere leaves the torus axis undetermined (every
        # normal line passes through the centre): the closed form can't succeed there
        sok, _, _, smx = _group_sphere_fit(X, vo, S)
        spherical = sok & (smx <= self.tol)
        return (cnt >= 6) & ~spherical & (undecided | (ok & good & (mx <= 22 * self.tol) & (R >= 0.09 * r)))

    def tori(self):
        self.noseed = np.zeros(self.U, bool)   # per pass
        u, v, w = self.seed_triples()
        keep = (self.label[u] < 0) & (self.label[v] < 0) & (self.label[w] < 0)
        u = _uniq(u[keep])
        order = np.lexsort((u, -self.uarea[u]))
        seeds = u[order]
        P = self._patches(seeds)
        screen = self._torus_screen(seeds, P)
        n_acc = len(self.surfaces)
        for i, seed in enumerate(seeds.tolist()):
            if self.label[seed] >= 0 or self.noseed[seed]:
                continue
            # local patch: seed + 2-ring of free units
            if len(self.surfaces) == n_acc:
                patch = P.indices[P.indptr[i]:P.indptr[i + 1]]
            else:
                patch = np.array([seed])
                for _ in range(2):
                    ring = self.uadj_idx[_ranges(self.uadj_ptr, patch)[0]]
                    patch = np.union1d(patch, ring[self.label[ring] < 0])
            m = _TorusModel(self)
            self.last_grown = None
            region = None
            if len(patch) >= 6 and screen[i]:
                region = self.try_region(m, seed, patch, {})
            if region is not None:
                self.accept(m, region)
            else:
                self.noseed[patch] = True
                if self.last_grown is not None:
                    self.noseed[self.last_grown] = True

    def _sphere_triples(self, u, v, w):
        """Vectorised algebraic sphere fit through the vertices of each triple; True where
        all vertices are within 2*tol and the radius is sane."""
        n = len(u)
        Xs, owners = [], []
        for x in (u, v, w):
            pos, lens = _ranges(self.vptr, x)
            Xs.append(self.points[self.vidx[pos]])
            owners.append(np.repeat(np.arange(n), lens))
        X, own = np.concatenate(Xs), np.concatenate(owners)
        cnt = np.bincount(own, minlength=n).astype(float)
        mean = np.stack([np.bincount(own, X[:, k], n) for k in range(3)], 1) / cnt[:, None]
        Q = X - mean[own]
        z = (Q * Q).sum(1)
        cols = (Q[:, 0], Q[:, 1], Q[:, 2], np.ones(len(Q)))
        A = np.empty((n, 4, 4))
        b = np.empty((n, 4))
        for i in range(4):
            b[:, i] = -np.bincount(own, cols[i] * z, n)
            for j in range(i, 4):
                A[:, i, j] = A[:, j, i] = np.bincount(own, cols[i] * cols[j], n)
        scale = np.abs(A).max(axis=(1, 2)) + 1e-300
        ok = np.abs(np.linalg.det(A / scale[:, None, None])) > 1e-14
        A[~ok] = np.eye(4)
        sol = np.linalg.solve(A, b[:, :, None])[:, :, 0]
        c = -sol[:, :3] / 2
        r2 = (c * c).sum(1) - sol[:, 3]
        ok &= r2 > 0
        r = np.sqrt(np.maximum(r2, 0))
        res = np.abs(np.linalg.norm(Q - c[own], axis=1) - r[own])
        mx = np.zeros(n)
        np.maximum.at(mx, own, res)
        return ok & (mx <= 2 * self.tol) & (r > 10 * self.tol) & (r < 10 * self.diag), c + mean, r, mx

    def _ring_pairs(self, u, v, w):
        """(i, unit) for every free unit edge-adjacent to triple i = (u[i], v[i], w[i]) and
        not part of it."""
        n = len(u)
        own, nb = [], []
        for x in (u, v, w):
            pos, lens = _ranges(self.uadj_ptr, x)
            own.append(np.repeat(np.arange(n), lens))
            nb.append(self.uadj_idx[pos])
        own, nb = np.concatenate(own), np.concatenate(nb)
        m = (self.label[nb] < 0) & (nb != u[own]) & (nb != v[own]) & (nb != w[own])
        key = _uniq(own[m] * self.U + nb[m])
        return key // self.U, key % self.U

    def _sphere_support(self, u, v, w, c, r):
        """Number of units around each seed triple that lie on the triple's sphere (all
        vertices within tol, facet facing the right way). A real sphere patch of the
        tessellation continues past any three of its facets; a sphere that happens to fit
        three facets of a thread or an organic shape does not."""
        i, nb = self._ring_pairs(u, v, w)
        cnt = np.zeros(len(u), np.int64)
        if len(i) == 0:
            return cnt
        pos, lens = _ranges(self.vptr, nb)
        X = self.points[self.vidx[pos]]
        po = np.repeat(np.arange(len(i)), lens)
        res = np.abs(np.linalg.norm(X - c[i[po]], axis=1) - r[i[po]])
        fit = np.maximum.reduceat(res, _seg_starts(lens)) <= self.tol
        # facing: same side as the triple (area-weighted normal . radial at the centroids)
        def rad(units, cc):
            d = self.ucen[units] - cc
            return d / np.maximum(np.linalg.norm(d, axis=1), 1e-300)[:, None]
        sgn = np.zeros(len(u))
        for x in (u, v, w):
            sgn += self.uarea[x] * np.einsum("ij,ij->i", self.un[x], rad(x, c))
        sgn = np.where(sgn >= 0, 1.0, -1.0)
        face = np.einsum("ij,ij->i", self.un[nb], rad(nb, c[i])) * sgn[i] >= self.cos_half_step
        np.add.at(cnt, i, fit & face)
        return cnt

    def cones(self, defer=False, only=None):
        # per pass; a region just set aside as a cylinder is no place to start a cone
        self.noseed = self.deferred.copy() if defer else np.zeros(self.U, bool)
        u, v, w = self.seed_triples(only)
        keep = (self.label[u] < 0) & (self.label[v] < 0) & (self.label[w] < 0)
        u, v, w = u[keep], v[keep], w[keep]
        if len(u) == 0:
            return
        sel = self._first_ok_per_unit(u, lambda i: self._cone_triples(u[i], v[i], w[i]), 2)
        u, v, w = u[sel], v[sel], w[sel]
        order = np.lexsort((np.arange(len(u)), -self.uarea[u]))
        noseed = self.noseed
        for i in order.tolist():
            if self.label[u[i]] >= 0 or self.label[v[i]] >= 0 or self.label[w[i]] >= 0 or noseed[u[i]]:
                continue
            m = _ConeModel(self)
            self.last_grown = None
            region = self.try_region(m, int(u[i]), [int(u[i]), int(v[i]), int(w[i])], {"polish": True})
            if region is not None:
                self.accept_or_defer(m, region, defer)
            elif self.last_grown is not None and self.hopeless:
                # don't restart from inside a patch that fitted one cone but could not
                # even cover the minimum arc (cone growth refits are the most expensive)
                noseed[self.last_grown] = True

    def merge(self):
        """Merge edge-adjacent regions of the same kind that are the same surface."""
        n = len(self.surfaces)
        if n < 2:
            return
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        lab = self.label
        src = np.repeat(np.arange(self.U), np.diff(self.uadj_ptr))
        dst = self.uadj_idx
        la, lb = lab[src], lab[dst]
        m = (la >= 0) & (lb >= 0) & (la < lb)
        pairs = np.unique(np.column_stack((la[m], lb[m])), axis=0) if m.any() else np.zeros((0, 2), int)
        changed = False
        for i, j in pairs.tolist():
            ri, rj = find(i), find(j)
            if ri == rj or self.kinds[ri] != self.kinds[rj]:
                continue
            if not self._same(self.surfaces[ri], self.surfaces[rj]):
                continue
            units = np.nonzero((lab == ri) | (lab == rj) | np.isin(lab, [k for k in range(n) if find(k) in (ri, rj)]))[0]
            mdl = {"cyl": _CylModel, "cone": _ConeModel, "sphere": _SphereModel, "torus": _TorusModel}[self.kinds[ri]](self)
            if not mdl.fit(units, polish=True):
                continue
            X, _ = mdl.unit_points(units)
            if np.abs(mdl.residual(X)).max() > self.tol:
                continue
            parent[rj] = ri
            self.surfaces[ri] = mdl.surface()
            self.models[ri] = mdl
            changed = True
        if changed:
            roots = sorted({find(k) for k in range(n)})
            remap = {r: i for i, r in enumerate(roots)}
            newlab = np.array([remap[find(k)] for k in range(n)])
            self.label = np.where(lab >= 0, newlab[np.maximum(lab, 0)], -1)
            self.surfaces = [self.surfaces[r] for r in roots]
            self.kinds = [self.kinds[r] for r in roots]
            self.models = [self.models[r] for r in roots]

    def region_sagitta(self, l):
        """(sagitta, facet step rad, closes all round?) of curved region l. The sagitta
        is the chordal deviation the exporter allowed: r (1 - cos(step / 2)) for the
        typical facet step around a cylinder / cone axis; for spheres and tori, 1.5x the
        typical deviation at the facet centroids (a triangle's centroid sits a bit inside
        its maximum). Step and closure are only defined for cylinders and cones."""
        kind, mdl = self.kinds[l], self.models[l]
        units = np.nonzero(self.label == l)[0]
        X, lens = mdl.unit_points(units)
        if kind in ("cyl", "cone"):
            step = float(np.median(mdl._widths(units, X, lens)))
            v = X - mdl.c
            h = v @ mdl.a
            rad = float(np.linalg.norm(v - np.outer(h, mdl.a), axis=1).mean())
            closed = _span(mdl.azimuth(X)) >= 2 * np.pi - 1.5 * step
            return rad * (1 - np.cos(step / 2)), step, closed
        dev = np.abs(mdl.residual(self.ucen[units]))
        return 1.5 * float(np.median(dev)), None, False

    def reject_misfits(self):
        """Drop curved regions whose facets cut far inside the fitted surface although
        their vertices lie on it: polyhedral geometry that happens to have its corners on
        a cylinder (a wide face flanked by two equal narrow chamfers -- any isosceles
        trapezoid profile is 'cyclic' -- those are already refused at growth time, see
        _Model.uneven_facets), or a gross fit. The deviation of a facet's centre from
        the surface is ~2/3 of its sagitta.
          * any curved region: deviation over MISFIT_PART_RATIO x the part's typical one
            (triangle-weighted median over its curved regions) and over MISFIT_MIN mm;
          * with fewer than 3 curved regions to compare, deviation over MISFIT_ABS mm.
        Full-turn cylinders are left to reject_polygon_prisms (and min_cyl_facets).
        (Exporters often limit the facet angle as well as the sagitta, so big radii get
        bigger sagittas: the part-relative limit is deliberately loose.)"""
        cen = self.P.mean(1)
        info = {}
        for l, kind in enumerate(self.kinds):
            if kind == "plane":
                continue
            units = np.nonzero(self.label == l)[0]
            tris = self.tri_order[_ranges(self.tptr, units)[0]]
            own = np.repeat(np.arange(len(units)), self.ntri[units])
            dev = np.zeros(len(units))
            np.maximum.at(dev, own, np.abs(self.models[l].residual(cen[tris])))
            info[l] = (kind, dev, len(tris))
        if not info:
            return
        devmax = np.array([v[1].max() for v in info.values()])
        w = np.array([v[2] for v in info.values()], float)
        o = np.argsort(devmax)
        cw = np.cumsum(w[o])
        typ = float(devmax[o][np.searchsorted(cw, cw[-1] / 2)])
        drop = []
        for (l, (kind, dev, ntri)), dmax in zip(info.items(), devmax):
            if kind == "cyl" and self.region_sagitta(l)[2]:
                continue   # a full turn: polygonal prism or not is reject_polygon_prisms' call
            if len(info) >= 3:
                if dmax > MISFIT_PART_RATIO * typ and dmax > MISFIT_MIN:
                    drop.append(l)
            elif dmax > MISFIT_ABS:
                drop.append(l)
        self._drop_surfaces(drop)

    def _drop_surfaces(self, drop):
        if not drop:
            return
        keep = [l for l in range(len(self.surfaces)) if l not in drop]
        remap = np.full(len(self.surfaces) + 1, -1, np.int64)
        remap[keep] = np.arange(len(keep))
        self.label = remap[self.label]          # -1 stays -1 (last entry)
        self.surfaces = [self.surfaces[l] for l in keep]
        self.kinds = [self.kinds[l] for l in keep]
        self.models = [self.models[l] for l in keep]

    def reject_polygon_prisms(self, min_facets=None):
        """A regular polygonal prism (octagon, 10-, 12-gon) fits a cylinder exactly; only
        the tessellation density tells it from a coarsely exported hole or boss.

        Exporters tessellate to a chordal (sag) tolerance, often with a maximum angle
        too, so the sagitta r (1 - cos(step/2)) is about the same on all genuinely curved
        faces of a part, whatever their radius (small holes get few facets, big ones
        many). A full-turn 'cylinder' with coarse facets (<= POLY_MAX_FACETS per turn) whose
        sagitta is over POLY_SAG_RATIO x the part's typical one (median over its other
        curved regions -- partial fillets, cones, spheres, tori) is a prism. With no other
        curved region to compare against, only a polygon no exporter would produce is
        rejected: sagitta > POLY_ABS_SAG and <= POLY_ABS_FACETS facets per turn.

        min_facets (int) overrides all this: full-turn cylinders with fewer facets per
        turn than that are dropped, others kept."""
        info = {}
        for l, kind in enumerate(self.kinds):
            if kind in ("cyl", "cone", "sphere", "torus"):
                info[l] = self.region_sagitta(l)
        cand = [l for l, (sag, step, closed) in info.items()
                if self.kinds[l] == "cyl" and closed]
        facets = {l: 2 * np.pi / max(info[l][1], 1e-9) for l in cand}   # per full turn
        drop = []
        if min_facets is not None:
            drop = [l for l in cand if facets[l] < min_facets - 0.5]
        else:
            cand = [l for l in cand if facets[l] < POLY_MAX_FACETS + 0.5]
            ref = [info[l][0] for l in info if l not in cand]
            typ = float(np.median(ref)) if ref else None
            for l in cand:
                sag, step, _ = info[l]
                if typ is not None:
                    if sag > POLY_SAG_RATIO * max(typ, 1e-12):
                        drop.append(l)
                elif sag > POLY_ABS_SAG and facets[l] < POLY_ABS_FACETS + 0.5:
                    drop.append(l)
        self._drop_surfaces(drop)

    def _same(self, A, B):
        ang = np.radians(0.5)
        if isinstance(A, Cylinder):
            if abs(A.radius - B.radius) > self.tol:
                return False
            if abs(A.axis @ B.axis) < np.cos(ang):
                return False
            d = B.origin - A.origin
            return np.linalg.norm(d - (d @ A.axis) * A.axis) <= self.tol
        if isinstance(A, Torus):
            return (abs(A.major - B.major) <= self.tol and abs(A.minor - B.minor) <= self.tol
                    and abs(A.axis @ B.axis) >= np.cos(ang) and np.linalg.norm(A.center - B.center) <= self.tol)
        if isinstance(A, Sphere):
            return abs(A.radius - B.radius) <= self.tol and np.linalg.norm(A.center - B.center) <= self.tol
        if abs(A.half_angle - B.half_angle) > np.radians(0.5) or A.axis @ B.axis < np.cos(ang):
            return False
        return np.linalg.norm(A.origin - B.origin) <= max(self.tol, self.tol / max(np.sin(A.half_angle), 1e-3))

    def planes(self):
        """Leftover triangles -> planar regions (full `tol`), 2+ triangles become planes."""
        labels = self.label[self.unit]
        free = labels < 0
        if not free.any():
            return labels
        nbr = self.nbr.copy()
        nbr[~free] = -1
        nbr[(nbr >= 0) & ~free[np.maximum(nbr, 0)]] = -1
        pu = _planar_units(self.P, self.tn, self.tarea, nbr, self.tol, self.angle_deg)
        tris = np.nonzero(free)[0]
        grp, inv, cnt = np.unique(pu[tris], return_inverse=True, return_counts=True)
        keep = cnt[inv] >= 2
        tris, inv = tris[keep], inv[keep]
        if len(tris) == 0:
            return labels
        _, inv = np.unique(inv, return_inverse=True)
        G = int(inv.max()) + 1
        X = self.P[tris].reshape(-1, 3)
        own = np.repeat(inv, 3)
        cnt = np.bincount(own, minlength=G).astype(float)
        cen = np.stack([np.bincount(own, X[:, k], G) for k in range(3)], 1) / cnt[:, None]
        d = X - cen[own]
        C = np.zeros((G, 3, 3))
        for i in range(3):
            for j in range(i, 3):
                C[:, i, j] = C[:, j, i] = np.bincount(own, d[:, i] * d[:, j], G)
        vec = np.linalg.eigh(C)[1][:, :, 0]
        mn = np.stack([np.bincount(inv, self.tn[tris, k] * self.tarea[tris], G) for k in range(3)], 1)
        vec[np.einsum("ij,ij->i", vec, mn) < 0] *= -1
        mn /= np.linalg.norm(mn, axis=1)[:, None]
        # least squares minimises the RMS, the contract is the MAX distance: try both the
        # LS normal and the area-weighted mean normal, centre each on its residual range
        # and keep whichever has the smaller worst-case distance
        best = None
        for nv in (vec, mn):
            dd = np.einsum("ij,ij->i", X - cen[own], nv[own])
            lo = np.full(G, np.inf)
            hi = np.full(G, -np.inf)
            np.minimum.at(lo, own, dd)
            np.maximum.at(hi, own, dd)
            cand = (nv, cen + nv * ((lo + hi) / 2)[:, None], (hi - lo) / 2)
            if best is None:
                best = cand
            else:
                pick = cand[2] < best[2]
                best = tuple(np.where(pick[:, None] if b.ndim == 2 else pick, c, b) for b, c in zip(best, cand))
        vec, cen = best[0], best[1]
        base = len(self.surfaces)
        labels[tris] = base + inv
        for i in range(G):
            self.surfaces.append(Plane(cen[i].copy(), vec[i].copy()))
            self.kinds.append("plane")
        return labels


def _surf_residual(sf, X):
    """Signed distance-like residual of points X to a result surface."""
    if isinstance(sf, Plane):
        return (X - sf.origin) @ sf.normal
    if isinstance(sf, Cylinder):
        v = X - sf.origin
        return np.linalg.norm(v - np.outer(v @ sf.axis, sf.axis), axis=1) - sf.radius
    if isinstance(sf, Cone):
        v = X - sf.origin
        h = v @ sf.axis
        rho = np.linalg.norm(v - np.outer(h, sf.axis), axis=1)
        return np.where(h >= 0, rho * np.cos(sf.half_angle) - h * np.sin(sf.half_angle), np.inf)
    if isinstance(sf, Sphere):
        return np.linalg.norm(X - sf.center, axis=1) - sf.radius
    v = X - sf.center
    h = v @ sf.axis
    rho = np.linalg.norm(v - np.outer(h, sf.axis), axis=1)
    return np.hypot(rho - sf.major, h) - sf.minor


def _n_normal_dirs(nrm, sep_deg, enough):
    """Number of distinct directions among unit normals (greedy clusters of radius
    sep_deg), counting no further than `enough`."""
    cos_sep = np.cos(np.radians(sep_deg))
    reps = np.zeros((0, 3))
    left = nrm
    while len(left) and len(reps) < enough:
        reps = np.vstack((reps, left[:1]))
        left = left[(left @ reps[-1]) < cos_sep]
    return len(reps)


def _flatten_bent_pairs(P, tn, labels, surfaces, tol):
    """After boundary refinement a curved region can be left with facets in fewer than
    MIN_FACET_DIRS directions (the rest went to the planes they really belong to). That
    is no longer a curved surface: it becomes a plane if its facets are coplanar within
    tol, otherwise its triangles stay facets (label -1). Surface indices are kept."""
    labels = labels.copy()
    for l, sf in enumerate(surfaces):
        if isinstance(sf, Plane):
            continue
        tris = np.nonzero(labels == l)[0]
        if len(tris) and _n_normal_dirs(tn[tris], MIN_DIR_SEP_DEG, MIN_FACET_DIRS) >= MIN_FACET_DIRS:
            continue
        if len(tris) == 0:
            continue
        X = P[tris].reshape(-1, 3)
        nrm = tn[tris].mean(0)
        nrm /= np.linalg.norm(nrm)
        org = X.mean(0)
        d = (X - org) @ nrm
        if np.ptp(d) <= 2 * tol:
            surfaces[l] = Plane(org + nrm * (d.max() + d.min()) / 2, nrm)
        else:
            labels[tris] = -1
    return labels


def _refine_boundaries(P, nbr, labels, surfaces, rounds=3):
    """Hand boundary triangles to the neighbouring region whose surface they fit much
    better. Along tangent joints (fillet -> flat face, cylinder -> corner sphere) both
    surfaces pass the tolerance test, and growth order decides who gets the first row;
    the triangle's far vertex says which surface it really belongs to. A move is undone if
    it would disconnect the region it leaves."""
    labels = labels.copy()
    F = len(labels)
    for _ in range(rounds):
        t = np.repeat(np.arange(F), 3)
        o = nbr.ravel()
        m = (o >= 0) & (labels[t] >= 0)
        t, o = t[m], o[m]
        m = (labels[o] >= 0) & (labels[o] != labels[t])
        t, o = t[m], o[m]
        if len(t) == 0:
            break
        cand = {}
        for ti, oi in zip(t.tolist(), o.tolist()):
            cand.setdefault(ti, set()).add(int(labels[oi]))
        moves = {}
        for ti, others in sorted(cand.items()):
            X = P[ti]
            own = np.abs(_surf_residual(surfaces[labels[ti]], X)).max()
            best, bres = None, own
            for L in sorted(others):
                r = np.abs(_surf_residual(surfaces[L], X)).max()
                if r < 0.25 * bres:
                    best, bres = L, r
            if best is not None and own > 1e-12:
                moves[ti] = best
        if not moves:
            break
        old = labels.copy()
        for ti, L in moves.items():
            labels[ti] = L
        # connectivity: any region that lost triangles must stay one piece, and every
        # moved triangle must touch its new region
        lost = {int(old[ti]) for ti in moves}
        bad = set()
        for L in lost:
            tris = np.nonzero(labels == L)[0]
            if len(tris) == 0:
                bad.add(L)
                continue
            idx = -np.ones(F, np.int64)
            idx[tris] = np.arange(len(tris))
            nb = nbr[tris]
            ok = (nb >= 0) & (labels[np.maximum(nb, 0)] == L)
            r = np.repeat(np.arange(len(tris)), 3)[ok.ravel()]
            c = idx[nb[ok]]
            g = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(tris), len(tris)))
            if connected_components(g, directed=False)[0] > 1:
                bad.add(L)
        for ti in moves:
            if int(old[ti]) in bad:
                labels[ti] = old[ti]
        if np.array_equal(labels, old):
            break
    return labels


def segment(points: np.ndarray, faces: np.ndarray, tol: float, angle_deg: float = 0.5,
            *, max_step_deg: float = 46.0, max_tilt_deg: float = 4.0, min_span_deg: float = 20.0,
            cones: bool = True, min_cyl_facets: int | None = None) -> tuple[np.ndarray, list]:
    """points (V,3) float64 welded vertices; faces (F,3) int, consistently wound CCW-outward.
    Returns labels (F,) int and surfaces list. labels[i] = index into surfaces, or -1 for a
    freeform triangle that matches nothing (it will stay its own flat facet).
    Every region (triangles sharing a label) is edge-connected.

    max_step_deg  widest facet (angle around the axis) still treated as part of a tessellated
                  cylinder/cone; 46 accepts octagons and 2-facet 90-degree fillets, rejects
                  hexagonal prisms.
    max_tilt_deg  allowed deviation of a facet normal from the ideal (perpendicular to a
                  cylinder axis, constant angle to a cone axis).
    min_span_deg  minimum angular coverage of a cylinder/cone region.
    min_cyl_facets  None: tell polygonal prisms from coarse cylinders automatically (by
                  the part's chordal tolerance, see _Seg.reject_polygon_prisms). An int:
                  full-turn cylinders with fewer facets per turn than this are left as
                  flat facets, all others kept (0 keeps every one)."""
    faces = np.asarray(faces, np.int64)
    F = len(faces)
    if F == 0:
        return np.zeros(0, np.int64), []
    s = _Seg(points, faces, tol, angle_deg, max_step_deg, max_tilt_deg, min_span_deg)
    s.build_units()
    s.models = []
    s.cylinders(defer=True)
    if cones:
        s.cones(defer=True)
    s.blocked[:] = False
    s.tori()
    s.spheres()
    if s.deferred.any():
        # what the doubly-curved passes left of the set-aside regions really is cylindrical
        s.accept_deferred()
        s.cylinders(only=s.deferred & (s.label < 0))
        if cones:
            s.cones(only=s.deferred & (s.label < 0))
    s.merge()
    s.reject_polygon_prisms(min_cyl_facets)
    s.reject_misfits()
    labels = s.planes()
    labels = _refine_boundaries(s.P, s.nbr, labels, s.surfaces)
    labels = _flatten_bent_pairs(s.P, s.tn, labels, s.surfaces, s.tol)
    return labels, s.surfaces
