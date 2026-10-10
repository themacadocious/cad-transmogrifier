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
  4. whatever planar units are left (>= 2 triangles) become planes; single triangles that
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


# curved regions are grown with a tighter tolerance than `tol`: the vertices of a
# tangent neighbour (flat face next to a fillet, sphere corner after a fillet) leave the
# surface quadratically, so a tighter band stops growth closer to the true boundary
GROW_TOL_FRAC = 0.5
UNIT_TOL_FRAC = 0.05


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


def _circle_fit(xy):
    """Algebraic (Kasa) circle fit followed by a few geometric Gauss-Newton steps."""
    m = xy.mean(0)
    q = xy - m
    A = np.column_stack((q, np.ones(len(q))))
    b = -(q * q).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = -sol[:2] / 2
    r2 = c @ c - sol[2]
    if not np.isfinite(r2) or r2 <= 0:
        return None
    r = np.sqrt(r2)
    for _ in range(3):
        d = q - c
        rho = np.linalg.norm(d, axis=1)
        if np.any(rho < 1e-12):
            break
        J = np.column_stack((-d / rho[:, None], -np.ones(len(q))))
        res = rho - r
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        c = c + step[:2]
        r = r + step[2]
    if not np.isfinite(r) or r <= 0:
        return None
    return c + m, float(r)


def _lm(resfun, nparam, iters=10):
    """Tiny Levenberg-Marquardt on a residual function of a parameter offset vector."""
    x = np.zeros(nparam)
    r = resfun(x)
    cost = r @ r
    lam = 1e-6
    h = 1e-7
    for _ in range(iters):
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
                x, r, cost = x + step, rn, cn
                lam = max(lam / 10, 1e-9)
                improved = True
                break
            lam *= 10
        if not improved or np.linalg.norm(step) < 1e-12:
            break
    return x


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
        cand = np.unique(cand[cand >= 0])
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
        # facet angular width around the axis must be at most max_step
        az = self.azimuth(X)
        az0 = np.repeat(self.azimuth(s.ucen[sub]), lens)
        d = np.mod(az - az0 + np.pi, 2 * np.pi) - np.pi
        width = np.maximum.reduceat(d, st) - np.minimum.reduceat(d, st)
        good &= width <= s.max_step
        ok[np.nonzero(ok)[0][~good]] = False
        return ok

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

    def doubly_curved_band(self, units, X, nd, span):
        """True if the region is a short band cut out of a doubly-curved surface (a row of
        a sphere / torus / revolved surface: two vertex rings always fit a cylinder or cone
        exactly). Signature: at least half of its *end* boundary (edges across the
        generators) continues smoothly (facet angle <= max_step) into the neighbouring
        surface, and it is short: length < 8 facet widths. Real cylinders end at sharp
        edges, or are long compared with their facets (fillet running into a corner blend)."""
        s = self.s
        tris = s.tri_order[np.concatenate([np.arange(s.tptr[u], s.tptr[u + 1]) for u in units.tolist()])]
        inreg = np.zeros(len(s.faces), bool)
        inreg[tris] = True
        nb = s.nbr[tris]
        k = np.nonzero((nb >= 0) & ~inreg[np.maximum(nb, 0)] | (nb < 0))
        t, e = tris[k[0]], k[1]
        if len(t) == 0:
            return False
        p0 = s.points[s.faces[t, e]]
        p1 = s.points[s.faces[t, (e + 1) % 3]]
        d = p1 - p0
        ln = np.linalg.norm(d, axis=1)
        g = self.gen_dir((p0 + p1) / 2)
        end = np.abs(np.einsum("ij,ij->i", d, g)) < 0.5 * ln
        if not end.any():
            return False
        # typical facet-to-facet turn inside the region; a smooth continuation turns by
        # about as much, a real edge (cap, chamfer, plane) turns by more
        inner = nb[inreg[np.maximum(nb, 0)] & (nb >= 0)]
        tin = np.repeat(tris, 3).reshape(-1, 3)[inreg[np.maximum(nb, 0)] & (nb >= 0)]
        cin = np.einsum("ij,ij->i", s.tn[tin], s.tn[inner])
        turns = np.arccos(np.clip(cin[cin < np.cos(np.radians(0.3))], -1, 1))
        step = np.median(turns) if len(turns) else s.max_step
        lim = np.cos(min(max(1.3 * step, np.radians(15)), s.max_step))
        other = nb[k]
        smooth = (other >= 0) & (np.einsum("ij,ij->i", s.tn[t], s.tn[np.maximum(other, 0)]) >= lim)
        if ln[end & smooth].sum() < 0.5 * ln[end].sum():
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
        a0, c0, R0 = self.a, self.c, self.R
        e1, e2 = self.e1, self.e2

        def res(x):
            a = _unit(a0 + x[0] * e1 + x[1] * e2)
            c = c0 + x[2] * e1 + x[3] * e2
            v = X - c
            r = v - np.outer(v @ a, a)
            return np.linalg.norm(r, axis=1) - (R0 + x[4])

        x = _lm(res, 5)
        a = _unit(a0 + x[0] * e1 + x[1] * e2)
        c = c0 + x[2] * e1 + x[3] * e2
        before = np.abs(res(np.zeros(5))).max()
        if np.abs(res(x)).max() <= max(before, self.s.tol):
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
        nd = _n_dirs(np.arctan2(s.un[units] @ self.e2, s.un[units] @ self.e1), np.radians(0.5))
        if nd < 2 or (nd == 2 and not self._tangent_neighbour(units)):
            return False
        return not self.doubly_curved_band(units, X, nd, span)

    def _tangent_neighbour(self, units):
        """A 2-facet arc is accepted only if a neighbouring face continues it tangentially
        (fillet between two flat faces): its normal equals the radial direction at the
        shared vertices."""
        s = self.s
        nb = np.unique(s.uadj_idx[_ranges(s.uadj_ptr, units)[0]])
        nb = nb[~np.isin(nb, units)]
        if len(nb) == 0:
            return False
        regv = np.unique(s.vidx[_ranges(s.vptr, units)[0]])
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

        x = _lm(res, 6)
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
        if span < s.min_span:
            return False
        nrm = s.un[units]
        nd = _n_dirs(np.arctan2(nrm @ self.e2, nrm @ self.e1), np.radians(0.5))
        if nd < 3:
            return False
        return not self.doubly_curved_band(units, X, nd, span)

    def slant_cos(self):
        return self.ca

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

            x = _lm(res, 4)
            if np.abs(res(x)).max() <= max(np.abs(res(np.zeros(4))).max(), s.tol):
                self.set(c0 + x[:3], R0 + x[3])
        nrm, w = s.un[units], s.uarea[units]
        self.sign = 1.0 if (w * np.einsum("ij,ij->i", nrm, self.radial(s.ucen[units]))).sum() >= 0 else -1.0
        return True

    def quick_ok(self, units):
        return len(units) >= 3

    def valid(self, units):
        s = self.s
        if not (s.tol * 10 < self.R < 10 * s.diag) or len(units) < 4:
            return False
        nrm = s.un[units]
        # angular extent of the patch
        if nrm.min(0) @ nrm.max(0) > 2 or np.min(nrm @ nrm.T) > np.cos(s.min_span):
            return False
        # doubly curved: normals must NOT all lie on one circle of the Gauss sphere
        # (a single row of a cylinder / cone / revolved surface fits a sphere exactly)
        m = nrm.mean(0)
        ev, vec = np.linalg.eigh((nrm - m).T @ (nrm - m))
        dev = np.abs((nrm - m) @ vec[:, 0]).max()
        return dev >= 0.01

    def surface(self):
        return Sphere(self.c.copy(), float(self.R), bool(self.sign > 0))


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
        self.min_cone, self.max_cone = np.radians(2.0), np.radians(88.0)
        P = self.points[self.faces]
        self.P = P
        cr = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        ln = np.linalg.norm(cr, axis=1)
        self.tarea = ln / 2
        self.tn = cr / np.maximum(ln, 1e-300)[:, None]
        self.nbr = edge_adjacency(self.faces)
        self.diag = float(np.linalg.norm(np.ptp(self.points, axis=0))) if len(self.points) else 0.0
        self.angle_deg = angle_deg

    # -------------------------------------------------- unit tables
    def build_units(self):
        F, V = len(self.faces), len(self.points)
        # units are built with a much tighter tolerance than planes: only exactly flat
        # facets (both triangles of a strip quad, a real flat face) stay together, so a
        # sliver of a neighbouring curved surface never gets welded onto a facet strip
        noise = 5e-7 * float(np.abs(self.points).max()) if len(self.points) else 0.0
        unit = _planar_units(self.P, self.tn, self.tarea, self.nbr, max(UNIT_TOL_FRAC * self.tol, noise),
                             self.angle_deg)
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
        key = np.unique(np.repeat(unit, 3) * V + self.faces.ravel())
        self.vidx = key % V
        self.vptr = np.searchsorted(key // V, np.arange(U + 1))
        # unit adjacency
        r = np.repeat(unit, 3)
        c = self.nbr.ravel()
        ok = c >= 0
        r, c = r[ok], unit[c[ok]]
        ok = r != c
        key = np.unique(r[ok] * U + c[ok])
        self.uadj_idx = key % U
        self.uadj_ptr = np.searchsorted(key // U, np.arange(U + 1))
        self.label = np.full(U, -1, np.int64)   # unit -> surface index
        self.noseed = np.zeros(U, bool)
        self.surfaces = []
        self.kinds = []

    # -------------------------------------------------- seeds
    def seed_triples(self, k_near=8):
        """Candidate seed triples (u, v, w): v, w among the k_near nearest-normal neighbour
        units of u (normal angle within max_step, not parallel), v and w not parallel either.
        Sorted by u, then nearest pairs first."""
        U = self.U
        e = np.zeros(0, np.int64)
        src = np.repeat(np.arange(U), np.diff(self.uadj_ptr))
        dst = self.uadj_idx
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
            self._pe_key, first = np.unique(key, return_index=True)
            self._pe_dir = d[first]
        key = a * self.U + b
        i = np.minimum(np.searchsorted(self._pe_key, key), len(self._pe_key) - 1)
        found = self._pe_key[i] == key
        return np.where(found[:, None], self._pe_dir[i], 0.0)

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
        while True:
            pos, _ = _ranges(s.uadj_ptr, new)
            cand = np.unique(s.uadj_idx[pos])
            cand = cand[~seen[cand] & (s.label[cand] < 0)]
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
                if not model.fit(reg, polish=True):
                    model.__dict__.update(saved)
                grown_at_fit = len(region)
                pos, _ = _ranges(s.uadj_ptr, reg)
                ring = np.unique(s.uadj_idx[pos])
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
        return np.array(sorted(region))

    def try_region(self, model, seed, init_units, fit_kw):
        """Initial fit on init_units, grow, polish, re-grow fixed. Region or None."""
        if not model.fit(np.asarray(init_units), **fit_kw):
            return None
        X, _ = model.unit_points(np.asarray(init_units))
        if np.abs(model.residual(X)).max() > self.tol:
            return None
        seedarr = np.array([seed])
        if not model.test(seedarr).all():
            return None
        region = self.grow(model, seedarr, refit=True)
        self.last_grown = region
        if len(region) < 2 or not model.quick_ok(region):
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
        if len(region) < 2 or not model.valid(region):
            return None
        return region

    def accept(self, model, region):
        self.label[region] = len(self.surfaces)
        self.surfaces.append(model.surface())
        self.kinds.append(model.kind)
        self.models.append(model)

    def cylinders(self):
        u, v, w = self.seed_triples()
        nu, nv, nw = self.un[u], self.un[v], self.un[w]
        ax1 = np.cross(nu, nv)
        ax2 = np.cross(nu, nw)
        ax1 /= np.linalg.norm(ax1, axis=1)[:, None]
        ax2 /= np.linalg.norm(ax2, axis=1)[:, None]
        dot = np.einsum("ij,ij->i", ax1, ax2)
        agree = np.abs(dot) >= np.cos(np.radians(2.0))
        u, v, w = u[agree], v[agree], w[agree]
        ax = ax1[agree] + ax2[agree] * np.sign(dot[agree])[:, None]
        ok = self._circle_triples(u, v, w, ax)
        u, v, w, ax = u[ok], v[ok], w[ok], ax[ok]
        # prefer triples whose shared edges run along the axis (facet strips of a
        # tessellated cylinder share generator lines; a row of a sphere/torus does not)
        along = (np.abs(np.einsum("ij,ij->i", self._pair_edge_dir(u, v), ax / np.linalg.norm(ax, axis=1)[:, None])) > 0.99) \
            & (np.abs(np.einsum("ij,ij->i", self._pair_edge_dir(u, w), ax / np.linalg.norm(ax, axis=1)[:, None])) > 0.99)
        o = np.lexsort((np.arange(len(u)), ~along, u))
        u, v, w, ax = u[o], v[o], w[o], ax[o]
        sel = self._first_per_unit(u, 2)
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
                    self.accept(m, region)
                    break

    def _circle_triples(self, u, v, w, ax):
        """Vectorised check that the vertices of each triple lie on one circle around `ax`
        (algebraic circle fit in the plane perpendicular to the axis)."""
        n = len(u)
        if n == 0:
            return np.zeros(0, bool)
        ax = ax / np.linalg.norm(ax, axis=1)[:, None]
        h = np.where(np.abs(ax[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        e1 = np.cross(ax, h)
        e1 /= np.linalg.norm(e1, axis=1)[:, None]
        e2 = np.cross(ax, e1)
        Xs, owners = [], []
        for x in (u, v, w):
            pos, lens = _ranges(self.vptr, x)
            Xs.append(self.points[self.vidx[pos]])
            owners.append(np.repeat(np.arange(n), lens))
        X, own = np.concatenate(Xs), np.concatenate(owners)
        cnt = np.bincount(own, minlength=n).astype(float)
        mean = np.stack([np.bincount(own, X[:, k], n) for k in range(3)], 1) / cnt[:, None]
        Q = X - mean[own]
        x = np.einsum("ij,ij->i", Q, e1[own])
        y = np.einsum("ij,ij->i", Q, e2[own])
        z = x * x + y * y
        cols = (x, y, np.ones_like(x))
        A = np.empty((n, 3, 3))
        b = np.empty((n, 3))
        for i in range(3):
            b[:, i] = -np.bincount(own, cols[i] * z, n)
            for j in range(i, 3):
                A[:, i, j] = A[:, j, i] = np.bincount(own, cols[i] * cols[j], n)
        ok = np.abs(np.linalg.det(A)) > 1e-18
        A[~ok] = np.eye(3)
        sol = np.linalg.solve(A, b[:, :, None])[:, :, 0]
        cx, cy = -sol[:, 0] / 2, -sol[:, 1] / 2
        r2 = cx * cx + cy * cy - sol[:, 2]
        ok &= r2 > 0
        r = np.sqrt(np.maximum(r2, 0))
        res = np.abs(np.hypot(x - cx[own], y - cy[own]) - r[own])
        mx = np.zeros(n)
        np.maximum.at(mx, own, res)
        # Kasa is slightly biased on short arcs; allow a little slack, the real check follows
        return ok & (mx <= 2 * self.tol) & (r > 10 * self.tol) & (r < 10 * self.diag)

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
        maxres = np.zeros(len(u))
        for x in (u, v, w):
            pos, lens = _ranges(self.vptr, x)
            owner = np.repeat(np.arange(len(x)), lens)
            X = self.points[self.vidx[pos]] - apex[owner]
            h = np.einsum("ij,ij->i", X, ax[owner])
            rho = np.linalg.norm(X - h[:, None] * ax[owner], axis=1)
            ang = np.arctan2(rho, h)
            # residuals relative to the median opening angle of the triple are checked
            # after the alpha estimate below; here store h/rho for it
            if x is u:
                alpha = np.zeros(len(u))
                np.maximum.at(alpha, owner, 0)
                a_sum = np.bincount(owner, ang, len(u)) / np.maximum(lens, 1)
                alpha = a_sum
            d = rho * np.cos(alpha[owner]) - h * np.sin(alpha[owner])
            d = np.where(h >= -self.tol, np.abs(d), np.inf)
            np.maximum.at(maxres, owner, d)
        ok &= maxres <= self.tol
        ok &= (alpha > self.min_cone) & (alpha < self.max_cone)
        return ok

    def spheres(self):
        u, v, w = self.seed_triples()
        keep = (self.label[u] < 0) & (self.label[v] < 0) & (self.label[w] < 0)
        u, v, w = u[keep], v[keep], w[keep]
        if len(u) == 0:
            return
        # centre: least-squares meeting point of the three facet normal lines
        A = np.zeros((len(u), 3, 3))
        b = np.zeros((len(u), 3))
        for x in (u, v, w):
            n = self.un[x]
            Pm = np.eye(3)[None] - n[:, :, None] * n[:, None, :]
            A += Pm
            b += np.einsum("ijk,ik->ij", Pm, self.ucen[x])
        det = np.linalg.det(A)
        ok = np.abs(det) > 1e-9
        A[~ok] = np.eye(3)
        c = np.linalg.solve(A, b[:, :, None])[:, :, 0]
        R = np.zeros(len(u))
        mx = np.zeros(len(u))
        mn = np.full(len(u), np.inf)
        for x in (u, v, w):
            pos, lens = _ranges(self.vptr, x)
            own = np.repeat(np.arange(len(x)), lens)
            dist = np.linalg.norm(self.points[self.vidx[pos]] - c[own], axis=1)
            np.maximum.at(mx, own, dist)
            np.minimum.at(mn, own, dist)
        # facet planes are chords, so allow the sagitta; the real check is the fit
        ok &= (mx - mn <= 0.1 * mx + 2 * self.tol) & (mx > 10 * self.tol) & (mx < 10 * self.diag)
        u, v, w = u[ok], v[ok], w[ok]
        sel = self._first_per_unit(u, 2)
        u, v, w = u[sel], v[sel], w[sel]
        order = np.lexsort((np.arange(len(u)), -self.uarea[u]))
        for i in order.tolist():
            if self.label[u[i]] >= 0 or self.label[v[i]] >= 0 or self.label[w[i]] >= 0:
                continue
            init = [int(u[i]), int(v[i]), int(w[i])]
            # widen the initial support by the 1-ring so the algebraic fit is well posed
            ring = np.unique(self.uadj_idx[_ranges(self.uadj_ptr, np.array(init))[0]])
            ring = ring[self.label[ring] < 0]
            if self.noseed[u[i]]:
                continue
            m = _SphereModel(self)
            self.last_grown = None
            region = self.try_region(m, int(u[i]), np.union1d(init, ring), {"polish": True})
            if region is not None:
                self.accept(m, region)
            elif self.last_grown is not None:
                self.noseed[self.last_grown] = True

    def cones(self):
        u, v, w = self.seed_triples()
        keep = (self.label[u] < 0) & (self.label[v] < 0) & (self.label[w] < 0)
        u, v, w = u[keep], v[keep], w[keep]
        if len(u) == 0:
            return
        ok = self._cone_triples(u, v, w)
        u, v, w = u[ok], v[ok], w[ok]
        sel = self._first_per_unit(u, 2)
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
                self.accept(m, region)

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
            mdl = {"cyl": _CylModel, "cone": _ConeModel, "sphere": _SphereModel}[self.kinds[ri]](self)
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

    def _same(self, A, B):
        ang = np.radians(0.5)
        if isinstance(A, Cylinder):
            if abs(A.radius - B.radius) > self.tol:
                return False
            if abs(A.axis @ B.axis) < np.cos(ang):
                return False
            d = B.origin - A.origin
            return np.linalg.norm(d - (d @ A.axis) * A.axis) <= self.tol
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
        base = len(self.surfaces)
        labels[tris] = base + inv
        for i in range(G):
            self.surfaces.append(Plane(cen[i].copy(), vec[i].copy()))
            self.kinds.append("plane")
        return labels


def segment(points: np.ndarray, faces: np.ndarray, tol: float, angle_deg: float = 0.5,
            *, max_step_deg: float = 46.0, max_tilt_deg: float = 4.0, min_span_deg: float = 20.0,
            cones: bool = True) -> tuple[np.ndarray, list]:
    """points (V,3) float64 welded vertices; faces (F,3) int, consistently wound CCW-outward.
    Returns labels (F,) int and surfaces list. labels[i] = index into surfaces, or -1 for a
    freeform triangle that matches nothing (it will stay its own flat facet).
    Every region (triangles sharing a label) is edge-connected.

    max_step_deg  widest facet (angle around the axis) still treated as part of a tessellated
                  cylinder/cone; 46 accepts octagons and 2-facet 90-degree fillets, rejects
                  hexagonal prisms.
    max_tilt_deg  allowed deviation of a facet normal from the ideal (perpendicular to a
                  cylinder axis, constant angle to a cone axis).
    min_span_deg  minimum angular coverage of a cylinder/cone region."""
    faces = np.asarray(faces, np.int64)
    F = len(faces)
    if F == 0:
        return np.zeros(0, np.int64), []
    s = _Seg(points, faces, tol, angle_deg, max_step_deg, max_tilt_deg, min_span_deg)
    s.build_units()
    s.models = []
    s.cylinders()
    if cones:
        s.cones()
    s.spheres()
    s.merge()
    labels = s.planes()
    return labels, s.surfaces
