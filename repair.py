"""Mesh repair: turn a welded-but-broken STL mesh into a closed, manifold, consistently
wound surface that stl2step can build a solid from.

    points, faces, notes = repair.repair(points, faces, tol)
    info = repair.diagnose(points, faces)

`repair` runs these steps, each only when its defect is present (a mesh that is already a
clean, outward-oriented closed manifold comes back as the *same* arrays):

  1. drop degenerate triangles (repeated vertex / zero area) and duplicate triangles; an
     opposite-wound duplicate pair is an internal wall between two touching bodies and both
     copies go
  2. non-manifold edges (>2 triangles):
     a. "twin" patches — two triangle patches with the same non-manifold boundary that lie on
        top of each other (two bodies sharing a face, triangulated differently): opposite
        wound → remove both (the bodies fuse); same wound → keep one
     b. internal zero-thickness walls left at odd-valence edges (one copy of a shared face)
     c. whatever is left (bodies touching along an edge or a curve): pair the triangles
        around each such edge into material wedges and split the vertices, so every body
        becomes its own closed manifold shell
  3. consistent winding per connected component (orientation propagation)
  4. boundary repair: split edges at T-junctions, snap near-coincident boundary vertices,
     then fill remaining small near-planar holes by ear clipping
  4b. caps — triangles whose apex lies within CAP_TOL_MULT × tol of the interior of their
     longest edge (OCCT rejects such faces) — are removed by flipping that edge with the
     triangle across it; no vertex moves
  5. drop tiny disconnected slivers (≈zero volume, negligible area) — reported, never silent
  6. orient every closed component outward; a shell fully inside another one (an internal
     void) points inward

Everything is numpy/scipy; per-defect Python loops only run over the defective parts.
"""

from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# holes with more than this many edges, or whose vertices stray further than
# HOLE_PLANAR_FRAC × their size from a plane, are left open; a fill larger than
# HOLE_WARN_FRAC of the mesh area is reported as a WARNING
HOLE_MAX_EDGES = 200
HOLE_PLANAR_FRAC = 0.05
HOLE_WARN_FRAC = 0.01
# T-junction / snap distance, in multiples of the weld tolerance (and of the bbox diagonal)
SNAP_TOL_MULT = 50.0
SNAP_DIAG_FRAC = 1e-5
# caps: a triangle whose apex lies within CAP_TOL_MULT × weld tol of the *interior* of its
# longest edge. The weld tolerance is the distance below which two points are treated as
# the same; an apex 10 of those from an edge is, at the precision of the mesh, on it, so the
# triangle is a zero-area fold that OCCT reports as a self-touching wire once ShapeFix grows
# vertex tolerances. Flipping the edge moves no vertex and changes the surface by at most
# that altitude (≤ 10 tol = 1e-6 × diag at the default weld tol: far below fit_tol 2e-5 × diag).
CAP_TOL_MULT = 10.0
CAP_FOLD_DEG = 10.0

# slivers: disconnected components with effective thickness below this × diag and area
# below SLIVER_AREA_FRAC of the total are dropped
SLIVER_THICK_FRAC = 1e-4
SLIVER_AREA_FRAC = 1e-3


# ------------------------------------------------------------------ topology helpers

class _Topo:
    """Edge table of a triangle mesh (recomputed after every structural change)."""

    def __init__(self, faces, nverts):
        F = len(faces)
        self.F = F
        he = np.stack([faces, np.roll(faces, -1, axis=1)], axis=2).reshape(-1, 2)
        self.he = he                                   # (3F, 2) half-edges, he[3f+k]
        lo = np.minimum(he[:, 0], he[:, 1]).astype(np.int64)
        hi = np.maximum(he[:, 0], he[:, 1]).astype(np.int64)
        key = lo * max(nverts, 1) + hi
        ukey, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        self.inv = inv.ravel()                         # (3F,) undirected edge id
        self.cnt = cnt                                 # (E,) triangles per edge
        self.edges = np.stack([ukey // max(nverts, 1), ukey % max(nverts, 1)], axis=1)
        self.fwd = he[:, 0] < he[:, 1]                 # half-edge runs lo -> hi
        nfwd = np.bincount(self.inv, weights=self.fwd, minlength=len(cnt))
        self.inconsistent = (cnt == 2) & (nfwd != 1)   # both triangles run it the same way
        self.hface = np.repeat(np.arange(F), 3)

    def pairs(self, mask_edges):
        """(k, 2) face pairs across the given count-2 edges."""
        sel = mask_edges[self.inv]
        idx = np.flatnonzero(sel)
        order = idx[np.argsort(self.inv[idx], kind="stable")]
        return self.hface[order].reshape(-1, 2), self.inv[order[::2]]

    def edge_faces(self, edge_ids):
        """dict edge id -> list of half-edge indices, for a (small) set of edges."""
        want = np.zeros(len(self.cnt), bool)
        want[edge_ids] = True
        out = defaultdict(list)
        for h in np.flatnonzero(want[self.inv]).tolist():
            out[int(self.inv[h])].append(h)
        return out


def _components(F, pairs):
    if F == 0:
        return 0, np.zeros(0, int)
    a = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(F, F))
    return connected_components(a, directed=False)


def _tri_props(points, faces):
    P = points[faces]
    cr = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]) / 2       # vector area
    vol = np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 1], P[:, 2])) / 6
    return cr, np.linalg.norm(cr, axis=1), vol


def _diag(points):
    return float(np.linalg.norm(np.ptp(points, axis=0))) if len(points) else 0.0


# ------------------------------------------------------------------ diagnose

def _degenerate_mask(points, faces, tol):
    f = faces
    rep = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
    P = points[f]
    area2 = np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    return rep | (area2 <= tol * tol)


def _duplicate_groups(faces):
    """Groups (lists of face indices) of triangles over the same 3 vertices."""
    canon = np.sort(faces, axis=1)
    _, inv, cnt = np.unique(canon, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    if cnt.max(initial=0) < 2:
        return []
    groups = defaultdict(list)
    for f in np.flatnonzero(cnt[inv] > 1).tolist():
        groups[int(inv[f])].append(f)
    return list(groups.values())


def _parity(tri):
    """+1 / -1: orientation of a triangle relative to its sorted vertex order."""
    a, b, c = (int(x) for x in tri)
    s = int(a < b) + int(b < c) + int(c < a)          # cyclic rotations of sorted order have s == 2
    return 1 if s == 2 else -1


def diagnose(points, faces, tol=None):
    """Counts of mesh defects. `self_intersections` is None (not computed)."""
    points = np.asarray(points, float)
    faces = np.asarray(faces)
    if tol is None:
        tol = max(_diag(points) * 1e-7, 1e-6)
    out = dict(triangles=int(len(faces)), vertices=int(len(points)))
    if len(faces) == 0:
        out.update(boundary_edges=0, nonmanifold_edges=0, inconsistent_edges=0, components=0,
                   closed_components=0, degenerate=0, duplicate=0, caps=0, volume=0.0,
                   self_intersections=None, closed_manifold=False)
        return out
    deg = _degenerate_mask(points, faces, tol)
    dups = sum(len(g) - 1 for g in _duplicate_groups(faces))
    t = _Topo(faces, len(points))
    pairs, _ = t.pairs(t.cnt == 2)
    ncomp, lab = _components(len(faces), pairs)
    open_comp = np.zeros(ncomp, bool)
    open_comp[lab[t.hface[t.cnt[t.inv] != 2]]] = True
    _, _, vol = _tri_props(points, faces)
    out.update(
        boundary_edges=int((t.cnt == 1).sum()),
        nonmanifold_edges=int((t.cnt > 2).sum()),
        inconsistent_edges=int(t.inconsistent.sum()),
        components=int(ncomp),
        closed_components=int((~open_comp).sum()),
        degenerate=int(deg.sum()),
        duplicate=int(dups),
        volume=float(vol.sum()),
        caps=int(_cap_mask(points, faces, CAP_TOL_MULT * tol).sum()),
        self_intersections=None,
    )
    out["closed_manifold"] = (out["boundary_edges"] == 0 and out["nonmanifold_edges"] == 0
                              and out["inconsistent_edges"] == 0 and out["degenerate"] == 0
                              and out["duplicate"] == 0)
    return out


# ------------------------------------------------------------------ repair

def repair(points: np.ndarray, faces: np.ndarray, tol: float, nested_as_voids: bool = True,
           cap_tol: float = None):
    """Return (points, faces, notes). `points` (V,3) welded vertices, `faces` (F,3) ints,
    `tol` the weld tolerance. Notes are human-readable lines about what was changed; when
    nothing needed fixing the input arrays are returned unchanged and notes is empty.

    nested_as_voids: a closed shell lying entirely inside another closed shell is an
    internal void and is wound inward (nesting parity). Pass False to only fix inside-out
    shells that are not inside anything (e.g. for multi-body STLs with embedded bodies).

    cap_tol: triangles whose apex is closer than this to the interior of their longest edge
    ("caps") are removed by flipping that edge (default CAP_TOL_MULT × tol; 0 disables)."""
    points_in, faces_in = points, faces
    points = np.asarray(points, float)
    faces = np.asarray(faces)
    if len(faces) == 0:
        return points_in, faces_in, []
    diag = _diag(points)
    tol = float(tol) if tol else max(diag * 1e-7, 1e-6)
    notes = []
    F = faces.astype(np.int64)
    P = points
    changed = False

    # 1. degenerate + duplicate triangles
    deg = _degenerate_mask(P, F, tol)
    if deg.any():
        F = F[~deg]
        notes.append(f"dropped {int(deg.sum())} degenerate triangles")
        changed = True
    groups = _duplicate_groups(F)
    if groups:
        drop, walls, dupl = [], 0, 0
        for g in groups:
            pos = [f for f in g if _parity(F[f]) > 0]
            neg = [f for f in g if _parity(F[f]) < 0]
            k = min(len(pos), len(neg))          # opposite pairs cancel (internal wall)
            drop += pos[:k] + neg[:k]
            walls += k
            rest = pos[k:] + neg[k:]
            drop += rest[1:]                     # same-wound copies: keep one
            dupl += max(len(rest) - 1, 0)
        F = np.delete(F, drop, axis=0)
        if dupl:
            notes.append(f"dropped {dupl} duplicate triangles")
        if walls:
            notes.append(f"removed {walls} back-to-back triangle pairs (internal walls between touching bodies)")
        changed = True

    # 2. non-manifold edges
    t = _Topo(F, len(P))
    if (t.cnt > 2).any():
        n0 = int((t.cnt > 2).sum())
        F, P, msgs = _fix_nonmanifold(P, F, tol)
        notes.append(f"{n0} non-manifold edges: " + "; ".join(msgs))
        changed = True
        t = _Topo(F, len(P))

    # 3. consistent winding
    if t.inconsistent.any():
        F, nflip, nonorient = _orient_consistently(F, t)
        if nflip:
            notes.append(f"flipped {nflip} triangles to make winding consistent")
            changed = True
        if nonorient:
            notes.append(f"WARNING: {nonorient} component(s) are non-orientable (Möbius-like)")
        t = _Topo(F, len(P))

    # 4. holes / cracks
    if (t.cnt == 1).any():
        F, P, msgs = _fix_boundaries(P, F, tol, diag)
        notes += msgs
        changed = True
        t = _Topo(F, len(P))
        # snapping can pinch two sheets together: one more non-manifold / winding pass
        if (t.cnt > 2).any():
            n0 = int((t.cnt > 2).sum())
            F, P, msgs = _fix_nonmanifold(P, F, tol)
            notes.append(f"{n0} non-manifold edges after crack repair: " + "; ".join(msgs))
            t = _Topo(F, len(P))
        if t.inconsistent.any():
            F, nflip, _ = _orient_consistently(F, t)
            notes.append(f"flipped {nflip} triangles to make winding consistent")
            t = _Topo(F, len(P))

    # 4b. cap triangles (apex almost on the opposite edge): OCCT rejects those faces
    cap_tol = CAP_TOL_MULT * tol if cap_tol is None else cap_tol
    if cap_tol > 0 and _cap_mask(P, F, cap_tol).any():
        F, msgs = _remove_caps(P, F, cap_tol, tol)
        if msgs:
            notes += msgs
            changed = True
            t = _Topo(F, len(P))

    # 5. slivers, 6. outward orientation
    F, msgs, ch = _components_pass(P, F, t, diag, nested_as_voids)
    notes += msgs
    changed |= ch

    if not changed:
        return points_in, faces_in, []

    # compact: drop unreferenced vertices
    used = np.unique(F)
    remap = np.full(len(P), -1, np.int64)
    remap[used] = np.arange(len(used))
    P, F = P[used], remap[F]
    d = diagnose(P, F, tol)
    if d["closed_manifold"]:
        notes.append(f"result: closed manifold, {d['components']} shell(s), volume {d['volume']:.6g}")
    else:
        notes.append("result still not closed/manifold: "
                      f"{d['boundary_edges']} boundary, {d['nonmanifold_edges']} non-manifold, "
                      f"{d['inconsistent_edges']} inconsistent edges")
    return P, F.astype(np.asarray(faces_in).dtype, copy=False), notes


# ------------------------------------------------------------------ step 2: non-manifold

def _fix_nonmanifold(P, F, tol):
    msgs = []
    # 2a. twin patches
    F, n_both, n_one, area = _remove_twin_patches(P, F)
    if n_both:
        msgs.append(f"removed {n_both} pairs of coincident back-to-back patches "
                    f"(bodies sharing a face, fused; shared area {area:.4g})")
    if n_one:
        msgs.append(f"removed {n_one} duplicate coincident patches")
    # 2b. internal walls at odd-valence edges
    t = _Topo(F, len(P))
    if (t.cnt > 2).any():
        F, nwall, area = _remove_internal_walls(P, F, t)
        if nwall:
            msgs.append(f"removed {nwall} zero-thickness internal wall/fin patches (area {area:.4g})")
    # 2c. split the rest into wedge-paired manifold sheets
    t = _Topo(F, len(P))
    if (t.cnt > 2).any():
        n = int((t.cnt > 2).sum())
        P, F, nsplit, nbad = _split_nonmanifold(P, F, t)
        msgs.append(f"split {nsplit} vertices so bodies touching along {n} edges become separate shells"
                    + (f" ({nbad} edges had an ambiguous fan, paired by angle)" if nbad else ""))
    return F, P, msgs


def _patches(t):
    """Faces grouped into patches connected through count-2 edges."""
    pairs, _ = t.pairs(t.cnt == 2)
    return _components(t.F, pairs)


def _patch_boundaries(t, lab):
    """For each patch: frozenset of edge ids used exactly once by that patch."""
    pe = lab[t.hface].astype(np.int64) * len(t.cnt) + t.inv
    u, c = np.unique(pe, return_counts=True)
    once = u[c == 1]
    out = defaultdict(set)
    for p, e in zip((once // len(t.cnt)).tolist(), (once % len(t.cnt)).tolist()):
        out[p].add(e)
    return {p: frozenset(s) for p, s in out.items()}


def _remove_twin_patches(P, F):
    t = _Topo(F, len(P))
    nm = t.cnt > 2
    n, lab = _patches(t)
    bnd = _patch_boundaries(t, lab)
    cr, ar, _ = _tri_props(P, F)
    vec = np.zeros((n, 3))
    np.add.at(vec, lab, cr)
    area = np.bincount(lab, weights=ar, minlength=n)
    by_sig = defaultdict(list)
    for p, b in bnd.items():
        if b and all(nm[e] for e in b):
            by_sig[b].append(p)
    drop = np.zeros(n, bool)
    n_both = n_one = 0
    members = None
    for b, ps in by_sig.items():
        if len(ps) < 2:
            continue
        if members is None:
            order = np.argsort(lab, kind="stable")
            starts = np.searchsorted(lab[order], np.arange(n + 1))
            members = lambda q: order[starts[q]:starts[q + 1]]  # noqa: E731
        vsets = {q: frozenset(np.unique(F[members(q)]).tolist()) for q in ps}
        used = set()
        for i, a in enumerate(ps):
            for c in ps[i + 1:]:
                if a in used or c in used:
                    continue
                A = max(area[a], area[c])
                if abs(area[a] - area[c]) > 1e-4 * A:
                    continue
                flat = (np.linalg.norm(vec[a]) > (1 - 1e-6) * area[a]) and (np.linalg.norm(vec[c]) > (1 - 1e-6) * area[c])
                if not (vsets[a] == vsets[c] or flat):
                    continue
                if np.linalg.norm(vec[a] + vec[c]) <= 1e-4 * A:
                    drop[[a, c]] = True
                    used |= {a, c}
                    n_both += 1
                elif np.linalg.norm(vec[a] - vec[c]) <= 1e-4 * A:
                    drop[c] = True
                    used |= {a, c}
                    n_one += 1
    removed = float(area[drop].sum()) / 2
    if drop.any():
        F = F[~drop[lab]]
    return F, n_both, n_one, removed


def _fan(P, t, e, hs):
    """Angular order of the triangles around edge e. Returns list of (theta, s, h) sorted,
    s=+1 if the triangle runs the edge lo->hi. Material is on the -theta side of an s=+1
    triangle and the +theta side of an s=-1 triangle."""
    lo, hi = t.edges[e]
    d = P[hi] - P[lo]
    d /= np.linalg.norm(d)
    e1 = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    out = []
    for h in hs:
        f = h // 3
        a, b = t.he[h]
        c = [v for v in t.he[3 * f:3 * f + 3, 0] if v != a and v != b]
        w = P[c[0]] - P[lo]
        out.append((float(np.arctan2(w @ e2, w @ e1)), 1 if a == lo else -1, h))
    # coincident triangles (same angle): put the one closing a wedge (s=+1) first, i.e. read
    # back-to-back faces as two bodies touching, not as a zero-thickness sheet of material
    out.sort(key=lambda x: (round(x[0] / 1e-9), -x[1], x[2]))
    return out


def _wedge_pairs(fan):
    """Pair triangles of a fan into material wedges. Returns (pairs of h, ok)."""
    k = len(fan)
    s = [x[1] for x in fan]
    if k % 2 == 0 and sum(s) == 0:
        for start in range(k):
            if s[start] == -1 and all(s[(start + i) % k] == (-1 if i % 2 == 0 else 1) for i in range(k)):
                return [(fan[(start + i) % k][2], fan[(start + i + 1) % k][2]) for i in range(0, k, 2)], True
    # ambiguous: pair angular neighbours with opposite orientation greedily
    left, pairs = list(range(k)), []
    while len(left) >= 2:
        best = None
        for i in range(len(left)):
            j = left[(i + 1) % len(left)]
            if s[left[i]] != s[j]:
                gap = (fan[j][0] - fan[left[i]][0]) % (2 * np.pi)
                if best is None or gap < best[0]:
                    best = (gap, i)
        if best is None:
            break
        i = best[1]
        a, b = left[i], left[(i + 1) % len(left)]
        pairs.append((fan[a][2], fan[b][2]))
        left = [x for x in left if x not in (a, b)]
    return pairs, False


def _remove_internal_walls(P, F, t):
    """Patches bounded only by odd-valence non-manifold edges whose removal leaves every one
    of those edges with a properly alternating (material/void) fan: zero-thickness walls."""
    n, lab = _patches(t)
    bnd = _patch_boundaries(t, lab)
    odd = (t.cnt > 2) & (t.cnt % 2 == 1)
    cand = [p for p, b in bnd.items() if b and all(odd[e] for e in b)]
    if not cand:
        return F, 0, 0.0
    efaces = t.edge_faces(np.unique(np.concatenate([list(bnd[p]) for p in cand])))
    _, ar, _ = _tri_props(P, F)
    drop = np.zeros(n, bool)
    for p in cand:
        ok = True
        for e in bnd[p]:
            hs = [h for h in efaces[e] if lab[h // 3] != p and not drop[lab[h // 3]]]
            if len(hs) % 2 or not _wedge_pairs(_fan(P, t, e, hs))[1]:
                ok = False
                break
        if ok:
            drop[p] = True
    if not drop.any():
        return F, 0, 0.0
    m = drop[lab]
    return F[~m], int(drop.sum()), float(ar[m].sum())


def _split_nonmanifold(P, F, t):
    """Pair triangles around every non-manifold edge into wedges, then give each fan of
    triangles around an affected vertex its own copy of that vertex."""
    nm_ids = np.flatnonzero(t.cnt > 2)
    efaces = t.edge_faces(nm_ids)
    # links between faces: count-2 edges + wedge pairs at nm edges, keyed by edge
    link = defaultdict(list)          # vertex -> list of (f, g) links through edges at it
    nbad = 0
    for e in nm_ids.tolist():
        pairs, ok = _wedge_pairs(_fan(P, t, e, efaces[e]))
        nbad += not ok
        lo, hi = t.edges[e]
        for h1, h2 in pairs:
            link[lo].append((h1 // 3, h2 // 3))
            link[hi].append((h1 // 3, h2 // 3))
    verts = set(t.edges[nm_ids].ravel().tolist())
    # faces around those vertices, and their count-2 links
    vmask = np.zeros(len(P), bool)
    vmask[list(verts)] = True
    vf = defaultdict(list)
    for f in np.flatnonzero(vmask[F].any(axis=1)).tolist():
        for v in F[f].tolist():
            if vmask[v]:
                vf[v].append(f)
    pairs2, eid2 = t.pairs(t.cnt == 2)
    sel = vmask[t.edges[eid2]].any(axis=1)
    for (f, g), e in zip(pairs2[sel].tolist(), eid2[sel].tolist()):
        for v in t.edges[e].tolist():
            if vmask[v]:
                link[v].append((f, g))
    F = F.copy()
    newpts = []
    nsplit = 0
    nv = len(P)
    for v in sorted(verts):
        fs = vf[v]
        idx = {f: i for i, f in enumerate(fs)}
        parent = list(range(len(fs)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for f, g in link[v]:
            if f in idx and g in idx:
                parent[find(idx[f])] = find(idx[g])
        roots = {}
        for f in fs:
            roots.setdefault(find(idx[f]), []).append(f)
        groups = sorted(roots.values(), key=len, reverse=True)
        for grp in groups[1:]:
            for f in grp:
                F[f][F[f] == v] = nv
            newpts.append(P[v])
            nv += 1
            nsplit += 1
    if newpts:
        P = np.vstack([P, np.array(newpts)])
    return P, F, nsplit, nbad


# ------------------------------------------------------------------ step 3: winding

def _orient_consistently(F, t):
    """Flip triangles so every count-2 edge is traversed once each way. Uses a doubled
    graph (face, flipped?) so it is fully vectorised."""
    n = len(F)
    pairs, eid = t.pairs(t.cnt == 2)
    same = t.inconsistent[eid]                       # same direction -> need opposite states
    a, b = pairs[:, 0], pairs[:, 1]
    src = np.concatenate([a, a + n])
    dst = np.concatenate([np.where(same, b + n, b), np.where(same, b, b + n)])
    ncomp, lab = connected_components(
        coo_matrix((np.ones(len(src)), (src, dst)), shape=(2 * n, 2 * n)), directed=False)
    _, comp = connected_components(
        coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False)
    nonorient = lab[:n] == lab[n:]
    ncc = comp.max() + 1
    root = np.full(ncc, -1)
    root[comp[::-1]] = np.arange(n)[::-1]             # first face of each component
    anchor = lab[root][comp]                          # doubled-component of (root, unflipped)
    flip = (lab[n:] == anchor) & ~nonorient
    # keep the majority orientation of each component (fewest flips); the outward pass
    # fixes the overall sign later
    nflip_c = np.bincount(comp, weights=flip, minlength=ncc)
    size_c = np.bincount(comp, minlength=ncc)
    invert = (nflip_c > size_c / 2)[comp] & ~nonorient
    flip = flip ^ invert
    F = F.copy()
    F[flip] = F[flip][:, ::-1]
    bad_comps = np.unique(comp[nonorient]).size
    return F, int(flip.sum()), int(bad_comps)


# ------------------------------------------------------------------ step 4: boundaries

def _boundary_loops(F, t):
    """Boundary loops as lists of vertices in the order of the *existing* half-edges
    (a->b of the open triangle), plus the face behind each loop edge."""
    bh = np.flatnonzero(t.cnt[t.inv] == 1)
    out = defaultdict(list)
    for h in bh.tolist():
        out[int(t.he[h, 0])].append(h)
    used = set()
    loops = []
    for h0 in bh.tolist():
        if h0 in used:
            continue
        loop, h = [], h0
        while h not in used:
            used.add(h)
            loop.append(h)
            nxt = [x for x in out[int(t.he[h, 1])] if x not in used]
            if not nxt:
                break
            h = nxt[0]
        loops.append(loop)
    return loops


def _seg_dist(p, a, b):
    ab = b - a
    L2 = ab @ ab
    if L2 == 0:
        return np.linalg.norm(p - a), 0.0
    s = float(np.clip((p - a) @ ab / L2, 0, 1))
    return float(np.linalg.norm(p - (a + s * ab))), s


def _fix_boundaries(P, F, tol, diag):
    msgs = []
    snap = max(SNAP_TOL_MULT * tol, SNAP_DIAG_FRAC * diag)
    n_t = n_snap = 0
    # a) T-junctions: a vertex of a boundary loop lying on another edge of the same loop
    #    (a long edge on one side of a crack, several short collinear ones on the other)
    #    -> split the long edge's triangle into a fan through those vertices
    for _ in range(10):
        t = _Topo(F, len(P))
        splits = {}                                   # face -> (h, [(s, v), ...])
        for loop in _boundary_loops(F, t):
            if len(loop) < 3:
                continue
            hl = np.array(loop)
            A, B = P[t.he[hl, 0]], P[t.he[hl, 1]]
            ab = B - A
            L2 = np.einsum("ij,ij->i", ab, ab)
            L = np.sqrt(L2)
            best = {}
            for v in np.unique(t.he[hl].ravel()).tolist():
                s = np.einsum("ij,ij->i", P[v] - A, ab) / np.where(L2 > 0, L2, 1)
                d = np.linalg.norm(P[v] - (A + np.clip(s, 0, 1)[:, None] * ab), axis=1)
                hit = ((d <= snap) & (L * np.minimum(s, 1 - s) > max(d.min(), tol))
                       & (t.he[hl, 0] != v) & (t.he[hl, 1] != v))
                if hit.any():
                    i = int(np.flatnonzero(hit)[np.argmin(d[hit])])
                    best[v] = (i, float(s[i]))
            for v, (i, sv) in best.items():
                h = int(hl[i])
                f = h // 3
                if f in splits and splits[f][0] != h:
                    continue                          # one edge per triangle per round
                splits.setdefault(f, (h, []))[1].append((sv, v))
        if not splits:
            break
        new, drop = [], []
        for f, (h, vs) in splits.items():
            a, b = t.he[h].tolist()
            c = [x for x in F[f].tolist() if x != a and x != b][0]
            chain = [a] + [v for _, v in sorted(vs)] + [b]
            new += [[chain[k], chain[k + 1], c] for k in range(len(chain) - 1)]
            drop.append(f)
            n_t += len(vs)
        F = np.vstack([np.delete(F, drop, axis=0), np.array(new, dtype=F.dtype)])
    if n_t:
        msgs.append(f"closed {n_t} T-junctions (vertex lying on an opposite edge of a crack)")

    # b) snap near-coincident boundary vertices (within the same loop)
    t = _Topo(F, len(P))
    remap = np.arange(len(P))
    for loop in _boundary_loops(F, t):
        vs = [int(t.he[h, 0]) for h in loop]
        if len(vs) < 3:
            continue
        X = P[vs]
        D = np.linalg.norm(X[:, None] - X[None], axis=2)
        iu = np.argwhere(np.triu(D <= snap, 1))
        for i, j in iu.tolist():
            a, b = remap[vs[i]], remap[vs[j]]
            if a != b:
                remap[remap == max(a, b)] = min(a, b)
                n_snap += 1
    if n_snap:
        F = remap[F]
        bad = (F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 0] == F[:, 2])
        F = F[~bad]
        # snapping may create duplicate / back-to-back triangles
        groups = _duplicate_groups(F)
        drop = []
        for g in groups:
            pos = [f for f in g if _parity(F[f]) > 0]
            neg = [f for f in g if _parity(F[f]) < 0]
            k = min(len(pos), len(neg))
            rest = pos[k:] + neg[k:]
            drop += pos[:k] + neg[:k] + rest[1:]
        F = np.delete(F, drop, axis=0)
        msgs.append(f"snapped {n_snap} near-coincident boundary vertices (gap ≤ {snap:.3g})")

    # c) fill small near-planar holes
    t = _Topo(F, len(P))
    filled, skipped = 0, []
    new = []
    for loop in _boundary_loops(F, t):
        vs = [int(t.he[h, 0]) for h in loop]
        closes = int(t.he[loop[-1], 1]) == vs[0]
        X = P[vs]
        perim = float(np.linalg.norm(np.roll(X, -1, axis=0) - X, axis=1).sum())
        if not closes or len(vs) < 3:
            skipped.append(f"open chain of {len(vs)} edges")
            continue
        if len(vs) > HOLE_MAX_EDGES:
            skipped.append(f"{len(vs)}-edge hole, perimeter {perim:.3g}")
            continue
        tris = _fill_loop(X, perim)
        if tris is None:
            skipped.append(f"non-planar {len(vs)}-edge hole, perimeter {perim:.3g}")
            continue
        # the loop runs along existing half-edges a->b, the fill must run b->a
        vs_arr = np.array(vs)
        new += [vs_arr[[i, k, j]] for i, j, k in tris]
        filled += 1
    if new:
        _, ar_old, _ = _tri_props(P, F)
        new = np.array(new, dtype=F.dtype)
        _, ar_new, _ = _tri_props(P, new)
        F = np.vstack([F, new])
        frac = ar_new.sum() / max(ar_old.sum(), 1e-300)
        msgs.append(("WARNING: " if frac > HOLE_WARN_FRAC else "")
                    + f"filled {filled} small hole{'s' if filled > 1 else ''} with {len(new)} triangles"
                    + f" (area {ar_new.sum():.4g}, {100 * frac:.2g}% of the surface)")
    if skipped:
        msgs.append(f"WARNING: left {len(skipped)} boundary loop(s) open: " + ", ".join(skipped[:5])
                    + (" ..." if len(skipped) > 5 else ""))
    return F, P, msgs


def _fill_loop(X, perim):
    """Ear-clip a closed 3D loop (in the order of the existing half-edges). Returns index
    triples (i, j, k) in that same orientation (caller reverses), or None if not planar."""
    n = len(X)
    if n == 3:
        return [(0, 1, 2)]
    c = X.mean(axis=0)
    # Newell normal (robust for non-convex loops)
    nrm = np.cross(X - c, np.roll(X, -1, axis=0) - c).sum(axis=0)
    ln = np.linalg.norm(nrm)
    if ln < 1e-15:
        _, _, vt = np.linalg.svd(X - c)
        nrm = vt[2]
    else:
        nrm = nrm / ln
    dev = np.abs((X - c) @ nrm).max()
    if dev > HOLE_PLANAR_FRAC * perim / 2 + 1e-12:
        return None
    u = np.cross(nrm, [1.0, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(nrm, u)
    Y = np.stack([(X - c) @ u, (X - c) @ w], axis=1)   # loop is CCW about nrm in (u, w)
    idx = list(range(n))
    out = []

    def cross2(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    guard = 0
    while len(idx) > 3 and guard < 4 * n * n:
        guard += 1
        m = len(idx)
        best = None
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            cr = cross2(Y[i0], Y[i1], Y[i2])
            if cr <= 0:
                continue
            tri = (Y[i0], Y[i1], Y[i2])
            inside = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = Y[j]
                if (cross2(tri[0], tri[1], p) >= 0 and cross2(tri[1], tri[2], p) >= 0
                        and cross2(tri[2], tri[0], p) >= 0):
                    inside = True
                    break
            if inside:
                continue
            # prefer the fattest ear (avoid slivers)
            e = [np.linalg.norm(tri[1] - tri[0]), np.linalg.norm(tri[2] - tri[1]), np.linalg.norm(tri[0] - tri[2])]
            q = cr / (max(e) ** 2 + 1e-300)
            if best is None or q > best[0]:
                best = (q, k)
        if best is None:
            # degenerate / self-overlapping projection: fan from the first vertex
            for k in range(1, len(idx) - 1):
                out.append((idx[0], idx[k], idx[k + 1]))
            return out
        k = best[1]
        out.append((idx[k - 1], idx[k], idx[(k + 1) % m]))
        idx.pop(k)
    out.append(tuple(idx))
    return out


# ------------------------------------------------------------------ step 4b: caps

def _cap_info(P, F):
    """Per triangle: altitude of the apex over the longest edge, index k of that edge
    (vertex k -> k+1), projection parameter s of the apex along it, and the apex's distance
    to the nearer end of the edge."""
    T = P[F]
    E = np.stack([T[:, 1] - T[:, 0], T[:, 2] - T[:, 1], T[:, 0] - T[:, 2]], axis=1)
    L = np.linalg.norm(E, axis=2)
    k = L.argmax(axis=1)
    r = np.arange(len(F))
    a = T[r, k]
    ab = E[r, k]
    c = T[r, (k + 2) % 3]
    Lk = L[r, k]
    h = np.linalg.norm(np.cross(ab, c - a), axis=1) / np.maximum(Lk, 1e-300)
    s = np.einsum("ij,ij->i", c - a, ab) / np.maximum(Lk * Lk, 1e-300)
    return h, k, s, np.minimum(s, 1 - s) * Lk


def _cap_mask(P, F, cap_tol):
    if len(F) == 0:
        return np.zeros(0, bool)
    h, _, s, end = _cap_info(P, F)
    # apex over the interior of the edge and not just next to an end (that is a needle)
    return (h < cap_tol) & (s > 0) & (s < 1) & (end > cap_tol)


def _remove_caps(P, F, cap_tol, tol, rounds=20):
    """Flip the longest edge of every cap triangle with the triangle across it.
    Cap (a,b,c) + neighbour (b,a,d) -> (a,d,c) + (d,b,c). Only flips that keep the mesh
    manifold (edge c-d must not exist), give two non-degenerate triangles that face the
    neighbour's way (no fold) and are not worse caps themselves are applied."""
    n0 = int(_cap_mask(P, F, cap_tol).sum())
    F = F.copy()
    nflip = nbent = 0
    worst_bent = 0.0
    cos_fold = np.cos(np.radians(CAP_FOLD_DEG))
    for _ in range(rounds):
        cap = _cap_mask(P, F, cap_tol)
        if not cap.any():
            break
        h, k, _, _ = _cap_info(P, F)
        t = _Topo(F, len(P))
        nv = max(len(P), 1)
        ukey = t.edges[:, 0] * nv + t.edges[:, 1]
        sel = np.flatnonzero(t.cnt[t.inv] == 2)
        order = sel[np.argsort(t.inv[sel], kind="stable")]
        mate = np.full(len(t.inv), -1)
        mate[order[0::2]] = order[1::2]
        mate[order[1::2]] = order[0::2]
        busy = np.zeros(len(F), bool)
        new_keys = set()
        done = 0
        for f in np.flatnonzero(cap)[np.argsort(h[cap])].tolist():
            hf = 3 * f + int(k[f])
            g = int(mate[hf])
            if busy[f] or g < 0 or busy[g // 3]:
                continue
            nb = g // 3
            a, b = (int(x) for x in t.he[hf])
            c = int(F[f, (k[f] + 2) % 3])
            d = int(F[nb, (g % 3 + 2) % 3])
            if c == d:
                continue
            key = min(c, d) * nv + max(c, d)
            i = np.searchsorted(ukey, key)
            if (i < len(ukey) and ukey[i] == key) or key in new_keys:
                continue
            new = np.array([[a, d, c], [d, b, c]])
            T = P[new]
            nn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
            area2 = np.linalg.norm(nn, axis=1)
            if (area2 <= tol * tol).any():
                continue
            Tn = P[F[nb]]
            n_nb = np.cross(Tn[1] - Tn[0], Tn[2] - Tn[0])
            n_nb /= np.linalg.norm(n_nb)
            if ((nn / area2[:, None]) @ n_nb < cos_fold).any():
                continue
            hn, _, _, _ = _cap_info(P, new)
            if hn.min() <= h[f]:
                continue
            bent = abs(float(n_nb @ (P[c] - P[a])))   # apex off the neighbour's plane
            if bent > tol:
                nbent += 1
                worst_bent = max(worst_bent, bent)
            F[f], F[nb] = new
            busy[[f, nb]] = True
            new_keys.add(key)
            done += 1
        nflip += done
        if not done:
            break
    left = int(_cap_mask(P, F, cap_tol).sum())
    if not nflip:
        # nothing changed: stay silent so the mesh still counts as untouched
        return F, []
    msg = (f"removed cap triangles by edge flips: {n0} -> {left} caps (apex within {cap_tol:.3g} "
           f"of the opposite edge; {nflip} flips")
    if nbent:
        msg += f", {nbent} across a crease, surface moved ≤ {worst_bent:.2g}"
    msg += ")"
    msgs = [msg]
    if left:
        msgs.append(f"{left} cap triangles left (thin strips where no flip raises the altitude)")
    return F, msgs


# ------------------------------------------------------------------ steps 5 + 6

def _components_pass(P, F, t, diag, nested_as_voids=True):
    msgs = []
    changed = False
    pairs, _ = t.pairs(t.cnt == 2)
    n, lab = _components(len(F), pairs)
    open_c = np.zeros(n, bool)
    open_c[lab[t.hface[t.cnt[t.inv] != 2]]] = True
    _, ar, vol = _tri_props(P, F)
    cvol = np.bincount(lab, weights=vol, minlength=n)
    carea = np.bincount(lab, weights=ar, minlength=n)
    total = carea.sum()

    # 5. slivers (only when there is more than one component)
    if n > 1:
        thick = 2 * np.abs(cvol) / np.maximum(carea, 1e-300)
        sliver = (carea < SLIVER_AREA_FRAC * total) & ((thick < SLIVER_THICK_FRAC * diag) | open_c)
        if sliver.all():
            sliver[:] = False
        if sliver.any():
            msgs.append(f"dropped {int(sliver.sum())} tiny disconnected sliver component(s) "
                        f"(total area {carea[sliver].sum():.3g}, volume {np.abs(cvol[sliver]).sum():.3g})")
            keep = ~sliver[lab]
            F, lab = F[keep], lab[keep]
            changed = True
            remap = np.cumsum(~sliver) - 1
            lab = remap[lab]
            cvol, carea, open_c = cvol[~sliver], carea[~sliver], open_c[~sliver]
            n = len(cvol)

    # 6. orientation of closed components
    closed = np.flatnonzero(~open_c)
    if len(closed) == 0:
        return F, msgs, changed
    want = np.ones(n)                       # +1 outward, -1 void
    if len(closed) > 1:
        depth = _nesting_depth(P, F, lab, closed)
        for c, d in zip(closed.tolist(), depth):
            want[c] = -1 if d % 2 else 1
    flip_c = np.zeros(n, bool)
    flip_c[closed] = np.sign(cvol[closed]) != want[closed]
    flip_c &= np.abs(cvol) > 0
    if not nested_as_voids:
        flip_c &= (cvol < 0) & (want > 0)
    if flip_c.any():
        m = flip_c[lab]
        F = F.copy()
        F[m] = F[m][:, ::-1]
        nvoid = int(((want < 0) & flip_c).sum())
        msgs.append(f"reversed {int(flip_c.sum())} inside-out closed shell(s)"
                    + (f" ({nvoid} of them internal voids, now inward-facing)" if nvoid else ""))
        changed = True
    return F, msgs, changed


def _points_in_mesh(Q, T, dirs=((0.5773, 0.5774, 0.5775), (-0.6, 0.31, 0.737), (0.29, -0.81, -0.51))):
    """Ray-parity inside test of points Q (m,3) against triangles T (k,3,3); majority of 3
    ray directions. Returns bool (m,)."""
    votes = np.zeros(len(Q), int)
    e1 = T[:, 1] - T[:, 0]
    e2 = T[:, 2] - T[:, 0]
    for d in dirs:
        d = np.asarray(d, float)
        d /= np.linalg.norm(d)
        h = np.cross(d, e2)
        a = np.einsum("ij,ij->i", e1, h)
        ok = np.abs(a) > 1e-18
        f = np.where(ok, 1.0 / np.where(ok, a, 1), 0)
        for qi, q in enumerate(Q):
            s = q - T[:, 0]
            u = f * np.einsum("ij,ij->i", s, h)
            qv = np.cross(s, e1)
            v = f * (qv @ d)
            tt = f * np.einsum("ij,ij->i", e2, qv)
            hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (tt > 0)
            votes[qi] += int(hit.sum()) % 2
    return votes >= 2


def _nesting_depth(P, F, lab, closed, samples=5):
    """For each closed component: how many other closed components fully contain it.
    Containment is only counted when every sample point (face centroids nudged inward)
    is inside, so overlapping-but-unmerged bodies don't count as nested."""
    lo = np.full((len(closed), 3), np.inf)
    hi = np.full((len(closed), 3), -np.inf)
    idx_of = {c: i for i, c in enumerate(closed.tolist())}
    order = np.argsort(lab, kind="stable")
    starts = np.searchsorted(lab[order], np.arange(lab.max() + 2))
    mem = {c: order[starts[c]:starts[c + 1]] for c in closed.tolist()}
    for c, i in idx_of.items():
        V = P[np.unique(F[mem[c]])]
        lo[i], hi[i] = V.min(0), V.max(0)
    depth = [0] * len(closed)
    rng = np.random.default_rng(0)
    for c, i in idx_of.items():
        inside_bb = [j for j in range(len(closed)) if j != i
                     and np.all(lo[j] <= lo[i]) and np.all(hi[j] >= hi[i])]
        if not inside_bb:
            continue
        fs = mem[c]
        pick = fs if len(fs) <= samples else rng.choice(fs, samples, replace=False)
        T = P[F[pick]]
        cen = T.mean(axis=1)
        nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-300
        # nudge off the surface (either side is fine: we test the other shell, not this one)
        size = np.linalg.norm(hi[i] - lo[i])
        Q = cen + nrm * 1e-6 * size
        for j in inside_bb:
            if _points_in_mesh(Q, P[F[mem[closed[j]]]]).all():
                depth[i] += 1
    return depth
