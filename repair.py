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
  4a. needles — edges shorter than CAP_TOL_MULT × tol — are collapsed (link condition,
     no fold, no degenerate/duplicate triangle), to the midpoint or the end that moves
     the surface least
  4b. caps — triangles whose apex lies within CAP_TOL_MULT × tol of the interior of their
     longest edge (OCCT rejects such faces) — are removed by flipping that edge with the
     triangle across it; no vertex moves
  4c. self-intersections inside one shell (grid broad phase + vectorised separating-axis
     test; pairs sharing a vertex and crossings between different shells are ignored):
     small clusters are cut out with a one-ring and the holes re-filled; clusters over
     SI_MAX_AREA_FRAC of their shell, or whose holes can't be filled cleanly, are left
     and reported as a WARNING
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
    """Counts of mesh defects. `self_intersections`: crossing triangle pairs within one shell."""
    points = np.asarray(points, float)
    faces = np.asarray(faces)
    if tol is None:
        tol = max(_diag(points) * 1e-7, 1e-6)
    out = dict(triangles=int(len(faces)), vertices=int(len(points)))
    if len(faces) == 0:
        out.update(boundary_edges=0, nonmanifold_edges=0, inconsistent_edges=0, components=0,
                   closed_components=0, degenerate=0, duplicate=0, caps=0, needles=0, volume=0.0,
                   self_intersections=0, closed_manifold=False)
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
        needles=int(_needle_edges(points, faces, CAP_TOL_MULT * tol).sum()),
        self_intersections=int(len(_self_intersections(points, faces, tol, lab))),
    )
    out["closed_manifold"] = (out["boundary_edges"] == 0 and out["nonmanifold_edges"] == 0
                              and out["inconsistent_edges"] == 0 and out["degenerate"] == 0
                              and out["duplicate"] == 0)
    return out


# ------------------------------------------------------------------ repair

def repair(points: np.ndarray, faces: np.ndarray, tol: float, nested_as_voids: bool = True,
           cap_tol: float = None, self_intersections: bool = True):
    """Return (points, faces, notes). `points` (V,3) welded vertices, `faces` (F,3) ints,
    `tol` the weld tolerance. Notes are human-readable lines about what was changed; when
    nothing needed fixing the input arrays are returned unchanged and notes is empty.

    nested_as_voids: a closed shell lying entirely inside another closed shell is an
    internal void and is wound inward (nesting parity). Pass False to only fix inside-out
    shells that are not inside anything (e.g. for multi-body STLs with embedded bodies).

    cap_tol: triangles whose apex is closer than this to the interior of their longest edge
    ("caps") are removed by flipping that edge, and edges shorter than it ("needles") are
    collapsed (default CAP_TOL_MULT × tol; 0 disables both).

    self_intersections: cut out and re-fill small clusters of triangles of one shell that
    cross each other (crossings between separate shells are left alone)."""
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

    cap_tol = CAP_TOL_MULT * tol if cap_tol is None else cap_tol
    # 4a. needles (an edge shorter than cap_tol): collapse it
    if cap_tol > 0 and _needle_edges(P, F, cap_tol).any():
        P, F, msgs = _collapse_needles(P, F, cap_tol, tol)
        if msgs:
            notes += msgs
            changed = True
            t = _Topo(F, len(P))

    # 4b. cap triangles (apex almost on the opposite edge): OCCT rejects those faces
    if cap_tol > 0 and _cap_mask(P, F, cap_tol).any():
        F, msgs = _remove_caps(P, F, cap_tol, tol)
        if msgs:
            notes += msgs
            changed = True
            t = _Topo(F, len(P))

    # 4c. self-intersections inside one shell (between shells is fine: fused later)
    if self_intersections:
        F, msgs, ch = _fix_self_intersections(P, F, tol)
        notes += msgs
        if ch:
            changed = True
            t = _Topo(F, len(P))

    # 5. slivers, 6. outward orientation
    F, msgs, ch = _components_pass(P, F, t, diag, nested_as_voids)
    notes += msgs
    changed |= ch

    if not changed:
        return points_in, faces_in, notes       # notes: only warnings about what was left

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


# ------------------------------------------------------------------ step 4a: needles

def _vertex_faces(F, nv):
    """CSR vertex -> incident faces."""
    fl = F.ravel()
    order = np.argsort(fl, kind="stable")
    ptr = np.searchsorted(fl[order], np.arange(nv + 1))
    return order // 3, ptr


def _needle_edges(P, F, cap_tol):
    T = P[F]
    L = np.linalg.norm(T[:, [1, 2, 0]] - T, axis=2)
    return L.min(axis=1) < cap_tol


def _collapse_needles(P, F, cap_tol, tol, rounds=20):
    """Collapse every interior edge shorter than cap_tol to whichever of (its midpoint,
    either end) moves the surface least. A collapse is applied only if it passes the
    link condition (the two end vertices share exactly the two opposite vertices, so the
    mesh stays manifold), leaves no degenerate or duplicate triangle and turns no
    surviving triangle by more than CAP_FOLD_DEG."""
    n0 = int(_needle_edges(P, F, cap_tol).sum())
    P = P.copy()
    F = F.copy()
    alive = np.ones(len(F), bool)
    cos_fold = np.cos(np.radians(CAP_FOLD_DEG))
    ncol, worst = 0, 0.0
    for _ in range(rounds):
        Fa = F[alive]
        idx_alive = np.flatnonzero(alive)
        t = _Topo(Fa, len(P))
        L = np.linalg.norm(P[t.edges[:, 0]] - P[t.edges[:, 1]], axis=1)
        short = np.flatnonzero((L < cap_tol) & (t.cnt == 2))
        if not len(short):
            break
        short = short[np.argsort(L[short])]
        vf_order, vf_ptr = _vertex_faces(Fa, len(P))
        locked = np.zeros(len(P), bool)
        done = 0
        for e in short.tolist():
            u, v = (int(x) for x in t.edges[e])
            if locked[u] or locked[v]:
                continue
            fu = idx_alive[vf_order[vf_ptr[u]:vf_ptr[u + 1]]]
            fv = idx_alive[vf_order[vf_ptr[v]:vf_ptr[v + 1]]]
            both = np.intersect1d(fu, fv)
            if len(both) != 2:
                continue
            ring_u = set(F[fu].ravel().tolist()) - {u}
            ring_v = set(F[fv].ravel().tolist()) - {v}
            opp = set(F[both].ravel().tolist()) - {u, v}
            if (ring_u & ring_v) != opp or len(opp) != 2:
                continue
            keep = np.setdiff1d(np.union1d(fu, fv), both)
            old = P[F[keep]]
            n_old = np.cross(old[:, 1] - old[:, 0], old[:, 2] - old[:, 0])
            a_old = np.linalg.norm(n_old, axis=1)
            h_old, _, _, _ = _cap_info(P, F[keep])
            solid = (a_old > 0) & (h_old >= cap_tol)            # trustworthy normals
            n_old = n_old / np.maximum(a_old, 1e-300)[:, None]
            newf = np.where(F[keep] == v, u, F[keep])
            if len({tuple(sorted(x)) for x in newf.tolist()}) != len(newf):
                continue
            best = None
            for x in (0.5 * (P[u] + P[v]), P[u], P[v]):
                pts = P[newf]
                pts[newf == u] = x
                n_new = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
                a_new = np.linalg.norm(n_new, axis=1)
                if (a_new <= tol * tol).any():
                    continue
                cosang = np.einsum("ij,ij->i", n_new / a_new[:, None], n_old)
                if (cosang[solid] < cos_fold).any():
                    continue
                err = np.abs(np.einsum("ij,ij->i", x - old[:, 0], n_old))[solid]
                err = float(err.max()) if len(err) else 0.0
                if best is None or err < best[0]:
                    best = (err, x)
            if best is None:
                continue
            P[u] = best[1]
            F[keep] = newf
            alive[both] = False
            locked[list(ring_u | ring_v | {u, v})] = True
            worst = max(worst, best[0])
            done += 1
        ncol += done
        if not done:
            break
    if not ncol:
        return P, F, []
    F = F[alive]
    left = int(_needle_edges(P, F, cap_tol).sum())
    msgs = [f"collapsed {ncol} needle edges shorter than {cap_tol:.3g}: {n0} -> {left} needle triangles "
            f"(surface moved ≤ {worst:.2g})"]
    return P, F, msgs


# ------------------------------------------------------------------ step 4c: self-intersections

SI_MAX_AREA_FRAC = 0.01     # a cluster larger than this fraction of its shell is left alone
SI_ITER = 3
SI_DEPTH_MULT = 10.0        # crossings shallower than this × tol (= cap_tol) are contact
SI_CHUNK = 40000


def _ranges(n):
    """For counts n: (owner index, offset within owner) of sum(n) items."""
    n = np.asarray(n, np.int64)
    own = np.repeat(np.arange(len(n)), n)
    off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    return own, off


_OFFS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])


def _tri_cells(T, s, base):
    """(triangle index, cell) for every grid cell of size s a triangle touches. Triangles
    whose bbox spans ≤ 2 cells per axis use their bbox cells; bigger ones are rasterised."""
    c0 = np.floor((T.min(axis=1) - base) / s).astype(np.int64)
    c1 = np.floor((T.max(axis=1) - base) / s).astype(np.int64)
    small = np.all(c1 - c0 <= 1, axis=1)
    si = np.flatnonzero(small)
    out_t, out_c = [], []
    a0, a1 = c0[si], c1[si]
    for o in _OFFS:
        ok = np.all(a0 + o <= a1, axis=1)
        out_t.append(si[ok])
        out_c.append(a0[ok] + o)
    bi = np.flatnonzero(~small)
    if len(bi):
        bt, bc = _big_tri_cells(T[bi], s, base)
        out_t.append(bi[bt])
        out_c.append(bc)
    return np.concatenate(out_t), np.concatenate(out_c)


def _big_tri_cells(T, s, base, chunk=20000):
    """Grid cells (i,j,k) touched by large triangles, without enumerating their whole
    bounding box: walk the cell columns along the triangle's dominant normal axis that
    the projected triangle overlaps (2D separating-axis test) and take the 1-4 cells the
    plane passes through in each column."""
    out_t, out_c = [], []
    for c0 in range(0, len(T), chunk):
        X = (T[c0:c0 + chunk] - base) / s
        n = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
        k = np.abs(n).argmax(axis=1)
        i, j = (k + 1) % 3, (k + 2) % 3
        r = np.arange(len(X))
        U = np.take_along_axis(X, i[:, None, None].repeat(3, 1), axis=2)[:, :, 0]
        V = np.take_along_axis(X, j[:, None, None].repeat(3, 1), axis=2)[:, :, 0]
        W = np.take_along_axis(X, k[:, None, None].repeat(3, 1), axis=2)[:, :, 0]
        u0, u1 = np.floor(U.min(1)).astype(np.int64), np.floor(U.max(1)).astype(np.int64)
        v0, v1 = np.floor(V.min(1)).astype(np.int64), np.floor(V.max(1)).astype(np.int64)
        nu, nv = u1 - u0 + 1, v1 - v0 + 1
        own, off = _ranges(nu * nv)
        cu = u0[own] + off // nv[own]
        cv = v0[own] + off % nv[own]
        # 2D SAT: projected triangle vs the unit square [cu, cu+1] x [cv, cv+1]
        keep = np.ones(len(own), bool)
        for e in range(3):
            a_u, a_v = U[own, e], V[own, e]
            b_u, b_v = U[own, (e + 1) % 3], V[own, (e + 1) % 3]
            c_u, c_v = U[own, (e + 2) % 3], V[own, (e + 2) % 3]
            nu_, nv_ = b_v - a_v, a_u - b_u                     # edge normal
            side = nu_ * (c_u - a_u) + nv_ * (c_v - a_v)        # where the triangle is
            sgn = np.sign(side)
            # most "inside" corner of the square along the normal
            cu_best = cu + (sgn * nu_ > 0)
            cv_best = cv + (sgn * nv_ > 0)
            proj = sgn * (nu_ * (cu_best - a_u) + nv_ * (cv_best - a_v))
            keep &= (proj >= -1e-9) | (sgn == 0)
        own, cu, cv = own[keep], cu[keep], cv[keep]
        # plane height range over the column
        nn = n[own]
        kk = k[own]
        nk = np.take_along_axis(nn, kk[:, None], 1)[:, 0]
        ni = np.take_along_axis(nn, ((kk + 1) % 3)[:, None], 1)[:, 0]
        nj = np.take_along_axis(nn, ((kk + 2) % 3)[:, None], 1)[:, 0]
        d = ni * U[own, 0] + nj * V[own, 0] + nk * W[own, 0]
        ws = [(d - ni * (cu + du) - nj * (cv + dv)) / nk for du in (0, 1) for dv in (0, 1)]
        ws = np.stack(ws, 1)
        w0 = np.floor(np.clip(ws.min(1), W[own].min(1), W[own].max(1))).astype(np.int64)
        w1 = np.floor(np.clip(ws.max(1), W[own].min(1), W[own].max(1))).astype(np.int64)
        own2, off2 = _ranges(w1 - w0 + 1)
        cw = w0[own2] + off2
        cell = np.zeros((len(own2), 3), np.int64)
        rr = np.arange(len(own2))
        o = own[own2]
        cell[rr, i[o]] = cu[own2]
        cell[rr, j[o]] = cv[own2]
        cell[rr, k[o]] = cw
        out_t.append(c0 + o)
        out_c.append(cell)
    return np.concatenate(out_t), np.concatenate(out_c)


def _candidate_pairs(T, F=None, tol=0.0, chunk=2_000_000):
    """Triangle pairs that may touch, from a multi-level uniform grid. Each triangle is
    rasterised (only the cells its plane actually crosses) at the level whose cell size
    matches its own size (≈ sqrt(area), so slivers are not filed by their length), and
    looked up in every coarser level. Pairs are kept only if their bounding boxes overlap
    (by more than tol) and, when F is given, they share no vertex. Returns (a, b), a < b."""
    if len(T) < 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    lo, hi = T.min(axis=1), T.max(axis=1)
    ext = (hi - lo).max(axis=1)
    span = float((hi.max(axis=0) - lo.min(axis=0)).max())
    if span == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    area2 = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    m = np.maximum(np.minimum(ext, np.maximum(2 * np.sqrt(area2), ext / 16)), span / 2 ** 16)
    s0 = float(np.percentile(m, 2))
    lvl = np.maximum(0, np.ceil(np.log2(m / s0))).astype(np.int64)
    base = lo.min(axis=0)
    B = np.int64(1 << 18)
    pa, pb = [], []

    def emit(a, b):
        for i in range(0, len(a), chunk):
            x, y = a[i:i + chunk], b[i:i + chunk]
            ok = x != y
            for k in range(3):
                ok &= (lo[x, k] < hi[y, k] - tol) & (lo[y, k] < hi[x, k] - tol)
            x, y = x[ok], y[ok]
            if F is not None:
                Fx, Fy = F[x], F[y]
                sh = np.zeros(len(x), bool)
                for p in range(3):
                    for q in range(3):
                        sh |= Fx[:, p] == Fy[:, q]
                x, y = x[~sh], y[~sh]
            pa.append(x)
            pb.append(y)
    levels = np.unique(lvl).tolist()
    for lv in levels:
        s = s0 * 2.0 ** lv
        ins = np.flatnonzero(lvl == lv)
        it, ic = _tri_cells(T[ins], s, base)
        it = ins[it]
        ik = (ic[:, 0] * B + ic[:, 1]) * B + ic[:, 2]
        order = np.argsort(ik, kind="stable")
        ik, it = ik[order], it[order]
        start = np.r_[0, np.flatnonzero(np.diff(ik)) + 1]
        size = np.diff(np.r_[start, len(ik)])
        gid = np.repeat(np.arange(len(start)), size)
        rep = size[gid] - (np.arange(len(ik)) - start[gid]) - 1
        own, off = _ranges(rep)
        emit(it[own], it[own + 1 + off])
        fin = np.flatnonzero(lvl < lv)
        if len(fin):
            qt, qc = _tri_cells(T[fin], s, base)
            qt = fin[qt]
            qk = (qc[:, 0] * B + qc[:, 1]) * B + qc[:, 2]
            g = np.searchsorted(ik, qk)
            g = np.minimum(g, len(ik) - 1)
            hit = ik[g] == qk
            qt, qk = qt[hit], qk[hit]
            gi = np.searchsorted(ik[start], qk)              # group index of each query
            own, off = _ranges(size[gi])
            emit(qt[own], it[start[gi][own] + off])
    a, b = np.concatenate(pa), np.concatenate(pb)
    if not len(a):
        return a, b
    lo_, hi_ = np.minimum(a, b), np.maximum(a, b)
    nT = np.int64(len(T))
    pk = np.unique(lo_ * nT + hi_)
    pk = pk[(pk // nT) != (pk % nT)]
    return pk // nT, pk % nT


def _tri_tri_sat(A, B, eps):
    """Vectorised separating-axis test for triangle pairs (n,3,3). True where the
    triangles overlap by more than eps (touching / sharing a point does not count)."""
    eA = A[:, [1, 2, 0]] - A
    eB = B[:, [1, 2, 0]] - B
    nA = np.cross(eA[:, 0], eA[:, 1])
    nB = np.cross(eB[:, 0], eB[:, 1])
    ax = [nA[:, None], nB[:, None],
          np.cross(eA[:, :, None], eB[:, None, :]).reshape(len(A), 9, 3),
          np.cross(nA[:, None], eA), np.cross(nB[:, None], eB)]
    ax = np.concatenate(ax, axis=1)                                  # (n, 17, 3)
    ln = np.linalg.norm(ax, axis=2)
    scale = np.maximum(np.linalg.norm(eA, axis=2).max(1), np.linalg.norm(eB, axis=2).max(1))
    valid = ln > 1e-9 * scale[:, None] ** 2
    ax = ax / np.where(valid, ln, 1)[:, :, None]
    pa = np.einsum("nvk,nak->nav", A, ax)
    pb = np.einsum("nvk,nak->nav", B, ax)
    # touching (overlap ≤ eps) counts as separated, except along a face normal (or an
    # edge-cross axis parallel to one): there only a real gap separates, otherwise
    # coplanar overlapping triangles would always look "touching"
    sep = (pa.max(2) <= pb.min(2) + eps) | (pb.max(2) <= pa.min(2) + eps)
    gap = (pa.max(2) < pb.min(2) - eps) | (pb.max(2) < pa.min(2) - eps)
    nAh = nA / np.maximum(np.linalg.norm(nA, axis=1), 1e-300)[:, None]
    nBh = nB / np.maximum(np.linalg.norm(nB, axis=1), 1e-300)[:, None]
    along_n = (np.abs(np.einsum("nak,nk->na", ax, nAh)) > 1 - 1e-9) | \
              (np.abs(np.einsum("nak,nk->na", ax, nBh)) > 1 - 1e-9)
    sep = np.where(along_n, gap, sep)
    return ~np.any(sep & valid, axis=1)


def _self_intersections(P, F, tol, lab=None):
    """(k, 2) pairs of triangles of the same shell that cross each other. Pairs sharing
    a vertex are skipped; triangles in different shells are ignored."""
    if len(F) < 2:
        return np.zeros((0, 2), np.int64)
    if lab is None:
        t = _Topo(F, len(P))
        pairs, _ = t.pairs(t.cnt == 2)
        _, lab = _components(len(F), pairs)
    T = P[F]
    a, b = _candidate_pairs(T, F, tol)
    keep = lab[a] == lab[b]
    a, b = a[keep], b[keep]
    # plane test: a proper crossing needs each triangle to have vertices clearly on
    # both sides of the other's plane (coplanar pairs go on to the in-plane test).
    # A triangle merely touching the other's plane, or crossing it by less than eps,
    # is contact, not a self-intersection.
    eps = SI_DEPTH_MULT * tol
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
    straddle = np.ones(len(a), bool)
    coplanar = np.ones(len(a), bool)
    for x, y in ((a, b), (b, a)):
        d = np.einsum("pvk,pk->pv", T[y] - T[x][:, None, 0], n[x])
        straddle &= (d.max(axis=1) > eps) & (d.min(axis=1) < -eps)
        coplanar &= np.abs(d).max(axis=1) <= eps
    keep = straddle | coplanar
    a, b = a[keep], b[keep]
    hit = np.zeros(len(a), bool)
    for i in range(0, len(a), SI_CHUNK):
        hit[i:i + SI_CHUNK] = _tri_tri_sat(T[a[i:i + SI_CHUNK]], T[b[i:i + SI_CHUNK]], eps)
    return np.stack([a[hit], b[hit]], axis=1)


def _loops_from_halfedges(he):
    """Chain directed edges (u->v) into closed loops; None if they don't close up."""
    nxt = defaultdict(list)
    for i, (u, v) in enumerate(he.tolist()):
        nxt[u].append(i)
    used = np.zeros(len(he), bool)
    loops = []
    for i0 in range(len(he)):
        if used[i0]:
            continue
        loop, i = [], i0
        while True:
            used[i] = True
            loop.append(int(he[i, 0]))
            cand = [j for j in nxt[int(he[i, 1])] if not used[j]]
            if int(he[i, 1]) == loop[0]:
                break
            if not cand:
                return None
            i = cand[0]
        loops.append(loop)
    return loops


def _fill_is_clean(fill, P, F, t, fs, cnt_in, vf, tol):
    """A hole fill must not reuse an edge that survives elsewhere in the mesh (other than
    the hole's own rim) and must not duplicate a surviving triangle."""
    if _degenerate_mask(P, fill, tol).any():
        return False
    nv = max(len(P), 1)
    ukey = t.edges[:, 0] * nv + t.edges[:, 1]
    he = np.stack([fill, np.roll(fill, -1, axis=1)], axis=2).reshape(-1, 2)
    lo, hi = he.min(axis=1), he.max(axis=1)
    k = lo.astype(np.int64) * nv + hi
    uk, c = np.unique(k, return_counts=True)
    i = np.minimum(np.searchsorted(ukey, uk), len(ukey) - 1)
    exists = ukey[i] == uk
    removed = exists & (cnt_in[i] == t.cnt[i])        # interior to the cut-out cluster
    rim = exists & ~removed                            # surviving edge: the hole's rim
    if (rim & (c != 1)).any() or ((~rim) & (c != 2)).any():
        return False
    order, ptr = vf
    gone = set(fs.tolist())
    for tri in fill.tolist():
        a = tri[0]
        for g in order[ptr[a]:ptr[a + 1]].tolist():
            if g not in gone and sorted(F[g].tolist()) == sorted(tri):
                return False
    return True


def _fix_self_intersections(P, F, tol):
    """MeshLab-style: drop the crossing triangles plus a one-ring around them and close
    the holes with the hole filler. Clusters larger than SI_MAX_AREA_FRAC of their shell,
    or whose holes can't be filled, are left alone and reported."""
    msgs = []
    n_first = None
    removed_area = 0.0
    nclusters = 0
    skipped = []
    for _ in range(SI_ITER):
        t = _Topo(F, len(P))
        pairs, _ = t.pairs(t.cnt == 2)
        _, lab = _components(len(F), pairs)
        hits = _self_intersections(P, F, tol, lab)
        if n_first is None:
            n_first = len(hits)
        if not len(hits):
            break
        bad = np.unique(hits)
        vm = np.zeros(len(P), bool)
        vm[F[bad]] = True
        ring = np.flatnonzero(vm[F].any(axis=1))
        # clusters: ring faces connected through shared edges
        inring = np.zeros(len(F), bool)
        inring[ring] = True
        cp = pairs[inring[pairs[:, 0]] & inring[pairs[:, 1]]]
        _, clab = _components(len(F), cp)
        _, ar, _ = _tri_props(P, F)
        shell_area = np.bincount(lab, weights=ar)
        drop = np.zeros(len(F), bool)
        new = []
        progress = False
        vf = _vertex_faces(F, len(P))
        for c in np.unique(clab[ring]).tolist():
            fs = ring[clab[ring] == c]
            area = ar[fs].sum()
            if area > SI_MAX_AREA_FRAC * shell_area[lab[fs[0]]]:
                skipped.append(f"{len(fs)} triangles, {100 * area / shell_area[lab[fs[0]]]:.2g}% of the shell")
                continue
            # hole boundary as half-edges of the triangles that stay (reverse of ours)
            hh = (3 * fs[:, None] + np.arange(3)).ravel()
            m = np.zeros(len(F), bool)
            m[fs] = True
            mate_in = np.zeros(len(hh), bool)
            e = t.inv[hh]
            # an edge is interior to the cluster if all its triangles are in the cluster
            sel = np.flatnonzero(m[t.hface])
            cnt_in = np.bincount(t.inv[sel], minlength=len(t.cnt))
            mate_in = cnt_in[e] == t.cnt[e]
            he = t.he[hh[~mate_in]][:, ::-1]
            loops = _loops_from_halfedges(he)
            if loops is None:
                skipped.append(f"{len(fs)} triangles, hole not a clean loop")
                continue
            fill = []
            for vs in loops:
                X = P[vs]
                perim = float(np.linalg.norm(np.roll(X, -1, axis=0) - X, axis=1).sum())
                tri = _fill_loop(X, perim) if 3 <= len(vs) <= HOLE_MAX_EDGES else None
                if tri is None:
                    fill = None
                    break
                va = np.array(vs)
                fill += [va[[i, k, j]] for i, j, k in tri]
            if fill is None:
                skipped.append(f"{len(fs)} triangles, hole not fillable")
                continue
            if not _fill_is_clean(np.array(fill), P, F, t, fs, cnt_in, vf, tol):
                skipped.append(f"{len(fs)} triangles, fill would not be manifold")
                continue
            drop[fs] = True
            new += fill
            removed_area += area
            nclusters += 1
            progress = True
        if not progress:
            break
        F = F[~drop]
        if new:
            F = np.vstack([F, np.array(new, dtype=F.dtype)])
    left = len(_self_intersections(P, F, tol)) if nclusters else n_first
    if nclusters:
        msgs.append(f"self-intersections: {n_first} -> {left} crossing triangle pairs "
                    f"(cut out {nclusters} cluster(s), area {removed_area:.3g}, and re-filled the holes)")
    if left:
        msgs.append(f"WARNING: {left} self-intersecting triangle pairs left"
                    + (" (" + "; ".join(sorted(set(skipped))[:3]) + ")" if skipped else ""))
    return F, msgs, bool(nclusters)
