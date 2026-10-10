"""Sanity checks on the corpus itself (independent of the converter)."""

import numpy as np
import pytest

from conftest import STLS
from corpus_common import (
    CORPUS, TYPE_CODE, match_cylinders, mesh_volume, open_edges, read_stl, truth_for,
)

ids = [p.stem for p in STLS]


@pytest.mark.parametrize("stl", STLS, ids=ids)
def test_stl_matches_labels_and_truth(stl):
    tris = read_stl(stl)
    lab = np.load(stl.with_suffix(".labels.npz"))
    gt = truth_for(stl)
    assert len(tris) == len(lab["labels"])
    assert open_edges(tris) == 0
    assert abs(mesh_volume(tris) - gt["volume"]) / gt["volume"] < 0.01
    assert len(lab["face_type"]) == gt["faces"]
    assert set(np.unique(lab["labels"])) == set(range(gt["faces"])), "every face got triangles"


@pytest.mark.parametrize("stl", STLS, ids=ids)
def test_labels_lie_on_their_surfaces(stl):
    tris = read_stl(stl)
    lab = np.load(stl.with_suffix(".labels.npz"))
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    for fi, t in enumerate(lab["face_type"]):
        sel = lab["labels"] == fi
        v = tris[sel].reshape(-1, 3)
        if t == TYPE_CODE["plane"]:
            o, nn = lab["plane_origin"][fi], lab["plane_normal"][fi]
            assert np.abs((v - o) @ nn).max() < 1e-3
            assert (n[sel] @ nn > 0).all(), "winding must agree with the outward normal"
        elif t == TYPE_CODE["cylinder"]:
            o, d, r = lab["cyl_origin"][fi], lab["cyl_axis"][fi], lab["cyl_radius"][fi]
            w = v - o
            dist = np.linalg.norm(w - np.outer(w @ d, d), axis=1)
            assert np.abs(dist - r).max() < 1e-3
            assert lab["face_sense"][fi] in (-1, 1)


def test_face_sense_holes_vs_bosses():
    lab = np.load(CORPUS / "box_hole_coarse.labels.npz")
    assert lab["face_sense"][lab["face_type"] == TYPE_CODE["cylinder"]].tolist() == [-1]
    lab = np.load(CORPUS / "boss_plate_coarse.labels.npz")
    assert lab["face_sense"][lab["face_type"] == TYPE_CODE["cylinder"]].tolist() == [1]


def test_scorer_recovers_all_cylinders_from_true_brep(scratch):
    """A STEP of the original B-rep must score 100% — proves the matcher, not the converter."""
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    import make_corpus
    from score import quiet_occt, score_step

    quiet_occt()
    for name in ("plate_holes_cbore", "rounded_block", "flanged_shaft", "two_bodies"):
        shape = make_corpus.clean(make_corpus.PARTS[name][0]())
        out = scratch / f"{name}_truth.step"
        w = STEPControl_Writer()
        w.Transfer(shape, STEPControl_AsIs)
        w.Write(str(out))
        r = score_step(out, truth_for(CORPUS / f"{name}.json"))
        assert r["valid"] and r["solids"] == r["truth_solids"]
        assert abs(r["vol_err_pct"]) < 1e-6
        assert r["cyl_hit"] == r["cyl_truth"] > 0
        assert r["faces_by_type"] == truth_for(CORPUS / f"{name}.json")["faces_by_type"]


def test_matcher_rejects_wrong_cylinders():
    t = [{"origin": [0, 0, 0], "axis": [0, 0, 1], "radius": 5.0}]
    assert match_cylinders(t, [{"origin": [0, 0, 0], "axis": [0, 0, -1], "radius": 5.04}]) == (1, 1)
    assert match_cylinders(t, [{"origin": [0, 0, 0], "axis": [0, 0, 1], "radius": 5.1}]) == (0, 1)
    assert match_cylinders(t, [{"origin": [0, 0, 0], "axis": [0, 0.03, 1], "radius": 5.0}]) == (0, 1)
    assert match_cylinders(t, [{"origin": [1, 0, 0], "axis": [0, 0, 1], "radius": 5.0}]) == (0, 1)
    # two identical holes in different places need two recovered cylinders
    t2 = t + [{"origin": [20, 0, 0], "axis": [0, 0, 1], "radius": 5.0}]
    assert match_cylinders(t2, t) == (1, 2)
