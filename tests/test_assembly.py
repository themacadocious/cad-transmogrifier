"""Multi-shell assembly: internal voids, overlapping bodies, bodies that only touch.

Meshes are generated from OCC shapes on the fly (tests/make_corpus.py helpers) and run
through stl2step.convert on both the prismatic and the faceted path."""
import math
import sys
from pathlib import Path

import pytest
from OCP.BRep import BRep_Builder
from OCP.STEPControl import STEPControl_Reader
from OCP.TopoDS import TopoDS_Compound

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import make_corpus as mc  # noqa: E402
import stl2step  # noqa: E402
from corpus_common import write_stl  # noqa: E402


def compound(*shapes):
    c, b = TopoDS_Compound(), BRep_Builder()
    b.MakeCompound(c)
    for s in shapes:
        b.Add(c, s)
    return c


CASES = {
    # name: (shape, expected volume, expected solids)
    "hollow": (lambda: mc.cut(mc.box(0, 0, 0, 30, 30, 30), mc.cyl(15, 15, 10, 5, 10)),
               27000 - math.pi * 25 * 10, 1),
    "overlap": (lambda: compound(mc.box(0, 0, 0, 30, 30, 5), mc.box(10, 10, 3, 10, 10, 5)), 4800, 1),
    "touching": (lambda: compound(mc.box(0, 0, 0, 30, 30, 5), mc.box(30, 30, 0, 10, 10, 5)), 5000, 2),
}


@pytest.mark.parametrize("prismatic", [True, False], ids=["prismatic", "faceted"])
@pytest.mark.parametrize("name", list(CASES))
def test_assembly(tmp_path, name, prismatic):
    make, want_vol, want_solids = CASES[name]
    tris = mc.mesh_with_labels(make(), 0.01, 0.1)[0]
    src = tmp_path / f"{name}.stl"
    write_stl(src, tris)
    out = stl2step.convert(src, tmp_path / f"{name}.step", prismatic=prismatic, progress=lambda m: None)

    r = STEPControl_Reader()
    r.ReadFile(str(out))
    r.TransferRoots()
    shape = r.OneShape()
    assert stl2step.BRepCheck_Analyzer(shape).IsValid()
    assert sum(1 for _ in stl2step._iter_solids(shape)) == want_solids
    # faceted holes lose a little volume to the chords
    assert stl2step.volume(shape) == pytest.approx(want_vol, rel=2e-3)
