"""Polygonal prisms vs coarse cylinders: a lone octagonal prism stays flat-sided by
default; --min-cyl-facets 0 forces it to a true cylinder."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stl2step  # noqa: E402
from corpus_common import write_stl  # noqa: E402


def prism(n, r, h):
    a = np.arange(n) * 2 * np.pi / n
    bot = np.c_[r * np.cos(a), r * np.sin(a), np.zeros(n)]
    top = bot + [0, 0, h]
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [[bot[i], bot[j], top[j]], [bot[i], top[j], top[i]],
                 [[0, 0, 0], bot[j], bot[i]], [[0, 0, h], top[i], top[j]]]
    return np.array(tris)


@pytest.mark.parametrize("forced", [None, 0], ids=["auto", "forced-cylinder"])
def test_lone_octagon(tmp_path, forced):
    src = tmp_path / "oct.stl"
    write_stl(src, prism(8, 10.0, 15.0))
    log = []
    stl2step.convert(src, tmp_path / "oct.step", progress=log.append, min_cyl_facets=forced)
    line = next(m for m in log if "prismatic:" in m)
    if forced is None:
        assert "cylinder" not in line and "10 plane" in line
    else:
        assert "1 cylinder" in line
        vol = next(m for m in log if "solid volume" in m)
        assert f"{math.pi * 100 * 15:.3f}" in vol
