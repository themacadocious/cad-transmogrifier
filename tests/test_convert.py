"""Run the current converter (stl2step.convert) over the whole corpus and check the result.

Hard requirements: valid solid(s), right number of bodies, <0.5% volume error.
Prismatic rebuild must also recover every truth cylinder and land near the CAD face
count (large sphere patches come back as a few pieces, hence "near").

Set STL2STEP_IMPL=module:function or path/to/file.py:function to test another converter.
"""

import os
import time

import pytest

from conftest import STLS
from corpus_common import truth_for
from score import load_impl, quiet_occt, score_step

_cache = {}


@pytest.fixture(scope="session")
def convert():
    fn = load_impl(os.environ.get("STL2STEP_IMPL", "stl2step:convert"))
    quiet_occt()
    return fn


@pytest.fixture
def result(request, convert, scratch):
    stl = request.param
    if stl not in _cache:
        out = scratch / (stl.stem + ".step")
        t0 = time.perf_counter()
        convert(str(stl), str(out), lambda m: None)
        _cache[stl] = score_step(out, truth_for(stl), time.perf_counter() - t0, src=stl)
    return _cache[stl]


ids = [p.stem for p in STLS]
files = pytest.mark.parametrize("result", STLS, ids=ids, indirect=True)


@files
def test_valid_solid(result):
    assert result["valid"], "BRepCheck_Analyzer rejects the output"
    assert result["solids"] == result["truth_solids"]


@files
def test_volume(result):
    assert abs(result["vol_err_pct"]) < 0.5, f"volume off by {result['vol_err_pct']:+.3f}% vs true B-rep"
    assert abs(result["mesh_vol_err_pct"]) < 0.5, f"volume off by {result['mesh_vol_err_pct']:+.3f}% vs input mesh"


with_cyls = [p for p in STLS if truth_for(p)["cylinders"]]


@pytest.mark.parametrize("result", with_cyls, ids=[p.stem for p in with_cyls], indirect=True)
def test_cylinders_recovered(result):
    assert result["cyl_hit"] == result["cyl_truth"]


@pytest.mark.parametrize("result", with_cyls, ids=[p.stem for p in with_cyls], indirect=True)
def test_face_count_near_truth(result):
    # analytic rebuild should give roughly the CAD face count, not one face per facet strip
    assert result["faces"] <= 1.5 * result["truth_faces"]
