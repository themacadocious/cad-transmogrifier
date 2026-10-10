"""GUI smoke test: real Tk window on a virtual display, converting two corpus files."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent


@pytest.mark.skipif(shutil.which("xvfb-run") is None, reason="needs xvfb-run")
def test_gui_converts(tmp_path):
    inputs = []
    for name in ("box_hole_coarse.stl", "tube_fine.stl"):
        shutil.copy(HERE / "corpus" / name, tmp_path / name)
        inputs.append(str(tmp_path / name))
    out = subprocess.run(
        ["xvfb-run", "-a", sys.executable, str(HERE / "gui_smoke.py"), *inputs],
        capture_output=True, text=True, timeout=180).stdout
    assert "done." in out
    assert "ERROR" not in out
    assert out.count("prismatic:") == 2
    for p in inputs:
        assert Path(p).with_suffix(".step").stat().st_size > 0
