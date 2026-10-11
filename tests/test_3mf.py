"""3MF input: build items, component transforms, units, production-extension paths."""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stl2step  # noqa: E402

HERE = Path(__file__).parent
CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
PROD = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"


def mesh_xml(oid, tris):
    pts, idx = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)
    verts = "".join(f'<vertex x="{x!r}" y="{y!r}" z="{z!r}"/>' for x, y, z in pts.tolist())
    faces = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in idx.reshape(-1, 3).tolist())
    return f'<object id="{oid}" type="model"><mesh><vertices>{verts}</vertices><triangles>{faces}</triangles></mesh></object>'


def volume(tris):
    return float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6)


@pytest.fixture
def box():
    return stl2step.read_stl(HERE / "corpus" / "box_hole_coarse.stl")


def write_3mf(path, model, extra=None):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("3D/3dmodel.model", model)
        for name, data in (extra or {}).items():
            z.writestr(name, data)


def test_build_transform_and_units(tmp_path, box):
    # translate by (100, 0, 0) and scale 10x via the cm unit
    model = (f'<model unit="centimeter" xmlns="{CORE}"><resources>{mesh_xml(1, box)}</resources>'
             '<build><item objectid="1" transform="1 0 0 0 1 0 0 0 1 100 0 0"/></build></model>')
    write_3mf(tmp_path / "a.3mf", model)
    tris = stl2step.read_mesh(tmp_path / "a.3mf")
    assert volume(tris) == pytest.approx(volume(box) * 1000, rel=1e-9)
    assert tris[..., 0].min() == pytest.approx((box[..., 0].min() + 100) * 10)


def test_components_and_production_paths(tmp_path, box):
    # object 2 = two instances of a mesh living in another model part (Bambu/Orca layout)
    part = f'<model unit="millimeter" xmlns="{CORE}"><resources>{mesh_xml(7, box)}</resources><build/></model>'
    comp = (f'<object id="2" type="model"><components>'
            f'<component p:path="/3D/Objects/part.model" objectid="7"/>'
            f'<component p:path="/3D/Objects/part.model" objectid="7" transform="1 0 0 0 1 0 0 0 1 0 0 500"/>'
            f'</components></object>')
    model = (f'<model unit="millimeter" xmlns="{CORE}" xmlns:p="{PROD}"><resources>{comp}</resources>'
             '<build><item objectid="2"/></build></model>')
    write_3mf(tmp_path / "b.3mf", model, {"3D/Objects/part.model": part})
    tris = stl2step.read_mesh(tmp_path / "b.3mf")
    assert len(tris) == 2 * len(box)
    assert volume(tris) == pytest.approx(2 * volume(box), rel=1e-9)
    assert tris[..., 2].max() == pytest.approx(box[..., 2].max() + 500)


def test_converts_end_to_end(tmp_path, box):
    model = (f'<model unit="millimeter" xmlns="{CORE}"><resources>{mesh_xml(1, box)}</resources>'
             '<build><item objectid="1"/></build></model>')
    write_3mf(tmp_path / "c.3mf", model)
    log = []
    out = stl2step.convert(tmp_path / "c.3mf", progress=log.append)
    assert out.suffix == ".step" and out.exists()
    assert any("1 cylinder" in m for m in log)
