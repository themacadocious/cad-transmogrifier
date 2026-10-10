import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from corpus_common import CORPUS  # noqa: E402

STLS = sorted(CORPUS.glob("*.stl"))


def pytest_configure(config):
    if not STLS:
        raise pytest.UsageError("tests/corpus is empty — run .venv/bin/python tests/make_corpus.py")


@pytest.fixture(scope="session")
def scratch(tmp_path_factory):
    return tmp_path_factory.mktemp("stl2step")
