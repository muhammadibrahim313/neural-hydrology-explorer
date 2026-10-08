"""Smoke test: the Streamlit app must render without raising, with or without model outputs."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "data" / "camels_us" / "camels_attributes_v2.0").exists(),
                    reason="data not downloaded")
def test_app_renders():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert "Neural Hydrology Explorer" in at.title[0].value
    # the basin selector and model radio exist
    assert len(at.selectbox) >= 1
    assert len(at.radio) >= 1
