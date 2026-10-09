"""Smoke test: the Streamlit app must render without raising, with or without model outputs."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "data" / "camels_us" / "camels_attributes_v2.0").exists(),
                    reason="data not downloaded")
def test_app_renders_all_tabs():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert "Neural Hydrology Explorer" in at.title[0].value
    assert len(at.tabs) == 5
    assert len(at.selectbox) >= 1 and len(at.radio) >= 1
    # switching basin and model must also render cleanly
    at.selectbox[0].select("08171300").run()
    assert not at.exception, [e.message for e in at.exception]
    model_radio = [r for r in at.radio if r.label == "Show"][0]
    model_radio.set_value("Transformer").run()
    assert not at.exception, [e.message for e in at.exception]
