"""The Kiraya app, run headless with Streamlit's AppTest: every page renders, reacts to inputs, explains problems and
never shows a Python exception. Skipped until the final model exists (run `python run_all.py` first)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))  # the app's own modules (common, ui)

from src import config  # noqa: E402

st_testing = pytest.importorskip("streamlit.testing.v1")
pytestmark = pytest.mark.skipif(
    not (config.MODELS_DIR / "final_model.joblib").exists() or not (config.MODELS_DIR / "input_schema.json").exists(),
    reason="final model not built yet (run python run_all.py)",
)
APP = str(ROOT / "app" / "streamlit_app.py")


def run_app(page=None):
    import streamlit as st

    st.cache_resource.clear()
    at = st_testing.AppTest.from_file(APP, default_timeout=60)
    at.run()
    if page:
        at.switch_page(page).run()
    return at


def page_html(at):
    return " ".join(str(getattr(el, "body", "")) for el in at.main)


def no_exceptions(at):
    assert not at.exception, [e.message for e in at.exception]


def estimate(at):
    at.button(key="a_go").click().run()
    no_exceptions(at)
    return page_html(at)


def test_estimate_page_shows_a_result_on_first_load():
    at = run_app()
    no_exceptions(at)
    html = page_html(at)
    assert "/ month" in html and "Likely range" in html and "Model agreement" in html and "How it compares" in html
    assert "Up to date" in html and "How to read this estimate" not in html


def test_changing_city_resets_locality_and_updates_after_pressing_estimate():
    at = run_app()
    at.selectbox(key="a_city").set_value("Delhi").run()
    no_exceptions(at)
    assert at.selectbox(key="a_locality_Delhi").value is None
    html = page_html(at)
    assert "for the <b>previous</b> home" in html and "Outdated" in html  # the old result is marked, not replaced
    html = estimate(at)
    assert "Delhi" in html and "Outdated" not in html and "Up to date" in html


def test_update_button_next_to_an_outdated_estimate():
    at = run_app()
    at.number_input(key="a_bhk").set_value(3).run()
    assert "Outdated" in page_html(at)
    at.button(key="a_update").click().run()
    no_exceptions(at)
    html = page_html(at)
    assert "Outdated" not in html and "3 BHK" in html


def test_reset_restores_the_example_home():
    at = run_app()
    at.number_input(key="a_bhk").set_value(4).run()
    at.selectbox(key="a_city").set_value("Kolkata").run()
    at.button(key="a_reset").click().run()
    no_exceptions(at)
    assert at.number_input(key="a_bhk").value == 2 and at.selectbox(key="a_city").value == "Mumbai"
    assert at.selectbox(key="a_locality_Mumbai").value == "Andheri West"


def test_failed_estimate_keeps_the_entries():
    at = run_app()
    at.selectbox(key="a_locality_Mumbai").set_value(None).run()
    at.number_input(key="a_size").set_value(1200).run()
    estimate(at)
    assert at.number_input(key="a_size").value == 1200


def test_affordability_calculator_uses_the_latest_estimate():
    at = run_app()  # the estimate page scores the example home first
    assert "Can you afford it" not in page_html(at)  # moved to its own page
    at.switch_page("views/afford.py").run()
    no_exceptions(at)
    assert "Enter your monthly take-home income" in page_html(at)
    assert at.segmented_control(key="af_source").value == "Latest estimate"
    at.number_input(key="af_income").set_value(150_000).run()
    no_exceptions(at)
    html = page_html(at)
    assert "of take-home income goes to housing" in html and "Where your income goes" in html
    assert "Rent against your budget" in html and "Likely high" in html


def test_affordability_calculator_with_my_own_rent():
    at = run_app("views/afford.py")
    at.segmented_control(key="af_source").set_value("My own figure").run()
    at.number_input(key="af_rent").set_value(60_000).run()
    at.number_input(key="af_income").set_value(100_000).run()
    no_exceptions(at)
    assert "Heavy" in page_html(at) and "60%" in page_html(at)
    at.number_input(key="af_rent").set_value(35_000).run()
    assert "Stretched" in page_html(at)
    at.number_input(key="af_other").set_value(70_000).run()  # costs above the whole income
    no_exceptions(at)
    html = page_html(at)
    assert "Short each month" in html and "other housing costs alone use up" in html


def test_size_measure_explains_itself_and_floors_are_gone():
    at = run_app()
    html = page_html(at)
    assert "Carpet: floor area inside the walls" in html and at.selectbox(key="a_area").value == "Carpet Area"
    assert "Total floors" not in html and not [w for w in at.selectbox if "floor" in str(w.key)]


def test_inputs_survive_switching_pages():
    at = run_app()
    at.number_input(key="a_bhk").set_value(3).run()
    at.button(key="a_go").click().run()
    at.switch_page("views/afford.py").run()
    at.number_input(key="af_income").set_value(90_000).run()
    at.switch_page("views/about.py").run()
    at.switch_page("views/estimate.py").run()
    assert at.number_input(key="a_bhk").value == 3 and "Outdated" not in page_html(at)
    at.switch_page("views/afford.py").run()
    assert at.number_input(key="af_income").value == 90_000
    no_exceptions(at)


def test_number_inputs_are_limited_to_the_data_range():
    """The +/- buttons stop at the smallest and largest values seen in the data."""
    import json

    numeric = json.loads((config.MODELS_DIR / "input_schema.json").read_text())["numeric"]
    at = run_app()
    for key, field in (("a_bhk", "BHK"), ("a_bath", "Bathroom"), ("a_size", "Size")):
        widget = at.number_input(key=key)
        assert (widget.proto.min, widget.proto.max) == (int(numeric[field]["min"]), int(numeric[field]["max"])), key
        assert widget.proto.has_min and widget.proto.has_max and widget.proto.min >= 1


def test_listing_details_cannot_be_cleared():
    at = run_app()
    at.segmented_control(key="a_furn").set_value(None).run()
    no_exceptions(at)
    assert at.segmented_control(key="a_furn").value == "Semi-Furnished"
    at.segmented_control(key="a_furn").set_value("Furnished").run()
    at.segmented_control(key="a_furn").set_value(None).run()
    assert at.segmented_control(key="a_furn").value == "Furnished"
    assert "not given" not in estimate(at)


def test_compare_page_shows_both_homes_and_the_difference():
    at = run_app("views/compare.py")
    no_exceptions(at)
    html = page_html(at)
    assert "Home A" in html and "Home B" in html and "a month more than Home" in html


def test_about_page_renders():
    at = run_app("views/about.py")
    no_exceptions(at)
    html = page_html(at)
    assert "About this study" in html and "so it was chosen" in html and "How the likely range works" in html
    assert "MAE" in html and "Model leaderboard" in html and "Chosen" in html
    assert "Rule of thumb" in html and "Median % error" in html
    if config.FINAL_ERRORS_CSV.exists() and config.FINAL_RESIDUALS_CSV.exists():
        assert "Residuals" in html and "By city" not in html
    else:
        assert "hasn't been generated yet" in html


def test_missing_model_shows_setup_message(monkeypatch):
    import common

    def missing():
        raise FileNotFoundError("models/final_model.joblib not found — run notebooks/final_selection.ipynb first.")

    monkeypatch.setattr(common, "load_final", missing)
    at = run_app()
    no_exceptions(at)
    assert "saved model isn" in page_html(at) and "run_all.py" in page_html(at)


def test_old_model_files_still_work(monkeypatch):
    """Models saved before the likely range existed load without error and show a rebuild banner."""
    import common

    real = common.load_final

    def old():
        model = real()
        model.__dict__.pop("error_profile", None)  # as unpickled from an older version
        return model

    monkeypatch.setattr(common, "load_final", old)
    at = run_app()
    no_exceptions(at)
    assert "earlier version" in page_html(at)
    at.switch_page("views/about.py").run()
    no_exceptions(at)


def test_budget_numbers():
    import common

    b = common.budget(30_000, 100_000, other=5_000, target=0.30)
    assert b["housing"] == 35_000 and b["share"] == 0.35 and b["left"] == 65_000
    assert b["max_rent"] == 25_000 and round(b["income_needed"]) == 116_667 and b["level"] == "stretched"
    assert common.budget(20_000, 100_000)["level"] == "ok"
    assert common.budget(50_000, 100_000)["level"] == "heavy"


def test_screen_reader_tables_are_clipped_by_a_wrapper():
    """A table ignores height/overflow, so clipping it directly left thousands of pixels of empty scroll."""
    import ui

    html = ui.sr_table("x", ["a", "b"], [["1", "2"]] * 30)
    assert html.startswith('<div class="sr-only"><table>') and html.endswith("</table></div>")
