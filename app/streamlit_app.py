"""Kiraya — monthly rent estimates for homes in six Indian cities.

    streamlit run app/streamlit_app.py        (from the project folder, after `python run_all.py`)

Uses only the saved final model (models/final_model.joblib) and its input schema (models/input_schema.json);
the dataset itself is not needed. Pages live in app/views/.
"""

import sys
from pathlib import Path

import streamlit as st

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP.parent))  # project root -> `import src`
sys.path.insert(0, str(APP))  # app/ -> `import ui, common`

import common  # noqa: E402
import ui  # noqa: E402

st.set_page_config(page_title="Kiraya — rent estimates", page_icon=str(APP / "assets" / "logo_icon.svg"), layout="wide")
st.html(ui.CSS)
st.logo(str(APP / "assets" / "logo.svg"), size="large")

try:
    common.resources()
except FileNotFoundError as missing:
    st.html(
        ui.problems(
            [str(missing), "Run python run_all.py in the project folder once, then reload this page."],
            heading="The saved model isn't there yet.",
        )
    )
    st.stop()
except Exception as broken:  # e.g. files saved by a much older version of the code
    st.html(
        ui.problems(
            [f"{type(broken).__name__}: {broken}", "Re-run python run_all.py to rebuild the model, then reload."],
            heading="The saved model couldn't be loaded.",
        )
    )
    st.stop()

common.keep_inputs()

pages = [
    st.Page(str(APP / "views" / "estimate.py"), title="Rent estimate", icon=":material/home_work:", default=True),
    st.Page(str(APP / "views" / "compare.py"), title="Compare two homes", icon=":material/compare_arrows:"),
    st.Page(
        str(APP / "views" / "afford.py"),
        title="Affordability calculator",
        icon=":material/account_balance_wallet:",
        url_path="affordability",
    ),
    st.Page(str(APP / "views" / "about.py"), title="About", icon=":material/info:"),
]
st.navigation(pages, position="top").run()
