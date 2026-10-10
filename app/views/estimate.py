"""Page: estimate one home's rent."""

from datetime import datetime

import common
import streamlit as st
import ui

if common.is_outdated():
    st.html(
        ui.banner(
            "The saved model files are from an earlier version, so the likely range and the neighbourhood comparison "
            "are limited. Run <code>python run_all.py</code> once to rebuild them, then reload this page."
        )
    )


def run_estimate(listing):
    """Score the listing and remember it, with the time, as the current estimate."""
    with st.spinner("Estimating rent…"):
        st.session_state["a_listing"] = dict(listing)
        st.session_state["a_result"] = common.estimate(listing)
        st.session_state["a_time"] = datetime.now().strftime("%I:%M %p").lstrip("0").lower()


def request_estimate():
    st.session_state["a_run"] = True


left, right = st.columns([5, 7], gap="small")

with left, st.container(key="card_form"):
    st.html(
        ui.card_head(
            "building",
            "Property details",
            "Choose from the values the model was trained on. The locality list follows the city.",
            "<b>*</b> Required",
        )
    )
    listing = common.listing_form("a_", common.DEFAULT_HOME)
    go, reset = st.columns([3, 1])
    go.button("Estimate rent", type="primary", key="a_go", width="stretch", on_click=request_estimate)
    reset.button(
        "Reset",
        key="a_reset",
        width="stretch",
        on_click=common.reset_form,
        args=("a_",),
    )

if st.session_state.pop("a_run", False) or "a_result" not in st.session_state:
    with right:  # the spinner shows where the result will appear
        run_estimate(listing)

result, errors = st.session_state["a_result"]
when = st.session_state.get("a_time")
stale = st.session_state.get("a_listing") != listing
old = "_old" if stale else ""

with right:
    if stale:
        u1, u2 = st.columns([3, 1], vertical_alignment="center")
        u1.html(ui.banner("You've changed the details — the estimate below is for the <b>previous</b> home."))
        u2.button("Update", key="a_update", type="primary", width="stretch", on_click=request_estimate)
    with st.container(key=f"card_result{old}"):
        if errors:
            st.html(
                ui.problems(
                    errors,
                    footer="Your entries are kept — correct them on the left and press <b>Estimate rent</b> again.",
                )
            )
        else:
            st.html(ui.result_card(result, when=when, stale=stale))

    if not errors:
        if result["warnings"]:
            st.html(ui.notes(result["warnings"]))

        with st.container(key=f"card_compare{old}"):
            st.html(ui.card_title("scale", "How it compares", "rent per sq ft"))
            st.html(ui.compare_bars(result))

# what would change the estimate: full width under the form and the result
if not errors:
    with st.container(key=f"card_whatif{old}"):
        st.html(ui.card_title("spark", "What would change the estimate", "one change at a time"))
        try:
            chart, rows = common.what_if_chart(st.session_state["a_listing"], result["rent"])
            if chart is None:
                st.html(ui.empty("No alternative versions of this home could be scored."))
            else:
                st.altair_chart(chart, width="stretch")
                st.html(
                    ui.sr_table(
                        "How the estimate changes",
                        ["If", "Change per month"],
                        rows[["label", "text"]].itertuples(index=False),
                    )
                )
                st.html(
                    '<p class="muted">Each bar re-runs the model with one thing changed. Small bars can point '
                    "the unexpected way — the model learned from real, noisy listings — so read only the large "
                    "ones as clear effects. These show what the model associates with higher rent, not what "
                    "causes it.</p>"
                )
        except Exception as exc:  # a bonus view; never let it break the page
            st.html(ui.empty(f"The what-if comparison isn't available here ({ui.esc(type(exc).__name__)})."))
