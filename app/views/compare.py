"""Page: compare the estimated rent of two homes."""

from datetime import datetime

import common
import streamlit as st
import ui

HOME_B = {
    **common.DEFAULT_HOME,
    "City": "Bangalore",
    "Area Locality": "Whitefield",
    "Size": 1100,
    "Area Type": "Super Area",
}


def request_compare():
    st.session_state["c_run"] = True


col_a, col_b = st.columns(2, gap="small")
listings = {}
for col, prefix, name, defaults in ((col_a, "ca_", "Home A", common.DEFAULT_HOME), (col_b, "cb_", "Home B", HOME_B)):
    with col, st.container(key=f"card_{prefix}form"):
        st.html(ui.card_head("building", name, "Different city, locality, size or furnishing — anything goes."))
        listings[name] = common.listing_form(prefix, defaults)
        st.button(f"Reset {name}", key=f"{prefix}reset", type="tertiary", on_click=common.reset_form, args=(prefix,))

st.button("Compare rents", type="primary", key="c_go", width="stretch", on_click=request_compare)
if st.session_state.pop("c_run", False) or "c_results" not in st.session_state:
    with st.spinner("Estimating both rents…"):
        st.session_state["c_listings"] = {k: dict(v) for k, v in listings.items()}
        st.session_state["c_results"] = {k: common.estimate(v) for k, v in listings.items()}
        st.session_state["c_time"] = datetime.now().strftime("%I:%M %p").lstrip("0").lower()

saved = st.session_state["c_listings"]
if saved != listings:
    st.html(
        ui.banner(
            "You've changed a home — the results below are for the <b>previous</b> details. "
            "Press <b>Compare rents</b> to update."
        )
    )

outcome, when = st.session_state["c_results"], st.session_state.get("c_time")
col_a, col_b = st.columns(2, gap="small")
for col, name in ((col_a, "Home A"), (col_b, "Home B")):
    result, errors = outcome[name]
    stale = saved.get(name) != listings[name]
    with col, st.container(key=f"card_{name[-1].lower()}_result{'_old' if stale else ''}"):
        st.html(ui.card_title("result", name))
        if errors:
            st.html(
                ui.problems(
                    errors,
                    heading=f"{name} can't be estimated yet:",
                    footer="Your entries are kept — correct them above and press <b>Compare rents</b> again.",
                )
            )
        else:
            st.html(ui.result_card(result, when=when, stale=stale, compact=True))
            if result["warnings"]:
                st.html(ui.notes(result["warnings"]))

(a, a_err), (b, b_err) = outcome["Home A"], outcome["Home B"]
if a and b:
    gap = b["rent"] - a["rent"]
    if abs(gap) < 0.03 * a["rent"]:
        headline = "Both homes are estimated at about the same rent."
    else:
        cheaper, dearer = ("A", "B") if gap > 0 else ("B", "A")
        ratio = max(a["rent"], b["rent"]) / min(a["rent"], b["rent"])
        how = f"about {ratio:.1f} times as much" if ratio >= 1.95 else f"{ratio - 1:.0%} more"
        headline = f"Home {dearer} is about {ui.inr(abs(gap), 100)} a month more than Home {cheaper} — {how}."
    with st.container(key=f"card_diff{'_old' if saved != listings else ''}"):
        st.html(f'<div class="diff-big">{ui.esc(headline)}</div>')
else:
    with st.container(key="card_diff"):
        st.html(ui.empty("The difference appears once both homes can be estimated."))
