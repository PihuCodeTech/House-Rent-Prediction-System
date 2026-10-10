"""Shared by every page: the loaded model and schema, the listing form, estimates and the what-if chart."""

import altair as alt
import pandas as pd
import streamlit as st
import ui

from src.prediction import InvalidListing, load_final, load_final_meta, load_input_schema, predict_listing, what_if


@st.cache_resource(show_spinner="Loading the model…")
def resources():
    """(model, schema, meta). Raises FileNotFoundError if run_all.py has not been run yet."""
    return load_final(), load_input_schema(), load_final_meta()


def model():
    return resources()[0]


def schema():
    return resources()[1]


def meta():
    return resources()[2]


def typical_error():
    """Mean test MAE in rupees from the final model's metadata, or None."""
    try:
        return float(meta()["results_mean_std_over_seeds"]["Test MAE"].split(" ± ")[0])
    except (KeyError, ValueError, AttributeError, TypeError):
        return None


def is_outdated():
    """True if the saved files predate the likely range / market medians (run_all.py fixes it)."""
    return not getattr(model(), "error_profile", None) or "market" not in schema()


DEFAULT_HOME = {
    "City": "Mumbai",
    "Area Locality": "Andheri West",
    "BHK": 2,
    "Bathroom": 2,
    "Size": 850,
    "Furnishing Status": "Semi-Furnished",
    "Point of Contact": "Contact Owner",
    "Tenant Preferred": "Bachelors/Family",
    "Area Type": "Carpet Area",
}


INPUT_PREFIXES = ("a_", "ca_", "cb_", "af_")  # estimate form, compare forms, affordability calculator


def keep_inputs():
    """Streamlit forgets a widget's value when its page isn't shown. Re-assigning the inputs on every run keeps
    them when you switch pages and come back. Buttons can't be assigned and are skipped. Widgets take their
    defaults from session state (never a `value=` argument), so this never conflicts with them."""
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(INPUT_PREFIXES) and not key.endswith(("_go", "_reset", "_update")):
            try:
                st.session_state[key] = st.session_state[key]
            except Exception:  # noqa: BLE001 — anything that can't be re-assigned simply isn't kept
                pass


FORM_FIELDS = ("city", "locality_", "bhk", "bath", "size", "area", "furn", "contact", "tenant")


def reset_form(prefix):
    """Forget every input of one form (and its remembered choices) so it shows the example home again."""
    for key in list(st.session_state.keys()):
        if key.startswith(prefix) and key[len(prefix) :].startswith(FORM_FIELDS):
            del st.session_state[key]


AREA_LABELS = {"Carpet Area": "Carpet", "Built Area": "Built-up", "Super Area": "Super built-up"}


def _keep_choice(key):
    """Clicking the selected option again would clear it; put the previous choice back instead."""
    if st.session_state.get(key) is None:
        st.session_state[key] = st.session_state.get(f"{key}_last")


def _choice(prefix, name, label, options, default, **kwargs):
    """A segmented control that always has exactly one option selected."""
    key = f"{prefix}{name}"
    default = default if default in options else options[0]
    st.session_state.setdefault(f"{key}_last", default)
    st.session_state.setdefault(key, default)
    value = st.segmented_control(label, options, key=key, on_change=_keep_choice, args=(key,), **kwargs)
    if value is None:  # e.g. state lost between runs
        value = st.session_state[f"{key}_last"]
    st.session_state[f"{key}_last"] = value
    return value


def listing_form(prefix, defaults, details_open=False):
    """Inputs for one home, laid out as in the design. Returns the listing dict for predict_listing."""
    sch = schema()
    choices, numeric, d = sch["choices"], sch["numeric"], defaults
    cities = sorted(choices["City"])

    c1, c2 = st.columns(2)
    with c1:
        st.html(ui.field_label("City", f"{len(cities)} cities", required=True))
        st.session_state.setdefault(f"{prefix}city", d["City"])
        city = st.selectbox("City", cities, key=f"{prefix}city", label_visibility="collapsed")
    localities = sch["localities"].get(city, [])
    by_lower = {name.lower(): name for name in localities}
    default_loc = by_lower.get(str(d.get("Area Locality", "")).lower()) if d["City"] == city else None
    with c2:
        st.html(ui.field_label("Locality", f"{len(localities)} listed"))
        st.session_state.setdefault(f"{prefix}locality_{city}", default_loc)
        locality = st.selectbox(
            "Locality",
            localities,
            index=None,
            key=f"{prefix}locality_{city}",
            placeholder="Search or type a locality",
            accept_new_options=True,
            label_visibility="collapsed",
        )
        if locality and locality.lower() not in by_lower:
            st.html(
                ui.field_note(
                    f"Not among the {len(localities)} {ui.esc(city)} localities in the data — the estimate will use "
                    "a typical locality effect for the city."
                )
            )

    def limits(field):
        r = numeric[field]
        return int(r["min"]), int(r["max"])

    def clamp(value, field):
        lo, hi = limits(field)
        return min(max(int(value), lo), hi)

    c3, c4 = st.columns(2)
    with c3:
        lo, hi = limits("BHK")
        st.html(ui.field_label("BHK", f"{lo}–{hi}", required=True))
        st.session_state.setdefault(f"{prefix}bhk", clamp(d["BHK"], "BHK"))
        bhk = st.number_input("BHK", lo, hi, step=1, key=f"{prefix}bhk", label_visibility="collapsed")
    with c4:
        lo, hi = limits("Bathroom")
        st.html(ui.field_label("Bathrooms", f"{lo}–{hi}", required=True))
        st.session_state.setdefault(f"{prefix}bath", clamp(d["Bathroom"], "Bathroom"))
        bath = st.number_input("Bathrooms", lo, hi, step=1, key=f"{prefix}bath", label_visibility="collapsed")

    area_options = [o for o in AREA_LABELS if o in choices["Area Type"]] + [
        o for o in choices["Area Type"] if o not in AREA_LABELS
    ]
    c5, c6 = st.columns(2)
    with c5:
        lo, hi = limits("Size")
        st.html(ui.field_label("Size (sq ft)", f"{lo:,}–{hi:,}", required=True))
        st.session_state.setdefault(f"{prefix}size", clamp(d["Size"], "Size"))
        size = st.number_input("Size (sq ft)", lo, hi, step=50, key=f"{prefix}size", label_visibility="collapsed")
    with c6:
        st.html(
            ui.field_label(
                "Size measured as",
                info="Carpet: floor area inside the walls. Built-up: carpet plus walls and balconies. Super "
                "built-up: built-up plus a share of lobbies, stairs and other common areas — the largest figure "
                "for the same flat.",
            )
        )
        st.session_state.setdefault(
            f"{prefix}area", d["Area Type"] if d["Area Type"] in area_options else area_options[0]
        )
        area = st.selectbox(
            "Size measured as",
            area_options,
            format_func=lambda o: AREA_LABELS.get(o, o),
            key=f"{prefix}area",
            label_visibility="collapsed",
        )

    with st.expander("Listing details (pre-filled)", expanded=details_open):
        furnishing = _choice(prefix, "furn", "Furnishing", choices["Furnishing Status"], d["Furnishing Status"])
        contact = _choice(
            prefix,
            "contact",
            "Listed by",
            choices["Point of Contact"],
            d["Point of Contact"],
            format_func=lambda c: c.replace("Contact ", ""),
        )
        tenant = _choice(prefix, "tenant", "Tenants preferred", choices["Tenant Preferred"], d["Tenant Preferred"])

    return {
        "City": city,
        "Area Locality": locality or "",
        "BHK": bhk,
        "Bathroom": bath,
        "Size": size,
        "Furnishing Status": furnishing,
        "Point of Contact": contact,
        "Tenant Preferred": tenant,
        "Area Type": area,
    }


def budget(rent, income, other=0.0, target=0.30):
    """Affordability numbers for a monthly rent: housing = rent + other costs, its share of take-home income, what is
    left, the most rent that keeps housing within `target` of income, and the income that rent would need."""
    housing = rent + other
    share = housing / income
    stretch = target + 0.10
    if share <= target:
        level, verdict = "ok", f"Within the common {target:.0%} guideline"
    elif share <= stretch:
        level, verdict = "stretched", f"Stretched — between {target:.0%} and {stretch:.0%} of income"
    else:
        level, verdict = "heavy", f"Heavy — more than {stretch:.0%} of income"
    return {
        "rent": rent,
        "other": other,
        "income": income,
        "target": target,
        "housing": housing,
        "share": share,
        "left": income - housing,
        "max_rent": target * income - other,
        "income_needed": housing / target,
        "level": level,
        "verdict": verdict,
    }


def estimate(listing):
    """(result, None) or (None, list of problems). Never raises."""
    try:
        return predict_listing(listing, model(), schema()), None
    except InvalidListing as bad:
        return None, bad.errors
    except Exception as exc:  # the model itself failed — explain instead of crashing
        return None, [f"The model couldn't score this home ({type(exc).__name__}: {exc})."]


def what_if_chart(listing, rent):
    """(Altair chart, DataFrame) of how the estimate changes when one thing changes; (None, None) if nothing to show.
    Bars are labelled with the signed change, so the colour is never the only cue."""
    rows = what_if(listing, model(), schema())
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    df["direction"] = df["change"].map(lambda c: "▲ rent goes up" if c >= 0 else "▼ rent goes down")
    df["text"] = df["change"].map(ui.signed_inr)
    df["pct"] = df["change"] / rent
    df["align"] = df["change"].map(lambda c: "left" if c >= 0 else "right")
    order = df.sort_values("change", ascending=False)["label"].tolist()
    span = max(df["change"].abs().max(), 1.0)
    base = alt.Chart(df).encode(
        y=alt.Y(
            "label:N",
            sort=order,
            title=None,
            axis=alt.Axis(labelLimit=170, labelFontSize=12, ticks=False, domain=False),
        )
    )
    x = alt.X(
        "change:Q",
        title="Change in monthly rent (₹)",
        scale=alt.Scale(domain=[min(df["change"].min(), 0) - 0.5 * span, max(df["change"].max(), 0) + 0.5 * span]),
        axis=alt.Axis(format="~s", labelFontSize=11),
    )
    bars = base.mark_bar(cornerRadius=4, height=16).encode(
        x=x,
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(domain=["▲ rent goes up", "▼ rent goes down"], range=[ui.T["rust"], ui.T["green"]]),
            legend=alt.Legend(
                title=None, orient="top", direction="horizontal", columns=2, labelFontSize=12, labelLimit=140, offset=6
            ),
        ),
        tooltip=[
            alt.Tooltip("label:N", title="If"),
            alt.Tooltip("text:N", title="Change"),
            alt.Tooltip("pct:Q", title="Change %", format="+.0%"),
        ],
    )
    up = base.transform_filter("datum.change >= 0").mark_text(align="left", dx=5, fontSize=11, color=ui.T["ink"])
    down = base.transform_filter("datum.change < 0").mark_text(align="right", dx=-5, fontSize=11, color=ui.T["ink"])
    labels = up.encode(x=x, text="text:N") + down.encode(x=x, text="text:N")
    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=ui.T["ink2"]).encode(x="x:Q")
    chart = (
        (bars + labels + rule)
        .properties(height=34 * len(df) + 50)
        .configure_view(stroke=None)
        .configure_axis(gridColor=ui.T["line"], labelColor=ui.T["ink2"], titleColor=ui.T["ink2"])
        .configure(background="transparent", font=ui.SANS.replace("'", ""))
    )
    return chart, df.sort_values("change", ascending=False)
