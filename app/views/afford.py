"""Page: affordability calculator — what share of take-home income a rent would take."""

import altair as alt
import common
import pandas as pd
import streamlit as st
import ui

LIMIT = 30  # % of take-home income: the common rule of thumb
COLORS = {"Rent": ui.T["rust"], "Other housing costs": "#D9A48F", "Left over": ui.T["green"]}

last_result, last_errors = st.session_state.get("a_result", (None, None))
estimate = last_result if last_result and not last_errors else None


def number(label, key, default, **kwargs):
    """A number input whose default lives in session state (so it survives page switches)."""
    st.session_state.setdefault(key, default)
    return st.number_input(label, key=key, **kwargs)


left, right = st.columns(2, gap="small")

# ----------------------------- inputs -----------------------------
with left, st.container(key="card_af_form"):
    st.html(
        ui.card_head(
            "wallet",
            "Your monthly budget",
            "Check any rent against your take-home income — or the one from your latest estimate.",
        )
    )
    if estimate:
        source = common._choice(
            "af_", "source", "Rent to check", ["Latest estimate", "My own figure"], "Latest estimate"
        )
    else:
        source = "My own figure"
    if source == "Latest estimate":
        rent = float(estimate["rent"])
        st.html(
            ui.field_note(
                f"<b>{ui.inr(rent, 100)}</b> a month — {ui.esc(ui.describe(estimate['listing']))}. "
                f"Likely range {ui.inr_short(estimate['low'])} – {ui.inr_short(estimate['high'])}."
            )
        )
    else:
        default_rent = int(round(estimate["rent"], -2)) if estimate else 25_000
        rent = number("Monthly rent (₹)", "af_rent", default_rent, min_value=1_000, max_value=10_000_000, step=1_000)
        if not estimate:
            st.html(ui.field_note("Estimate a home on the <b>Rent estimate</b> page to check its rent here."))
    income = number(
        "Monthly take-home income (₹)",
        "af_income",
        None,
        min_value=0,
        max_value=100_000_000,
        step=5_000,
        placeholder="e.g. 150000",
    )
    other = number("Other monthly housing costs (₹)", "af_other", 0, min_value=0, max_value=10_000_000, step=500)
    st.html(ui.field_note("Maintenance, electricity, water, internet — anything paid monthly to live there."))

limit = LIMIT

# ----------------------------- results -----------------------------
with right:
    if not income:
        with st.container(key="card_af_empty"):
            st.html(ui.card_title("wallet", "Budget check"))
            st.html(
                ui.empty(
                    "Enter your monthly take-home income to see what share of it this rent would take, where the rest "
                    "goes, and how much rent fits your budget."
                )
            )
        st.stop()

    b = common.budget(float(rent), float(income), float(other or 0), limit / 100)

    with st.container(key="card_af_result"):
        st.html(ui.card_title("wallet", "Budget check", f"rent {ui.inr(rent, 100)} a month"))
        st.html(ui.budget_summary(b))
        left_label = "Left each month" if b["left"] >= 0 else "Short each month"
        max_note = ui.inr(b["max_rent"], 100) if b["max_rent"] > 0 else "—"
        st.html(
            ui.tiles(
                [
                    (left_label, ui.inr(abs(b["left"]), 100), ""),
                    (f"Most rent within {limit}%", max_note, ""),
                    (f"Income needed for {limit}%", ui.inr(b["income_needed"], 1000), ""),
                ]
            )
        )
        if b["max_rent"] <= 0:
            st.html(ui.notes([f"Your other housing costs alone use up the {limit}% budget."]))
        st.html(
            f'<p class="muted" style="margin-top:.6rem">Spending at most {limit}% of take-home income on housing is a '
            "common rule of thumb, not a rule — it ignores savings, loan payments and household size.</p>"
        )

# ----------------------------- charts (full width) -----------------------------
c1, c2 = st.columns(2, gap="small")

# chart 1: where the income goes
with c1, st.container(key="card_af_split"):
    st.html(ui.card_title("pie", "Where your income goes"))
    parts = [("Rent", b["rent"]), ("Other housing costs", b["other"]), ("Left over", max(b["left"], 0.0))]
    split = pd.DataFrame([(k, v) for k, v in parts if v > 0], columns=["part", "amount"])
    split["pct"] = split["amount"] / b["income"]
    split["label"] = [f"{k}: {ui.inr(v, 100)} ({p:.0%})" for k, v, p in split[["part", "amount", "pct"]].values]
    donut = (
        alt.Chart(split)
        .mark_arc(innerRadius=62, outerRadius=105, stroke="#FFFFFF", strokeWidth=2)
        .encode(
            theta=alt.Theta("amount:Q", stack=True),
            order=alt.Order("order:Q"),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=split["label"].tolist(), range=[COLORS[p] for p in split["part"]]),
                legend=alt.Legend(title=None, orient="bottom", direction="vertical", labelFontSize=12, labelLimit=320),
            ),
            tooltip=[alt.Tooltip("part:N", title=""), alt.Tooltip("label:N", title="Amount")],
        )
        .transform_calculate(order=f"indexof({split['part'].tolist()}, datum.part)")
    )
    centre = alt.Chart(pd.DataFrame({"t": [f"{b['share']:.0%}"], "s": ["on housing"]}))
    big = centre.mark_text(fontSize=24, fontWeight="bold", dy=-8, color=ui.T["ink"]).encode(text="t:N")
    small = centre.mark_text(fontSize=12, dy=14, color=ui.T["ink2"]).encode(text="s:N")
    st.altair_chart(
        (donut + big + small)
        .properties(height=330)
        .configure_view(stroke=None)
        .configure(background="transparent", font=ui.SANS.replace("'", "")),
        width="stretch",
    )
    if b["left"] < 0:
        st.html(ui.notes([f"Housing costs are {ui.inr(-b['left'], 100)} more than the income entered."]))
    st.html(ui.sr_table("Where your income goes", ["Part", "Amount"], split[["part", "label"]].values))

# chart 2: rents against the budget
with c2, st.container(key="card_af_rents"):
    aside = f"dashed line = most rent within {limit}% ({ui.inr_short(b['max_rent'])})" if b["max_rent"] > 0 else ""
    st.html(ui.card_title("bars", "Rent against your budget", aside))
    if source == "Latest estimate":
        rents = [("Likely low", estimate["low"]), ("Estimate", estimate["rent"]), ("Likely high", estimate["high"])]
    else:
        rents = [("This rent", float(rent))]
    bars = pd.DataFrame(rents, columns=["name", "rent"])
    bars["share"] = (bars["rent"] + b["other"]) / b["income"]
    bars["fits"] = [f"within {limit}%" if s <= b["target"] else f"above {limit}%" for s in bars["share"]]
    bars["text"] = [f"{ui.inr_short(r)} · {s:.0%}" for r, s in bars[["rent", "share"]].values]
    top = max(bars["rent"].max(), b["max_rent"]) * 1.25
    enc_x = alt.X(
        "name:N",
        sort=None,
        title=None,
        scale=alt.Scale(paddingInner=0.45, paddingOuter=0.3),
        axis=alt.Axis(labelAngle=0, labelFontSize=12),
    )
    enc_y = alt.Y("rent:Q", title="Monthly rent (₹)", scale=alt.Scale(domain=[0, top]), axis=alt.Axis(format="~s"))
    col = (
        alt.Chart(bars)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=enc_x,
            y=enc_y,
            color=alt.Color(
                "fits:N",
                scale=alt.Scale(domain=[f"within {limit}%", f"above {limit}%"], range=[ui.T["green"], ui.T["rust"]]),
                legend=alt.Legend(title=None, orient="bottom", labelFontSize=12),
            ),
            tooltip=[
                alt.Tooltip("name:N", title=""),
                alt.Tooltip("text:N", title="Rent · share of income"),
            ],
        )
    )
    labels = alt.Chart(bars).mark_text(dy=-9, fontSize=12, color=ui.T["ink"]).encode(x=enc_x, y=enc_y, text="text:N")
    layers = col + labels
    if b["max_rent"] > 0:
        line = pd.DataFrame({"y": [b["max_rent"]]})
        layers = layers + alt.Chart(line).mark_rule(color=ui.T["ink"], strokeDash=[5, 4]).encode(y="y:Q")
    st.altair_chart(
        layers.properties(height=330)
        .configure_view(stroke=None)
        .configure_axis(gridColor=ui.T["line"], labelColor=ui.T["ink2"], titleColor=ui.T["ink2"])
        .configure(background="transparent", font=ui.SANS.replace("'", "")),
        width="stretch",
    )
    st.html(
        ui.sr_table("Rent against your budget", ["Rent", "Amount and share of income"], bars[["name", "text"]].values)
    )
