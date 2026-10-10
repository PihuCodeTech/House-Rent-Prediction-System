"""Page: the study behind the app — what was compared, how accurate the chosen model is, where it goes wrong, and
its limits."""

import altair as alt
import common
import pandas as pd
import streamlit as st
import ui

from src import config
from src.evaluation import read_results

model, meta = common.model(), common.meta()
results = meta.get("results_mean_std_over_seeds", {})
name = meta.get("model", model.name)
profile = getattr(model, "error_profile", None) or {}
coverage = int(round(profile.get("coverage", 0.8) * 100))
seeds = list(getattr(model, "seeds", config.SEEDS))
SETUP_HINT = "Run <code>python run_all.py --only final_selection</code> to add it."


def split(key):
    """(mean, std) floats from a 'mean ± std' results cell, or (None, None)."""
    try:
        mean, spread = (float(x) for x in results[key].split(" ± "))
        return mean, spread
    except (KeyError, ValueError, AttributeError):
        return None, None


def load_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None
    except Exception:  # noqa: BLE001 — a damaged file just hides that section
        return None


try:
    _, NUM = read_results()
except Exception:  # noqa: BLE001
    NUM = None
ERRORS, RESIDUALS = load_csv(config.FINAL_ERRORS_CSV), load_csv(config.FINAL_RESIDUALS_CSV)
overall = ERRORS[ERRORS["Group"] == "Overall"].iloc[0] if ERRORS is not None and len(ERRORS) else None
n_test = int(overall["Test predictions"]) if overall is not None else None

mae, mae_sd = split("Test MAE")
rmse, rmse_sd = split("Test RMSE")
r2, r2_sd = split("Test R²")
mdape, mdape_sd = split("Test Median % Error")
RULE = "Baseline (Locality × Size)"


def num_row(model_name):
    """The numeric results row of one model (or None)."""
    if NUM is None or model_name not in set(NUM["Model"]):
        return None
    return NUM.set_index("Model").loc[model_name]


rule_row = num_row(RULE)
tied = []  # models whose mean test RMSE is within one standard deviation of the lowest
if NUM is not None and "Test RMSE mean" in NUM:
    ranked = NUM.sort_values("Test RMSE mean").set_index("Model")
    first = ranked.iloc[0]
    tied = list(ranked[ranked["Test RMSE mean"] <= first["Test RMSE mean"] + first["Test RMSE std"]].index)

# ----------------------------- 1. the study -----------------------------
with st.container(key="card_about"):
    st.html(ui.card_head("info", "About this study", "What was compared, how, and why this model was chosen."))
    st.html(
        '<p class="lead" style="margin-top:.9rem">How well can a home\'s monthly rent be predicted from its listing '
        "alone — city, locality, size, bedrooms, bathrooms, furnishing and who listed it?</p>"
    )
    per_seed = f"{n_test // len(seeds):,} per seed" if n_test else "15% of listings"
    st.html(
        ui.facts(
            [
                ("Data", "4,731 listings", "6 cities · posted Apr–Jul 2022 · MagicBricks, via Kaggle"),
                ("Test homes", per_seed, f"{n_test:,} predictions over 3 seeds" if n_test else "held out each seed"),
                ("Models", "13 compared", "3 baselines (incl. locality × size), 4 linear, 1 tree, 5 ensembles"),
                ("Tuning", "Grid → Bayesian", "5-fold cross-validation on training rows only"),
                ("Splits", "70 / 15 / 15", f"seeds {', '.join(map(str, seeds))} — every metric is mean ± std"),
            ]
        )
    )
    others = [m for m in tied if m != name]
    if others:
        why = (
            f"On test RMSE it is tied with {ui.esc(' and '.join([', '.join(others[:-1]), others[-1]] if len(others) > 1 else others))} — the differences are smaller than the "
            f"seed-to-seed spread — and among them it has the <b>lowest typical error</b> "
            f"({ui.inr(mae) if mae else '–'} MAE), <b>so it was chosen</b>."
        )
    else:
        why = f"It has the lowest average test error ({ui.inr(rmse) if rmse else '–'} RMSE), <b>so it was chosen</b>."
    beat = ""
    if rule_row is not None and rmse and mdape is not None:
        beat = (
            f" Against the rule of thumb (locality rent per sq ft × size) it cuts RMSE by "
            f"{1 - rmse / rule_row['Test RMSE mean']:.0%} and the median error from "
            f"{rule_row['Test Median % Error mean']:.0f}% to {mdape:.0f}%."
        )
    st.html(
        f'<p class="verdict"><b>{ui.esc(name)}</b> — {why}{beat} The estimates in this app are the average of its '
        "three seed-trained models.</p>"
    )
    st.html(
        ui.tiles(
            [
                (
                    "MAE",
                    ui.inr(mae) if mae else "–",
                    f"Average absolute error per month · ± {ui.inr(mae_sd)} across seeds"
                    if mae
                    else "Average absolute error",
                ),
                (
                    "RMSE",
                    ui.inr(rmse) if rmse else "–",
                    f"Penalises large errors · ± {ui.inr(rmse_sd)} across seeds" if rmse else "Penalises large errors",
                ),
                (
                    "Median % error",
                    f"{mdape:.1f}%" if mdape is not None else "–",
                    f"Typical miss relative to the rent · ± {mdape_sd:.1f} pts"
                    if mdape is not None
                    else "Typical miss",
                ),
                (
                    "R²",
                    f"{r2:.2f}" if r2 is not None else "–",
                    f"Share of rent variation explained · ± {r2_sd:.2f}"
                    if r2 is not None
                    else "Share of rent variation explained",
                ),
            ]
        )
    )
    if mae and rmse:
        worst = ""
        if ERRORS is not None:
            cities = ERRORS[ERRORS["Group"] == "City"].sort_values("RMSE", ascending=False)
            if len(cities) >= 2:
                worst = f", mostly expensive homes in {cities['Value'].iloc[0]} and {cities['Value'].iloc[1]}"
        st.html(
            f'<p class="muted">Measured on test homes the model never saw while training. <b>Why is RMSE '
            f"{rmse / mae:.1f}× the MAE?</b> A typical prediction is off by about {ui.inr(mae, 100)}, but a minority "
            f"of large misses{ui.esc(worst)} weigh heavily in RMSE, which squares each error. The charts below show "
            "how the errors are spread.</p>"
        )

# ----------------------------- 2. leaderboard -----------------------------
with st.container(key="card_leaderboard"):
    st.html(ui.card_title("trophy", "Model leaderboard", "test RMSE · lower is better"))
    if NUM is None:
        st.html(ui.empty("The results table (reports/results.csv) wasn't found. Run <code>python run_all.py</code>."))
    else:
        board = NUM[~NUM["Model"].isin(["Baseline (Mean)", "Baseline (Median)"])].sort_values("Test RMSE mean")
        pct = board["Test Median % Error mean"] if "Test Median % Error mean" in board else [None] * len(board)
        rows = list(zip(board["Model"], board["Test RMSE mean"], board["Test RMSE std"], pct, strict=True))
        st.html(ui.leaderboard(rows, name, rule=RULE))
        notes = [
            "Whiskers show ± 1 standard deviation over the three seeds. Median % error is the typical miss "
            "relative to the home's own rent — unlike RMSE in rupees, it isn't dominated by a few very expensive "
            "homes."
        ]
        if len(tied) > 1:
            notes.append(
                f"{', '.join(tied[:-1])} and {tied[-1]} are within one standard deviation of the lowest RMSE — effectively tied; "
                f"the tie is broken by the lowest MAE."
            )
        if rule_row is not None:
            notes.append(
                f"The striped bar is the rule of thumb a person would use — the locality's median rent per sq ft × "
                f"size ({ui.inr(rule_row['Test RMSE mean'])} RMSE, {rule_row['Test Median % Error mean']:.0f}% median "
                "error). The gap to it is what the machine-learning models really add. Always guessing the average "
                "rent (not shown) is far worse."
            )
        st.html(f'<p class="muted" style="margin-top:.7rem">{ui.esc(" ".join(notes))}</p>')

# ----------------------------- 3. where it goes wrong -----------------------------
with st.container(key="card_errors"):
    st.html(
        ui.card_title(
            "target", "Where the model goes wrong", f"{n_test:,} test predictions" if n_test else "test predictions"
        )
    )
    if ERRORS is None or RESIDUALS is None:
        st.html(ui.empty(f"The error breakdown for the final model hasn't been generated yet. {SETUP_HINT}"))
    else:
        st.html(
            "<p class=\"muted\">Each seed's model scored on that seed's test rows (rows it never trained on), pooled "
            "over the three seeds. Residual = actual rent − predicted rent: positive means the model guessed too "
            "low.</p>"
        )
        c1, c2 = st.columns(2, gap="small")
        with c1:
            hist = RESIDUALS.copy()
            hist["mid"] = (hist["bin_start"] + hist["bin_end"]) / 2
            hist["range"] = [
                (f"≤ {ui.inr(e)}" if o and s < 0 else f"≥ {ui.inr(s)}" if o else f"{ui.inr(s)} to {ui.inr(e)}")
                for s, e, o in zip(hist["bin_start"], hist["bin_end"], hist["open_ended"], strict=True)
            ]
            st.html('<p class="card-title" style="font-size:1rem;margin:.4rem 0 .2rem">Residuals</p>')
            bars = (
                alt.Chart(hist)
                .mark_bar(color=ui.T["rust"], opacity=0.85)
                .encode(
                    x=alt.X(
                        "bin_start:Q",
                        title="Actual − predicted rent (₹ / month)",
                        axis=alt.Axis(format="~s", values=list(range(-50_000, 50_001, 25_000))),
                        scale=alt.Scale(domain=[-50_000, 50_000]),
                    ),
                    x2="bin_end:Q",
                    y=alt.Y("count:Q", title="Test predictions"),
                    y2=alt.datum(0),
                    tooltip=[alt.Tooltip("range:N", title="Residual"), alt.Tooltip("count:Q", title="Predictions")],
                )
            )
            zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=ui.T["ink"], strokeDash=[4, 3]).encode(x="x:Q")
            st.altair_chart(
                (bars + zero)
                .properties(height=380)
                .configure_view(stroke=None)
                .configure_axis(gridColor=ui.T["line"], labelColor=ui.T["ink2"], titleColor=ui.T["ink2"])
                .configure(background="transparent", font=ui.SANS.replace("'", "")),
                width="stretch",
            )
            within = float(overall["Within ±20%"]) if overall is not None else None
            summary = (
                f" {within:.0f}% of predictions were within ±20% of the actual rent; the median miss was "
                f"{float(overall['Median % error']):.0f}% of the rent."
                if within is not None
                else ""
            )
            st.html(
                '<p class="muted">Bars are ₹2,500 wide; the two outer bars also hold every miss beyond ±₹50,000.'
                f"{summary}</p>"
                + ui.sr_table("Residual histogram", ["Residual", "Predictions"], hist[["range", "count"]].values)
            )
        with c2:
            scatter = config.VIZ_DIR / "final_actual_vs_predicted.png"
            if scatter.exists():
                st.image(
                    str(scatter),
                    caption="Actual vs predicted (log scales). Points on the dashed line are perfect predictions.",
                    width="stretch",
                )
            else:
                st.html(ui.empty(f"The actual-vs-predicted plot is missing. {SETUP_HINT}"))

# ----------------------------- 4. range, inputs, limits -----------------------------
c1, c2 = st.columns(2, gap="small")
with c1, st.container(key="card_about_range"):
    st.html(ui.card_title("chart", "How the likely range works"))
    n_pred = profile.get("n_test_predictions")
    st.html(
        f'<p class="muted">The model\'s {f"{n_pred:,} " if n_pred else ""}predictions on test homes it never trained '
        f"on were split into five equal-sized price bands by predicted rent. Within each band, the range covers "
        f"the middle {coverage}% of actual ÷ predicted ratios, so a home predicted at ₹40,000 gets the spread seen "
        "for other homes predicted at that price level. It is calibrated by price level only — not by city or for "
        f"a particular home — and is an empirical range, not a formal {coverage}% prediction interval. It is wide "
        "because similar listings really do rent for different amounts.</p>"
    )
with c2, st.container(key="card_about_knows"):
    st.html(ui.card_title("tag", "What the model knows"))
    st.html(
        '<p class="muted">City, locality, size, bedrooms, bathrooms, furnishing, who listed the home, the preferred '
        "tenants and how the size was measured. Floor and building height were tried and dropped — they changed "
        "the error by only about ₹600. It does not know the building's age, "
        "amenities, exact location or condition — and the what-if bars on the estimate page show what the model "
        "associates with higher rent, not what causes it.</p>"
    )

with st.container(key="card_about_limits"):
    st.html(ui.card_title("pin", "Limits and data"))
    st.html(
        '<p class="muted">About 4,700 listings from one website (MagicBricks), posted April–July 2022, in Mumbai, Delhi, '
        "Bangalore, Chennai, Hyderabad and Kolkata — rents have moved since. Homes outside 20–8,000 sq ft, unusual "
        "layouts and localities with few listings get less reliable estimates, and the app says so when that "
        "happens. Model versions were chosen by test RMSE, which makes the reported test errors slightly "
        "optimistic. This is an estimate, not a valuation.<br><br>Data: House Rent Prediction Dataset by Sourav "
        "Banerjee (Kaggle), collected from MagicBricks.</p>"
    )
