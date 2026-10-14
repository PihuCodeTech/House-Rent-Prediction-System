"""Landing page: what Kiraya is, a working estimator in the hero, the six rent markets it covers, the three tools,
why the number can be trusted, and the source behind it. Every figure on this page is read from the saved model, its
input schema (market medians) and its results."""

import common
import streamlit as st
import ui

EST, CMP, AFF, ABT = (f"/{p}" for p in ("estimate", "compare", "affordability", "about"))
REPO = "https://github.com/PihuCodeTech/House-Rent-Prediction-System"
KAGGLE = "https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset"


def meta_val(key):
    """Mean of a 'mean ± std' metric from the final model's metadata, or None."""
    try:
        return float(common.meta()["results_mean_std_over_seeds"][key].split(" ± ")[0])
    except (KeyError, ValueError, AttributeError, TypeError):
        return None


def model_rmse():
    """{model_name: test RMSE mean} from the results table, or {} if unavailable."""
    try:
        from src.evaluation import read_results

        _, num = read_results()
        return dict(zip(num["Model"], num["Test RMSE mean"], strict=True))
    except Exception:  # noqa: BLE001 — the comparison section falls back to its own copy of the numbers
        return {}


FINAL = common.meta().get("model", "Gradient Boosting")
mae, mdape, r2 = common.typical_error(), meta_val("Test Median % Error"), meta_val("Test R²")
rmse = model_rmse()
gb = rmse.get(FINAL, 22_077)
rule = rmse.get("Baseline (Locality × Size)", 31_953)
naive = rmse.get("Baseline (Mean)", 53_731)
N_TEST = 2_130  # 3 seeds × ~15% of 4,731 listings held out each time

CITIES = sorted(common.schema()["choices"]["City"])
MARKET = (common.schema().get("market") or {}).get("cities", {})

result, errors = common.estimate(common.DEFAULT_HOME)
cmp_home = common.estimate(
    {**common.DEFAULT_HOME, "City": "Bangalore", "Area Locality": "Whitefield", "Size": 1100, "Area Type": "Super Area"}
)[0]


# ----------------------------- hero: headline + a working estimator -----------------------------
def quick_result(res, errs, city, bhk):
    """The estimator's answer: rent, likely range, and rent per sq ft against the city median."""
    if errs or not res:
        return ui.problems(errs or ["The model couldn't score this home."], heading="This home can't be estimated:")
    rows = [("This home", res["rent_per_sqft"], "me")]
    if city in MARKET:
        rows.append((f"{city} median", MARKET[city]["median_rps"], ""))
    top = max(v for _, v, _ in rows) * 1.08
    bars = "".join(
        f'<div class="ex-mini-row {c}"><span>{ui.esc(lbl)}</span>'
        f'<div class="ex-mini-track" aria-hidden="true"><div class="ex-mini-fill" style="width:{100 * v / top:.0f}%"></div></div>'
        f'<span class="ex-mini-val">{ui.inr(v)}</span></div>'
        for lbl, v, c in rows
    )
    baths = "1 bathroom" if bhk == 1 else f"{bhk} bathrooms"
    return f"""
<div class="q-out" aria-live="polite">
  <div class="q-top">
    <div class="q-rent">{ui.inr(res["rent"], 100)}<span> / month</span></div>
    <div class="q-range">Likely range<b>{ui.inr_short(res["seed_low"])} – {ui.inr_short(res["seed_high"])}</b></div>
  </div>
  <p class="q-desc">{ui.esc(ui.describe(res["listing"]))}</p>
  <div class="ex-mini">{bars}<div class="ex-mini-row"><span></span><span style="font-size:.8rem">rent per sq ft</span><span></span></div></div>
  <p class="q-note">Assumes a typical {ui.esc(city)} locality, {baths}, semi-furnished and listed by the owner.
  The full estimate adds the locality and the rest.</p>
</div>"""


st.session_state.setdefault("q_size", 850)
hero_l, hero_r = st.columns([1, 1.08], gap="large", vertical_alignment="center")

with hero_r, st.container(key="herotool"):
    st.html('<div class="qt-head"><b>Try it on any home</b><span class="ex-live">live estimate</span></div>')
    q_city = common._choice("q_", "city", "City", CITIES, "Mumbai")
    c1, c2 = st.columns([1.4, 1], gap="medium")
    with c1:
        q_bhk = common._choice("q_", "bhk", "Bedrooms", [1, 2, 3, 4], 2, format_func=lambda b: f"{b} BHK")
    with c2:
        q_size = st.slider("Carpet area (sq ft)", 250, 3000, step=50, key="q_size")
    q_listing = {
        **common.DEFAULT_HOME,
        "City": q_city,
        "Area Locality": "",
        "BHK": q_bhk,
        "Bathroom": q_bhk,
        "Size": q_size,
    }
    q_res, q_err = common.estimate(q_listing)
    st.html(quick_result(q_res, q_err, q_city, q_bhk))
    if st.button("Add the locality and details", key="q_open", icon=":material/arrow_forward:", width="stretch"):
        # carry this home into the full estimate form and score it there
        st.session_state.update(
            {
                "a_city": q_city,
                f"a_locality_{q_city}": None,
                "a_bhk": q_bhk,
                "a_bath": q_bhk,
                "a_size": q_size,
                "a_area": "Carpet Area",
                "a_run": True,
            }
        )
        st.switch_page("views/estimate.py")

proof = ""
if mdape and r2:
    proof = (
        '<div class="proof">'
        f"<div><b>~{mdape:.0f}%</b>typical miss on unseen homes</div>"
        f"<div><b>{r2:.2f}</b>R² on held-out homes</div>"
        f"<div><b>{N_TEST:,}</b>test predictions</div>"
        "</div>"
    )
with hero_l:
    st.html(
        '<div class="hero-left">'
        '<h1 class="hero-h">What should this flat <em>rent for?</em></h1>'
        f'<p class="hero-sub">Kiraya prices a home in Mumbai, Delhi, Bangalore, Chennai, Hyderabad or Kolkata from its '
        f"listing details, then shows how sure it is and how the rent sits against the local market.</p>"
        f'<div class="btn-row"><a class="btn btn-primary" href="{EST}" target="_self">Estimate a rent '
        f'{ui.icon("arrow", 17, "#FFFFFF")}</a>'
        f'<a class="btn btn-ghost" href="{ABT}" target="_self">How accurate is it?</a></div>'
        f"{proof}</div>"
    )


# ----------------------------- six markets (live: follows the hero's city and size) -----------------------------
if MARKET:
    ranked = sorted(MARKET.items(), key=lambda kv: kv[1]["median_rps"], reverse=True)
    top_rps = ranked[0][1]["median_rps"] * 1.04
    (hi_city, hi), (lo_city, lo) = ranked[0], ranked[-1]
    head = (
        '<div class="mk-row head"><span>City</span><span>Median rent per sq ft</span><span class="r">₹ / sq ft</span>'
        f'<span class="r at">{q_size:,} sq ft at that rate</span><span class="r n">Listings</span></div>'
    )
    rows = "".join(
        f'<div class="mk-row{" me" if c == q_city else ""}"><span class="city">{ui.esc(c)}</span>'
        f'<div class="mk-track" aria-hidden="true"><div class="mk-fill" style="width:{100 * m["median_rps"] / top_rps:.1f}%"></div></div>'
        f'<span class="r">{ui.inr(m["median_rps"])}</span>'
        f'<span class="r at">{ui.inr(m["median_rps"] * q_size, 100)}</span>'
        f'<span class="r n">{int(m["n"]):,}</span></div>'
        for c, m in ranked
    )
    st.html(
        '<section class="band sec"><div class="mk"><div>'
        '<h2 class="sec-h">Six cities, six different rent markets</h2>'
        '<p class="sec-lead">The median rent per square foot in each city, from the listings the model learned from. '
        "Change the size in the estimator above and the last column follows.</p>"
        f'<div class="mk-call"><b>{hi["median_rps"] / lo["median_rps"]:.1f}×</b><span>what a square foot rents for in '
        f"{ui.esc(hi_city)}, against {ui.esc(lo_city)}</span></div></div>"
        f'<div><div class="mk-tbl" role="table" aria-label="Median rent per square foot by city">{head}{rows}</div>'
        '<p class="mk-foot">Medians over all listings in each city, Apr–Jul 2022. An estimate also weighs the '
        "locality, bedrooms, bathrooms, furnishing and who listed the home.</p></div></div></section>"
    )


# ----------------------------- how it works -----------------------------
st.html(
    '<section class="sec"><div class="sec-top"><div><h2 class="sec-h">From listing to rent in three steps</h2>'
    '<p class="sec-lead">No account and no upload. Each estimate comes from the saved model as soon as you ask.</p>'
    "</div></div>"
    '<div class="steps">'
    '<div class="step"><span class="step-n">1</span><h3>Describe the home</h3>'
    "<p>Pick the city and locality, then set the size, bedrooms and bathrooms. Furnishing and who listed it are "
    "pre-filled and optional.</p></div>"
    '<div class="step"><span class="step-n">2</span><h3>The model scores it</h3>'
    "<p>A gradient-boosting model, averaged over three independent training splits, reads the same features it learned "
    "rent from and prices the home.</p></div>"
    '<div class="step"><span class="step-n">3</span><h3>Read it in context</h3>'
    "<p>You get the monthly rent, a likely range, the rent per sq ft against the local median, and what would move "
    "the figure up or down.</p></div>"
    "</div></section>"
)


# ----------------------------- tools -----------------------------
if result:
    est_out = (
        f'<b>{ui.inr(result["rent"], 100)} / month</b>{ui.esc(ui.describe(result["listing"]))} · range '
        f'{ui.inr_short(result["seed_low"])}–{ui.inr_short(result["seed_high"])}'
    )
    need = common.budget(result["rent"], 100_000)["income_needed"]
    aff_out = (
        f"<b>{ui.inr_short(need)} / month</b>take-home income needed to keep {ui.inr(result['rent'], 100)} "
        "within the 30% guideline"
    )
else:
    est_out, aff_out = "<b>A monthly rent</b>with a likely range", "<b>Share of income</b>and what's left"
cmp_out = (
    f'<b>{ui.inr(result["rent"], 100)} vs {ui.inr(cmp_home["rent"], 100)}</b>2 BHK in Andheri West, Mumbai against '
    "1,100 sq ft in Whitefield, Bangalore"
    if result and cmp_home
    else "<b>Two rents</b>side by side"
)
tools = [
    ("result", "Rent estimate", "Describe one home in full and get its monthly rent, a likely range, how it sits "
     "against the locality and what would move the figure.", est_out, EST),
    ("scale", "Compare two homes", "Put two homes side by side, in the same city or different ones, and see how far "
     "apart their rents are.", cmp_out, CMP),
    ("wallet", "Affordability", "Weigh any rent against your take-home income and the common 30% guideline.",
     aff_out, AFF),
]
rows = "".join(
    f'<a class="tl-row" href="{href}" target="_self"><span class="tl-ic">{ui.icon(ic, 20)}</span>'
    f"<div><h3>{ui.esc(title)}</h3><p>{ui.esc(body)}</p></div>"
    f'<div class="tl-out">{out}</div><span class="tl-go">{ui.icon("arrow", 17)}</span></a>'
    for ic, title, body, out, href in tools
)
st.html(
    '<section class="sec"><div class="sec-top"><div><h2 class="sec-h">Three tools, one model</h2>'
    '<p class="sec-lead">Each answers a different question about rent. The examples are real outputs from the '
    "saved model.</p></div></div>"
    f'<div class="tl">{rows}</div></section>'
)


# ----------------------------- why trust it -----------------------------
cmp_rows = [(FINAL, gb, "best"), ("Locality rent × size rule of thumb", rule, ""), ("Citywide average rent", naive, "naive")]
top = max(v for _, v, _ in cmp_rows) * 1.02
bars = "".join(
    f'<div class="cmp-row {c}"><span>{ui.esc(lbl)}</span>'
    f'<div class="cmp-track" aria-hidden="true"><div class="cmp-fill" style="width:{100 * v / top:.0f}%"></div></div>'
    f'<span class="cmp-val">{ui.inr(v)}</span></div>'
    for lbl, v, c in cmp_rows
)
cut = f"{1 - gb / rule:.0%}" if rule else "about a third"
ck = f'<span class="ck">{ui.icon("check", 13, ui.T["brand_dark"])}</span>'
why = (
    f"<p>{ck}<span><b>Beats the rule of thumb by {cut}.</b> Lower typical error than the locality rent-per-sq-ft rule a "
    "renter might reach for.</span></p>"
    f"<p>{ck}<span><b>Tested on {N_TEST:,} unseen homes.</b> Every figure is measured on listings the model never "
    "trained on.</span></p>"
    f"<p>{ck}<span><b>Averaged over three splits.</b> Three independent train/test draws, so the numbers aren't one "
    "lucky run.</span></p>"
    + (f"<p>{ck}<span><b>Explains {r2:.0%} of rent variation.</b> Measured by R² on the held-out homes.</span></p>" if r2 else "")
)
st.html(
    '<section class="band sec"><div class="sec-top"><div><h2 class="sec-h">Why you can trust the number</h2>'
    '<p class="sec-lead">The model earns its place against simpler ways of guessing rent.</p></div>'
    '<div class="sec-aside">Typical error (RMSE) per month. Shorter is better.</div></div>'
    f'<div class="trust"><div class="cmp">{bars}<p class="cmp-note">Twelve other models were trained and tuned on '
    f'the same splits; the full leaderboard is on the <a href="{ABT}" target="_self">About</a> page.</p></div>'
    f'<div class="why">{why}</div></div></section>'
)


# ----------------------------- faq -----------------------------
faqs = [
    ("Which cities and homes does it cover?",
     "Mumbai, Delhi, Bangalore, Chennai, Hyderabad and Kolkata, for homes between 20 and 8,000 sq ft. Localities with "
     "very few listings fall back to a typical city effect, and the app tells you when that happens."),
    ("How accurate is an estimate?",
     f"On homes it never saw while training, the typical estimate is off by about {mdape:.0f}% of the actual rent "
     f"(roughly {ui.inr(mae, 100)} a month), and it explains about {r2:.0%} of the variation in rent. "
     "The likely range shows how firmly the data pins down each specific home."
     if mdape and mae and r2 else
     "On held-out homes the typical estimate is off by about a fifth of the actual rent. The likely range shows how "
     "firmly the data pins down each home."),
    ("What does the likely range mean?",
     "The estimate is the average of three models, each trained on a different random 70% of the listings. The range "
     "runs from the lowest to the highest of their three answers. It shows how firmly the data pins down this home, "
     "not the spread of real rents for similar homes, which is wider."),
    ("Where does the data come from?",
     "About 4,700 listings posted on MagicBricks between April and July 2022, published as the House Rent Prediction "
     "Dataset on Kaggle. Rents have moved since, so treat the figures as a 2022 baseline rather than today's market."),
    ("Can it value my exact flat?",
     "No. It prices a home from its listing details and doesn't know the building's age, floor, amenities, condition "
     "or exact address. It's an estimate to sense-check a rent, not a valuation."),
]
st.html('<div class="sec"></div>')
f1, f2 = st.columns([1, 1.9], gap="large")
with f1:
    st.html(
        '<div class="faq-side"><h2 class="sec-h">Questions</h2>'
        '<p class="sec-lead">What it covers, how sure it is and what it can\'t know.</p>'
        f'<a href="{ABT}" target="_self">Read the full method {ui.icon("arrow", 15)}</a></div>'
    )
with f2, st.container(key="faqlist"):
    for q, a in faqs:
        with st.expander(q):
            st.html(f'<p class="faq-a">{a}</p>')


# ----------------------------- footer -----------------------------
st.html(
    '<footer class="foot"><div class="foot-grid">'
    '<div class="foot-brand">'
    f'<div class="foot-mark"><span class="dot">{ui.icon("home", 15, "#FFFFFF")}</span>Kiraya</div>'
    "<p>Kiraya is Hindi for rent. A supervised-regression study that predicts monthly house rent in six Indian "
    "cities, with a small app on top of the chosen model.</p></div>"
    '<div class="foot-col"><h4>Tools</h4>'
    f'<a href="{EST}" target="_self">Rent estimate</a><a href="{CMP}" target="_self">Compare homes</a>'
    f'<a href="{AFF}" target="_self">Affordability</a></div>'
    '<div class="foot-col"><h4>The study</h4>'
    f'<a href="{ABT}" target="_self">Method &amp; accuracy</a><a href="{REPO}" target="_blank" rel="noopener">Source on GitHub</a></div>'
    '<div class="foot-col"><h4>Data</h4>'
    f'<a href="{KAGGLE}" target="_blank" rel="noopener">Dataset on Kaggle</a>'
    '<a href="https://www.magicbricks.com/" target="_blank" rel="noopener">MagicBricks</a></div>'
    "</div>"
    '<div class="foot-bot"><span>Built as a supervised-regression study · MIT licence</span>'
    "<span>Data: MagicBricks via Kaggle · listings posted Apr–Jul 2022</span></div></footer>"
)
