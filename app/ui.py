"""Look and feel of the Kiraya app: design tokens, page CSS, rupee formatting, icons and the HTML pieces
(result card, notes bar, comparison bars, problems, metric tiles).

Design system "Ledger Teal": cool off-white paper, a deep teal brand for anything good or chosen, and a single
clay-amber signal reserved for caution (rent going up, over budget). Space Grotesk carries the headlines and every
number; Inter carries the body and controls. Colour always means something; it is never decoration.
"""

import base64
import html
import re

T = {  # design tokens
    "bg": "#F4F6F5",  # cool paper, not cream
    "card": "#FFFFFF",
    "line": "#DCE4E0",
    "field": "#EEF2F0",
    "ink": "#15211E",  # green-black
    "ink2": "#566560",  # muted text, ≥ 5:1 on paper and white
    "note": "#EBF1EE",
    # teal = brand, chosen, good, within budget, rent going down
    "brand": "#0E6E5C",
    "brand_dark": "#0A5648",
    "brand_soft": "#E3F0EC",
    # amber = the one caution colour: rent going up, over budget, warnings
    "signal": "#B5651D",
    "signal_dark": "#8A4C13",
    "signal_soft": "#F6EADB",
    # legacy keys the chart code still reads, mapped to the new semantics
    "green": "#0E6E5C",  # good / within / rent down  (teal)
    "rust": "#B5651D",  # caution / over / rent up     (amber)
    "rust_dark": "#8A4C13",
    "rust_soft": "#F6EADB",
}

# ----------------------------- icons (24px line icons) -----------------------------
_ICON_PATHS = {
    "building": '<rect x="5" y="3" width="14" height="18" rx="1.5"/><path d="M9 7h1M14 7h1M9 11h1M14 11h1M9 15h1M14 15h1M10 21v-3h4v3"/>',
    "result": '<path d="M3 3v18h18"/><path d="m7 15 4-4 3 3 5-6"/>',
    "pin": '<path d="M12 21s-6-5.4-6-11a6 6 0 0 1 12 0c0 5.6-6 11-6 11z"/><circle cx="12" cy="10" r="2.2"/>',
    "tag": '<path d="M3 12V4a1 1 0 0 1 1-1h8l9 9-9 9z"/><circle cx="8" cy="8" r="1.5"/>',
    "spark": '<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6"/>',
    "scale": '<path d="M12 3v18M5 7h14M5 7l-3 7a3 3 0 0 0 6 0zM19 7l-3 7a3 3 0 0 0 6 0z"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
    "chart": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "wallet": '<rect x="3" y="6" width="18" height="14" rx="2"/><path d="M3 10h18M16 15h2M7 6l9-3 1 3"/>',
    "trophy": '<path d="M8 4h8v5a4 4 0 0 1-8 0zM8 6H5a3 3 0 0 0 3 4M16 6h3a3 3 0 0 1-3 4M12 13v4M8 21h8M9 17h6"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "home": '<path d="M3 11 12 4l9 7M5 10v10h14V10M10 20v-5h4v5"/>',
    "pie": '<path d="M12 3v9h9a9 9 0 1 1-9-9z"/><path d="M15 3.5A9 9 0 0 1 20.5 9H15z"/>',
    "bars": '<path d="M4 20h16M7 16V9M12 16V5M17 16v-4"/>',
    "arrow": '<path d="M5 12h14M13 6l6 6-6 6"/>',
    "check": '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
    "alert": '<path d="M12 4 2.8 19.5h18.4z"/><path d="M12 10v4.5M12 17v.3"/>',
    "x": '<path d="M6 6l12 12M18 6 6 18"/>',
}


def icon(name, size=20, color=None):
    """A line icon as an <img> (inline <svg> is removed by Streamlit's HTML sanitiser)."""
    color = color or T["brand"]
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{_ICON_PATHS[name]}</svg>'
    )
    data = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{data}" width="{size}" height="{size}" alt="" class="ico">'


# ----------------------------- numbers -----------------------------
def inr(value, round_to=1):
    """Rupees with Indian digit grouping: 114714 -> '₹1,14,714'; round_to=100 -> '₹1,14,700'."""
    n = int(round(float(value) / round_to) * round_to)
    digits = str(abs(n))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        digits = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head) + "," + tail
    return f"{'−' if n < 0 else ''}₹{digits}"


def inr_short(value):
    """Compact rupees: 79,300 -> '₹79k', 1,18,000 -> '₹1.18L'."""
    v = float(value)
    if v >= 1e5:
        return f"₹{v / 1e5:.2f}".rstrip("0").rstrip(".") + "L"
    if v >= 1e3:
        return f"₹{v / 1e3:.0f}k"
    return inr(v)


def signed_inr(value, round_to=100):
    return ("+" if value >= 0 else "−") + inr(abs(value), round_to)


def esc(text):
    return html.escape(str(text))


# ----------------------------- page CSS -----------------------------
DISPLAY = "'Space Grotesk', 'Inter', ui-sans-serif, system-ui, sans-serif"  # headlines and numbers
SANS = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"  # body + UI

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {{
  --sp-1:.25rem; --sp-2:.5rem; --sp-3:.75rem; --sp-4:1rem; --sp-5:1.5rem; --sp-6:2rem;
  --r-card:14px; --r-ctrl:10px; --r-chip:8px;
  --lift: 0 1px 2px rgba(21,33,30,.05), 0 8px 24px rgba(21,33,30,.07);
}}
html, body, .stApp, .stMarkdown, button, input, textarea, select, [data-baseweb] {{ font-family: {SANS}; }}
.card-title, .res-rent, .tile b, .aff-big, .diff-big, .hero-h, .hero-rent, .num,
.facts b, .lb-val, .bars-val, .stat-n {{ font-family: {DISPLAY}; }}
.res-rent, .tile b, .aff-big, .lb-val, .bars-val, .num, .facts b, .hero-rent, .stat-n {{ font-variant-numeric: tabular-nums lining-nums; }}
.stApp {{ background: {T["bg"]}; color: {T["ink"]}; }}
.block-container {{ padding-top: 6.4rem; padding-bottom: 3.5rem; max-width: min(1680px, 92vw); }}
/* larger monitors: use more width and scale type up a touch so it doesn't read small */
@media (min-width: 1600px) {{ html {{ font-size: 16.5px; }} }}
@media (min-width: 2000px) {{ html {{ font-size: 17.5px; }} .block-container {{ max-width: min(1860px, 90vw); }} }}
div[data-testid="stElementContainer"]:has(div.sr-only):not(:has(p)) {{ position: absolute; width: 1px; height: 1px; overflow: hidden; }}
.sr-only {{ position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0);
  white-space: nowrap; border: 0; }}
a {{ color: {T["brand"]}; }}

/* top bar */
header[data-testid="stHeader"] {{ background: {T["card"]}; border-bottom: 1px solid {T["line"]}; height: 5rem; }}
header[data-testid="stHeader"] > div {{ height: 100%; align-items: center; }}
img[data-testid="stHeaderLogo"] {{ height: 2.6rem; max-width: 26rem; }}
a[data-testid="stTopNavLink"] {{ background: transparent !important; border-radius: 0; padding: .7rem .2rem; margin: 0 .9rem;
  border-bottom: 2px solid transparent; }}
a[data-testid="stTopNavLink"] p {{ font-weight: 500; font-size: 1.08rem; }}
a[data-testid="stTopNavLink"] [data-testid="stIconMaterial"] {{ font-size: 1.3rem; }}
a[data-testid="stTopNavLink"][aria-current="page"] {{ border-bottom-color: {T["brand"]}; }}
a[data-testid="stTopNavLink"][aria-current="page"] p, a[data-testid="stTopNavLink"][aria-current="page"] [data-testid="stIconMaterial"] {{ color: {T["brand"]} !important; }}
.ico {{ display: inline-block; vertical-align: middle; flex: 0 0 auto; }}

/* panels: a container whose key starts with "card" is a flat panel (border, no float shadow) */
div[class*="st-key-card"] {{
  background: {T["card"]}; border: 1px solid {T["line"]}; border-radius: var(--r-card);
  padding: 22px 24px; transition: opacity .2s ease, filter .2s ease;
}}
/* the answer and the two compared homes are the focal objects: a quiet lift and a teal top edge */
div[class*="st-key-card_result"], div[class*="st-key-card_a_result"], div[class*="st-key-card_b_result"] {{
  box-shadow: var(--lift); border-top: 3px solid {T["brand"]}; padding-top: 20px;
}}
div[class*="st-key-card"][class*="_old"] {{ opacity: .5; filter: grayscale(.6); }}

.card-title {{ display: flex; align-items: center; gap: .5rem; font-size: 1.22rem; font-weight: 600; color: {T["ink"]}; margin: 0; letter-spacing: -.01em; }}
.card-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem;
  border-bottom: 1px solid {T["line"]}; padding-bottom: .8rem; margin-bottom: .2rem; }}
.card-sub {{ color: {T["ink2"]}; font-size: .97rem; margin: .3rem 0 0; line-height: 1.5; }}
.card-aside {{ color: {T["ink2"]}; font-size: .9rem; white-space: nowrap; }}
.card-aside b {{ color: {T["brand"]}; }}

/* rows of panels line up to the tallest */
div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] > div[data-testid="stLayoutWrapper"]:last-child:has(> div[class*="st-key-card"]) {{
  flex: 1 1 auto; display: flex; flex-direction: column; }}
div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] > div[data-testid="stLayoutWrapper"]:last-child > div[class*="st-key-card"] {{
  flex: 1 1 auto; }}

/* hover / focus info tips */
.tip {{ position: relative; display: inline-flex; align-items: center; justify-content: center; width: 15px; height: 15px;
  margin-left: .35rem; border-radius: 50%; border: 1.3px solid {T["ink2"]}; color: {T["ink2"]}; font-size: 10px; font-weight: 700;
  font-style: italic; font-family: Georgia, 'Times New Roman', serif; cursor: help; vertical-align: 1px; line-height: 1; }}
.tip:hover, .tip:focus {{ border-color: {T["brand"]}; color: {T["brand"]}; outline: none; }}
.tip .tip-text {{ visibility: hidden; opacity: 0; position: absolute; left: 50%; bottom: calc(100% + 8px); transform: translateX(-50%);
  width: 240px; background: {T["ink"]}; color: #FFFFFF; font: 400 .8rem/1.45 {SANS}; font-style: normal; text-align: left;
  border-radius: 8px; padding: .55rem .7rem; box-shadow: 0 8px 20px rgba(0,0,0,.18); z-index: 1000; transition: opacity .12s ease; }}
.tip .tip-text::after {{ content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px; border: 5px solid transparent;
  border-top-color: {T["ink"]}; }}
.tip:hover .tip-text, .tip:focus .tip-text {{ visibility: visible; opacity: 1; }}

/* field labels with hints on the right */
.fl {{ display: flex; justify-content: space-between; align-items: baseline; gap: .6rem; margin: .35rem 0 -.55rem; }}
.fl span:first-child {{ font-weight: 600; font-size: .93rem; color: {T["ink"]}; }}
.fl .req {{ color: {T["signal_dark"]}; margin-left: .15rem; }}
.fl .hint {{ font-size: .83rem; color: {T["ink2"]}; text-align: right; }}
.field-note {{ font-size: .86rem; color: {T["ink2"]}; line-height: 1.45; margin: -.35rem 0 .2rem; }}
.field-note b {{ color: {T["ink"]}; font-weight: 600; }}
[data-testid="stWidgetLabel"] p {{ font-size: .93rem !important; font-weight: 600; }}

/* inputs */
div[data-baseweb="select"] > div, div[data-testid="stNumberInputContainer"], div[data-baseweb="input"] {{
  background: {T["field"]} !important; border-color: {T["line"]} !important; border-radius: var(--r-ctrl) !important;
}}
div[data-testid="stNumberInputContainer"] input, div[data-baseweb="select"] input, input::placeholder {{ font-weight: 500; }}
input::placeholder {{ font-weight: 400; color: {T["ink2"]} !important; opacity: .85; }}
div[data-baseweb="select"] > div {{ min-height: 48px; }}
/* control text: up from Streamlit's 14px default so values read at the same size as the labels around them */
div[data-baseweb="select"] div, div[data-testid="stNumberInputContainer"] input, div[data-baseweb="input"] input {{ font-size: 1rem !important; }}
button[data-testid^="stBaseButton"] p {{ font-size: 1rem; }}
div[data-testid="stExpander"] summary p {{ font-size: 1rem; }}
div[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {{ font-size: .9rem; }}
div[data-testid="stNumberInput"] button {{ background: transparent; }}

/* listing-details expander as an inset panel */
div[data-testid="stExpander"] details {{ background: {T["note"]}; border: 1px solid {T["line"]}; border-radius: 12px; }}
div[data-testid="stExpander"] summary p {{ font-weight: 600; }}

/* buttons */
button[data-testid="stBaseButton-primary"] {{
  background: {T["brand"]}; border-color: {T["brand"]}; border-radius: var(--r-ctrl); min-height: 50px; font-weight: 600;
}}
button[data-testid="stBaseButton-primary"]:hover {{ background: {T["brand_dark"]}; border-color: {T["brand_dark"]}; }}
button[data-testid="stBaseButton-primary"] p {{ font-size: 1rem; font-weight: 600; }}
button[data-testid="stBaseButton-secondary"] {{ border-radius: var(--r-ctrl); min-height: 50px; border-color: {T["line"]}; background: {T["card"]}; color: {T["ink"]}; }}
button[data-testid="stBaseButton-secondary"]:hover {{ border-color: {T["brand"]}; color: {T["brand"]}; }}
button[data-testid="stBaseButton-tertiary"] p {{ font-size: .92rem; color: {T["ink2"]}; }}
a[data-testid="stPageLink-NavLink"] {{ background: {T["brand"]}; border-radius: var(--r-ctrl); padding: .7rem 1.1rem; justify-content: center; }}
a[data-testid="stPageLink-NavLink"]:hover {{ background: {T["brand_dark"]}; }}
a[data-testid="stPageLink-NavLink"] p, a[data-testid="stPageLink-NavLink"] span, a[data-testid="stPageLink-NavLink"] [data-testid="stIconMaterial"] {{ color: #FFFFFF !important; font-weight: 600; }}

/* result */
.kicker {{ font-size: .9rem; font-weight: 600; color: {T["ink2"]}; }}
.res-top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; flex-wrap: wrap; }}
.res-rent {{ font-weight: 600; font-size: 3.1rem; line-height: 1.05; letter-spacing: -.03em; color: {T["ink"]}; margin-top: .25rem; }}
.res-rent span {{ font-size: 1.2rem; font-weight: 500; color: {T["ink2"]}; letter-spacing: 0; font-family: {SANS}; }}
.res-sub {{ color: {T["ink2"]}; font-size: 1.02rem; margin-top: .4rem; }}
.status {{ display: inline-flex; align-items: center; gap: .35rem; font-size: .85rem; font-weight: 600; border-radius: 999px;
  padding: .3rem .7rem; white-space: nowrap; border: 1px solid; }}
.status.ok {{ color: {T["brand_dark"]}; background: {T["brand_soft"]}; border-color: #C5E0D6; }}
.status.old {{ color: {T["signal_dark"]}; background: {T["signal_soft"]}; border-color: #E8CFA9; }}
.status-when {{ display: block; text-align: right; font-size: .82rem; color: {T["ink2"]}; margin-top: .35rem; }}
.res-tiles {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .8rem; margin-top: 1.1rem;
  border-top: 1px solid {T["line"]}; padding-top: 1.1rem; }}
.res-tiles .tile b {{ font-size: 1.35rem; }}
.res-tiles .tile.hl {{ background: {T["brand_soft"]}; border-color: #C5E0D6; }}
.res-tiles .tile.hl b {{ color: {T["brand_dark"]}; }}
.res-mini .res-rent {{ font-size: 2.4rem; }}

/* bars comparing rent per sq ft */
.bars-row {{ display: grid; grid-template-columns: 12rem 1fr 4.5rem; gap: .7rem; align-items: center; margin: .85rem 0; font-size: .98rem; }}
.bars-track {{ background: {T["field"]}; border-radius: 6px; height: 12px; }}
.bars-fill {{ height: 12px; border-radius: 6px; background: #C3D0CB; }}
.bars-row.me {{ font-weight: 600; }}
.bars-row.me .bars-fill {{ background: {T["brand"]}; }}
.bars-val {{ text-align: right; }}

/* leaderboard */
.lb-row {{ display: grid; grid-template-columns: 1.6rem 15rem 1fr 9.5rem 8.5rem; gap: .8rem; align-items: center; padding: .42rem 0;
  border-bottom: 1px solid {T["line"]}; font-size: .93rem; }}
.lb-row:last-child {{ border-bottom: none; }}
.lb-rank {{ color: {T["ink2"]}; text-align: right; }}
.lb-track {{ background: {T["field"]}; border-radius: 5px; height: 10px; position: relative; }}
.lb-fill {{ height: 10px; border-radius: 5px; background: #C3D0CB; }}
.lb-err {{ position: absolute; top: -3px; height: 16px; border-left: 1.5px solid {T["ink2"]}; border-right: 1.5px solid {T["ink2"]}; }}
.lb-val {{ text-align: right; }}
.lb-val small {{ color: {T["ink2"]}; font-family: {SANS}; }}
.lb-row.win {{ font-weight: 600; }}
.lb-row.win .lb-fill {{ background: {T["brand"]}; }}
.lb-row.base {{ color: {T["ink2"]}; }}
.lb-row.base .lb-fill {{ background: repeating-linear-gradient(135deg, #CBD6D1 0 6px, #E2E9E5 6px 12px); }}
.lb-head {{ font-size: .83rem; font-weight: 600; color: {T["ink2"]}; padding-top: 0; }}
.tag {{ display: inline-block; font-size: .76rem; font-weight: 600; color: {T["brand_dark"]};
  background: {T["brand_soft"]}; border: 1px solid #C5E0D6; border-radius: 6px; padding: .05rem .4rem; margin-left: .4rem; vertical-align: 1px; }}
.tag.grey {{ color: {T["ink2"]}; background: {T["note"]}; border-color: {T["line"]}; }}

/* affordability */
.aff-big {{ font-size: 2rem; font-weight: 600; line-height: 1.1; }}
.aff-big small {{ font-size: 1.02rem; font-weight: 500; color: {T["ink2"]}; font-family: {SANS}; }}
.aff-track {{ position: relative; height: 14px; border-radius: 7px; background: {T["field"]}; margin: .9rem 0 .3rem; overflow: hidden; }}
.aff-fill {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 7px; }}
.aff-mark {{ position: absolute; top: -2px; bottom: -2px; border-left: 2px dashed {T["ink2"]}; }}
.aff-scale {{ position: relative; height: 1.2rem; font-size: .8rem; color: {T["ink2"]}; }}
.aff-scale span {{ position: absolute; transform: translateX(-50%); }}
.verdict-line {{ display: flex; gap: .5rem; align-items: center; font-weight: 600; margin: .6rem 0 .2rem; }}

/* notes, banners, problems */
.notes {{ background: {T["note"]}; border: 1px solid {T["line"]}; border-radius: 12px; padding: .8rem 1.1rem; }}
.notes p {{ display: flex; gap: .65rem; margin: .25rem 0; font-size: .98rem; color: {T["ink"]}; line-height: 1.45; }}
.notes p::before {{ content: ""; flex: 0 0 7px; height: 7px; margin-top: .45em; border-radius: 50%; background: {T["brand"]}; }}
.banner {{ background: {T["signal_soft"]}; border: 1px solid #E8CFA9; color: {T["ink"]}; border-radius: 12px; padding: .75rem 1rem; font-size: .98rem; }}
.problems {{ background: {T["signal_soft"]}; border: 1px solid #E8CFA9; border-radius: 12px; padding: .9rem 1.1rem; }}
.problems p {{ margin: .25rem 0; font-size: .98rem; }}
.empty {{ color: {T["ink2"]}; background: {T["note"]}; border: 1px dashed {T["line"]}; border-radius: 12px; padding: 1rem 1.1rem;
  font-size: .97rem; line-height: 1.5; }}

/* compare + about */
.diff-big {{ font-size: 1.5rem; font-weight: 600; color: {T["ink"]}; letter-spacing: -.01em; }}
.tiles {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin: 1rem 0 .4rem; }}
.tiles-4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
.tile {{ border: 1px solid {T["line"]}; border-radius: var(--r-ctrl); padding: .85rem 1rem; background: #FBFCFC; min-width: 0; }}
.tile small {{ display: block; color: {T["ink2"]}; font-size: .86rem; font-weight: 600; }}
.tile b {{ display: block; font-size: 1.6rem; font-weight: 600; margin: .3rem 0 .15rem; letter-spacing: -.01em; }}
.tile span {{ display: block; color: {T["ink2"]}; font-size: .9rem; line-height: 1.4; }}
.facts {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .8rem; margin: 1.1rem 0 0; }}
.facts div {{ border-top: 1px solid {T["ink"]}; padding-top: .65rem; }}
.facts small {{ display: block; font-size: .83rem; font-weight: 600; color: {T["ink2"]}; }}
.facts b {{ display: block; font-size: 1.1rem; margin-top: .2rem; }}
.facts span {{ display: block; font-size: .9rem; color: {T["ink2"]}; line-height: 1.35; margin-top: .1rem; }}
.verdict {{ background: {T["brand_soft"]}; border: 1px solid #C5E0D6; border-radius: 12px; padding: .85rem 1.05rem;
  margin: 1.1rem 0 0; font-size: 1.03rem; line-height: 1.55; color: {T["ink"]}; }}
.muted {{ color: {T["ink2"]}; font-size: .97rem; line-height: 1.55; }}
.lead {{ font-size: 1.05rem; line-height: 1.6; color: {T["ink"]}; }}

/* =========================== shared page furniture =========================== */
::selection {{ background: {T["brand_soft"]}; color: {T["brand_dark"]}; }}
html {{ scrollbar-color: #C3D0CB transparent; }}
input, textarea {{ caret-color: {T["brand"]}; }}
a {{ text-underline-offset: .18em; }}

/* page header for the tool pages: says where you are and what the page answers */
.ph {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 1.5rem; flex-wrap: wrap;
  padding: .4rem 0 1.1rem; margin-bottom: .3rem; border-bottom: 1px solid {T["line"]}; }}
.ph h1 {{ font-family: {DISPLAY}; font-size: clamp(1.9rem, 2.8vw, 2.6rem); font-weight: 600; letter-spacing: -.025em;
  line-height: 1.1; color: {T["ink"]}; margin: 0; padding: 0; text-wrap: balance; }}
.ph p {{ color: {T["ink2"]}; font-size: 1.08rem; line-height: 1.5; margin: .45rem 0 0; max-width: 64ch; }}
.ph-aside {{ display: flex; gap: 1.6rem; color: {T["ink2"]}; font-size: .9rem; }}
.ph-aside b {{ display: block; font-family: {DISPLAY}; font-variant-numeric: tabular-nums; font-size: 1.2rem; font-weight: 600; color: {T["ink"]}; }}

/* segmented controls: the chosen option reads as chosen */
button[data-testid="stBaseButton-segmented_controlActive"] {{ background: {T["brand"]} !important; border-color: {T["brand"]} !important; color: #fff !important; }}
button[data-testid="stBaseButton-segmented_controlActive"] p {{ color: #fff !important; font-weight: 600; }}
button[data-testid="stBaseButton-segmented_control"] {{ background: {T["card"]}; border-color: {T["line"]}; }}
button[data-testid="stBaseButton-segmented_control"]:hover {{ border-color: {T["brand"]}; color: {T["brand"]}; }}

/* =========================== landing page =========================== */
.stat-n {{ font-weight: 600; color: {T["brand_dark"]}; }}

/* hero: headline left, a working estimator right */
.hero-left {{ padding: 1.2rem 0 .6rem; }}
.hero-h {{ font-size: clamp(2.6rem, 4.4vw, 4.1rem); line-height: 1; letter-spacing: -.04em; font-weight: 600; color: {T["ink"]};
  margin: 0 0 1.25rem; padding: 0; max-width: 13ch; text-wrap: balance; }}
.hero-h em {{ font-style: normal; color: {T["brand"]}; }}
.hero-sub {{ font-size: 1.22rem; line-height: 1.55; color: {T["ink2"]}; max-width: 46ch; margin: 0 0 1.8rem; }}
.btn-row {{ display: flex; gap: .8rem; flex-wrap: wrap; margin-bottom: 2.1rem; }}
.btn {{ display: inline-flex; align-items: center; gap: .55rem; font-weight: 600; font-size: 1.05rem; border-radius: var(--r-ctrl);
  padding: .82rem 1.3rem; text-decoration: none; border: 1px solid transparent;
  transition: background .15s ease, border-color .15s ease, color .15s ease, gap .15s ease; }}
.btn-primary {{ background: {T["brand"]}; color: #fff !important; }}
.btn-primary:hover {{ background: {T["brand_dark"]}; gap: .75rem; }}
.btn-ghost {{ background: {T["card"]}; color: {T["ink"]} !important; border-color: {T["line"]}; }}
.btn-ghost:hover {{ border-color: {T["brand"]}; color: {T["brand"]} !important; }}
.proof {{ display: grid; grid-template-columns: repeat(3, auto); justify-content: start; gap: 0 2.4rem; border-top: 1px solid {T["line"]}; padding-top: 1.1rem; max-width: 34rem; }}
.proof div {{ font-size: .92rem; color: {T["ink2"]}; line-height: 1.35; }}
.proof b {{ display: block; font-family: {DISPLAY}; font-variant-numeric: tabular-nums; font-size: 1.45rem; font-weight: 600; letter-spacing: -.02em; color: {T["ink"]}; }}

/* the hero estimator panel (a real form: Streamlit widgets inside a keyed container) */
div[class*="st-key-herotool"] {{ background: {T["card"]}; border: 1px solid {T["line"]}; border-radius: 18px;
  box-shadow: 0 1px 2px rgba(21,33,30,.05), 0 24px 48px -12px rgba(21,33,30,.16); padding: 1.15rem 1.4rem 1.3rem; gap: .85rem; }}
.qt-head {{ display: flex; justify-content: space-between; align-items: center; gap: 1rem; margin: -1.15rem -1.4rem .1rem; padding: .7rem 1.4rem;
  background: {T["brand_soft"]}; border-bottom: 1px solid #C5E0D6; border-radius: 18px 18px 0 0; }}
.qt-head b {{ font-size: .95rem; color: {T["brand_dark"]}; font-weight: 600; }}
.ex-live {{ display: inline-flex; align-items: center; gap: .4rem; font-size: .84rem; font-weight: 600; color: {T["brand_dark"]}; }}
.ex-live::before {{ content: ""; width: 7px; height: 7px; border-radius: 50%; background: {T["brand"]}; animation: exlive 2.4s ease-in-out infinite; }}
@keyframes exlive {{ 0%,100% {{ opacity: .35; }} 50% {{ opacity: 1; }} }}
div[class*="st-key-herotool"] [data-testid="stWidgetLabel"] p {{ font-size: .88rem !important; color: {T["ink2"]}; }}
div[class*="st-key-herotool"] button[data-testid^="stBaseButton-segmented_control"] {{ min-height: 2.4rem; padding: 0 .62rem; }}
div[class*="st-key-herotool"] button[data-testid^="stBaseButton-segmented_control"] p {{ font-size: .9rem; }}
.q-out {{ border-top: 1px solid {T["line"]}; padding-top: 1rem; }}
.q-top {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem; flex-wrap: wrap; }}
.q-rent {{ font-family: {DISPLAY}; font-variant-numeric: tabular-nums; font-size: clamp(2.4rem, 3.4vw, 3.2rem); font-weight: 600;
  letter-spacing: -.035em; line-height: 1; color: {T["ink"]}; animation: qin .45s cubic-bezier(.16,1,.3,1); }}
.q-rent span {{ font-family: {SANS}; font-size: 1rem; font-weight: 500; letter-spacing: 0; color: {T["ink2"]}; }}
@keyframes qin {{ from {{ opacity: .35; transform: translateY(4px); filter: blur(2px); }} to {{ opacity: 1; transform: none; filter: none; }} }}
.q-range {{ text-align: right; font-size: .88rem; color: {T["ink2"]}; }}
.q-range b {{ display: block; font-family: {DISPLAY}; font-variant-numeric: tabular-nums; font-size: 1.25rem; font-weight: 600; color: {T["brand_dark"]}; }}
.q-desc {{ font-size: .95rem; color: {T["ink2"]}; margin: .45rem 0 .2rem; }}
.ex-mini {{ margin: .85rem 0 0; }}
.ex-mini-row {{ display: grid; grid-template-columns: 8.5rem 1fr 3.4rem; gap: .6rem; align-items: center; font-size: .9rem; color: {T["ink2"]}; margin: .4rem 0; }}
.ex-mini-row.me {{ color: {T["ink"]}; font-weight: 600; }}
.ex-mini-track {{ background: {T["field"]}; border-radius: 5px; height: 9px; }}
.ex-mini-fill {{ height: 9px; border-radius: 5px; background: #C3D0CB; transition: width .45s cubic-bezier(.16,1,.3,1); }}
.ex-mini-row.me .ex-mini-fill {{ background: {T["brand"]}; }}
.ex-mini-val {{ text-align: right; font-family: {DISPLAY}; font-variant-numeric: tabular-nums; }}
.q-note {{ font-size: .85rem; color: {T["ink2"]}; line-height: 1.45; margin: .7rem 0 0; }}

/* section scaffolding */
.sec {{ margin: 5.2rem 0 0; }}
.sec-top {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 2rem; }}
.sec-h {{ font-family: {DISPLAY}; font-size: clamp(1.9rem, 2.8vw, 2.6rem); font-weight: 600; letter-spacing: -.028em; line-height: 1.1;
  color: {T["ink"]}; margin: 0; padding: 0; text-wrap: balance; max-width: 22ch; }}
.sec-lead {{ color: {T["ink2"]}; font-size: 1.1rem; line-height: 1.6; max-width: 56ch; margin: .7rem 0 0; }}
.sec-aside {{ color: {T["ink2"]}; font-size: .93rem; max-width: 26ch; text-align: right; line-height: 1.45; }}
/* a full-bleed white band, drawn with a shadow + clip so it never causes sideways scroll */
.band {{ background: {T["card"]}; box-shadow: 0 0 0 100vmax {T["card"]}; clip-path: inset(0 -100vmax);
  border-top: 1px solid {T["line"]}; padding: 4.2rem 0 4.4rem; margin-top: 5rem; }}
.band.sec {{ margin-top: 4.6rem; }}

/* the six markets */
.mk {{ display: grid; grid-template-columns: minmax(0, .8fr) minmax(0, 1.5fr); gap: 4rem; align-items: start; }}
.mk-call {{ margin-top: 1.8rem; display: flex; align-items: baseline; gap: .9rem; border-top: 1px solid {T["line"]}; padding-top: 1.2rem; }}
.mk-call b {{ font-family: {DISPLAY}; font-variant-numeric: tabular-nums; font-size: 2.6rem; font-weight: 600; letter-spacing: -.035em; color: {T["brand_dark"]}; line-height: 1; }}
.mk-call span {{ color: {T["ink2"]}; font-size: .98rem; line-height: 1.45; max-width: 26ch; }}
.mk-row {{ display: grid; grid-template-columns: 9.4rem minmax(0, 1fr) 4.4rem 10rem 4.6rem; gap: 1rem; align-items: center;
  padding: .82rem .2rem; border-bottom: 1px solid {T["line"]}; font-size: 1.02rem; color: {T["ink"]}; }}
.mk-row.head {{ font-size: .84rem; font-weight: 600; color: {T["ink2"]}; padding-top: 0; }}
.mk-row .r {{ text-align: right; font-family: {DISPLAY}; font-variant-numeric: tabular-nums; }}
.mk-row.head .r {{ font-family: {SANS}; }}
.mk-row .n {{ color: {T["ink2"]}; font-size: .92rem; }}
.mk-track {{ height: 12px; background: {T["field"]}; border-radius: 6px; }}
.mk-fill {{ height: 12px; border-radius: 6px; background: #C3D0CB; transition: width .45s cubic-bezier(.16,1,.3,1); }}
.mk-row.me {{ font-weight: 600; }}
.mk-row.me .mk-fill {{ background: {T["brand"]}; }}
.mk-row.me .city::after {{ content: "your pick"; margin-left: .45rem; font: 600 .72rem/1 {SANS}; color: {T["brand_dark"]}; background: {T["brand_soft"]};
  border: 1px solid #C5E0D6; border-radius: 5px; padding: .15rem .3rem; vertical-align: 2px; }}
.mk-foot {{ color: {T["ink2"]}; font-size: .9rem; line-height: 1.5; margin: .9rem 0 0; }}

/* how it works: a real sequence, so the steps are numbered */
.steps {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 2.2rem; }}
.step {{ position: relative; padding-top: 1.4rem; border-top: 2px solid {T["line"]}; }}
.step:first-child {{ border-top-color: {T["brand"]}; }}
.step-n {{ position: absolute; top: -.95rem; left: 0; display: inline-flex; width: 1.9rem; height: 1.9rem; align-items: center; justify-content: center;
  border-radius: 8px; background: {T["brand"]}; color: #fff; font-family: {DISPLAY}; font-weight: 600; font-size: .95rem; }}
.step h3 {{ margin: .4rem 0 .4rem; padding: 0; font-size: 1.22rem; font-weight: 600; color: {T["ink"]}; }}
.step p {{ color: {T["ink2"]}; font-size: 1.02rem; line-height: 1.55; margin: 0; max-width: 40ch; }}

/* the tools, as an index of rows: what it answers, a real output, a way in */
.tl {{ border-top: 1px solid {T["ink"]}; }}
a.tl-row {{ display: grid; grid-template-columns: 2.6rem minmax(0, 1.05fr) minmax(0, 1fr) 2.4rem; gap: 1.6rem; align-items: center;
  padding: 1.6rem .4rem; border-bottom: 1px solid {T["line"]}; text-decoration: none; color: {T["ink"]};
  transition: background .2s ease, padding .2s cubic-bezier(.16,1,.3,1); }}
a.tl-row:hover {{ background: {T["card"]}; padding-left: 1rem; padding-right: 1rem; }}
.tl-ic {{ width: 2.6rem; height: 2.6rem; border-radius: 10px; background: {T["brand_soft"]}; display: inline-flex; align-items: center; justify-content: center; }}
.tl-row h3 {{ font-family: {DISPLAY}; font-size: 1.5rem; font-weight: 600; letter-spacing: -.02em; margin: 0 0 .3rem; padding: 0; color: {T["ink"]}; }}
.tl-row p {{ color: {T["ink2"]}; font-size: 1.02rem; line-height: 1.5; margin: 0; max-width: 46ch; }}
.tl-out {{ font-size: .95rem; color: {T["ink2"]}; line-height: 1.45; border-left: 1px solid {T["line"]}; padding-left: 1.4rem; }}
.tl-out b {{ display: block; font-family: {DISPLAY}; font-variant-numeric: tabular-nums; font-size: 1.38rem; font-weight: 600; color: {T["ink"]}; letter-spacing: -.015em; margin-bottom: .1rem; }}
.tl-go {{ justify-self: end; width: 2.4rem; height: 2.4rem; border-radius: 50%; border: 1px solid {T["line"]}; display: inline-flex; align-items: center; justify-content: center;
  transition: background .2s ease, border-color .2s ease, transform .2s cubic-bezier(.16,1,.3,1); }}
a.tl-row:hover .tl-go {{ background: {T["brand"]}; border-color: {T["brand"]}; transform: translateX(3px); }}
a.tl-row:hover .tl-go img {{ filter: brightness(0) invert(1); }}

/* why trust it: comparison + supporting points */
.trust {{ display: grid; grid-template-columns: 1.3fr .9fr; gap: 3.4rem; align-items: start; }}
.cmp-row {{ display: grid; grid-template-columns: 13rem 1fr 5.5rem; gap: 1rem; align-items: center; font-size: 1rem; margin: 1rem 0; color: {T["ink2"]}; }}
.cmp-row.best {{ color: {T["ink"]}; font-weight: 600; }}
.cmp-track {{ background: {T["field"]}; border-radius: 6px; height: 15px; }}
.cmp-fill {{ height: 15px; border-radius: 6px; background: #C3D0CB; }}
.cmp-row.best .cmp-fill {{ background: {T["brand"]}; }}
.cmp-row.naive .cmp-fill {{ background: repeating-linear-gradient(135deg, #CBD6D1 0 6px, #E2E9E5 6px 12px); }}
.cmp-val {{ text-align: right; font-family: {DISPLAY}; font-variant-numeric: tabular-nums; }}
.cmp-note {{ color: {T["ink2"]}; font-size: .92rem; line-height: 1.5; margin: 1.1rem 0 0; max-width: 60ch; }}
.why p {{ display: flex; gap: .7rem; align-items: flex-start; margin: 0 0 1.05rem; font-size: 1.02rem; line-height: 1.5; color: {T["ink"]}; }}
.why p > span {{ flex: 1; min-width: 0; }}
.why p b {{ font-weight: 600; }}
.why .ck {{ flex: 0 0 1.4rem; height: 1.4rem; margin-top: .02rem; display: inline-flex; align-items: center; justify-content: center;
  border-radius: 50%; background: {T["brand_soft"]}; }}

/* faq: Streamlit expanders restyled as a divided list, beside its heading */
div[class*="st-key-faqlist"] div[data-testid="stExpander"] details {{ background: transparent; border: none; border-top: 1px solid {T["line"]}; border-radius: 0; }}
div[class*="st-key-faqlist"] div[data-testid="stExpander"]:last-of-type details {{ border-bottom: 1px solid {T["line"]}; }}
div[class*="st-key-faqlist"] div[data-testid="stExpander"] summary {{ padding: 1.1rem .2rem; }}
div[class*="st-key-faqlist"] div[data-testid="stExpander"] summary:hover p {{ color: {T["brand"]}; }}
div[class*="st-key-faqlist"] div[data-testid="stExpander"] summary p {{ font-size: 1.12rem; font-weight: 600; color: {T["ink"]}; }}
div[class*="st-key-faqlist"] div[data-testid="stExpanderDetails"] {{ padding: 0 .2rem 1.2rem; }}
div[class*="st-key-faqlist"] {{ gap: 0; }}
.faq-a {{ color: {T["ink2"]}; font-size: 1.02rem; line-height: 1.6; max-width: 68ch; margin: 0; }}
.faq-side {{ position: sticky; top: 5rem; }}
.faq-side a {{ display: inline-flex; align-items: center; gap: .4rem; margin-top: 1.2rem; font-weight: 600; font-size: 1rem; text-decoration: none; }}
.faq-side a:hover {{ gap: .6rem; }}

/* footer */
.foot {{ margin-top: 5rem; border-top: 1px solid {T["line"]}; padding-top: 2.4rem; }}
.foot-grid {{ display: grid; grid-template-columns: 2.2fr 1fr 1fr 1fr; gap: 2rem; }}
.foot-brand p {{ color: {T["ink2"]}; font-size: .97rem; line-height: 1.55; max-width: 36ch; margin: .6rem 0 0; }}
.foot-mark {{ display: flex; align-items: center; gap: .5rem; font-family: {DISPLAY}; font-size: 1.3rem; font-weight: 600; color: {T["ink"]}; }}
.foot-mark .dot {{ width: 1.5rem; height: 1.5rem; border-radius: 7px; background: {T["brand"]}; display: inline-flex; align-items: center; justify-content: center; }}
.foot-col h4 {{ font-size: .88rem; font-weight: 600; color: {T["ink2"]}; margin: 0 0 .7rem; padding: 0; }}
.foot-col a {{ display: block; color: {T["ink"]}; text-decoration: none; font-size: .98rem; padding: .25rem 0; }}
.foot-col a:hover {{ color: {T["brand"]}; }}
.foot-bot {{ display: flex; justify-content: space-between; gap: 1.2rem; flex-wrap: wrap; border-top: 1px solid {T["line"]};
  margin-top: 2rem; padding-top: 1.2rem; color: {T["ink2"]}; font-size: .9rem; }}

/* compare: the answer first */
.vs {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.25fr); gap: 2.4rem; align-items: center; }}
.vs-row {{ display: grid; grid-template-columns: 4.6rem minmax(0, 1fr) 6.2rem; gap: .9rem; align-items: center; margin: .55rem 0; font-size: .98rem; }}
.vs-row span:first-child {{ font-weight: 600; }}
.vs-track {{ height: 14px; background: {T["field"]}; border-radius: 7px; }}
.vs-fill {{ height: 14px; border-radius: 7px; background: #C3D0CB; }}
.vs-row.hi .vs-fill {{ background: {T["brand"]}; }}
.vs-row .r {{ text-align: right; font-family: {DISPLAY}; font-variant-numeric: tabular-nums; font-weight: 600; }}
.vs-sub {{ color: {T["ink2"]}; font-size: .96rem; line-height: 1.5; margin: .5rem 0 0; }}

/* affordability: income presets in the empty state */
.need {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .8rem; margin: 1rem 0 .3rem; }}

/* focus */
input, input:focus, input:focus-visible, textarea:focus, textarea:focus-visible {{ outline: none !important; box-shadow: none !important; }}
div[data-baseweb="select"] > div:focus-within, div[data-testid="stNumberInputContainer"]:focus-within {{
  border-color: {T["brand"]} !important; box-shadow: 0 0 0 3px {T["brand_soft"]} !important;
}}
div[data-testid="stNumberInputContainer"] div[data-baseweb="input"],
div[data-testid="stNumberInputContainer"] div[data-baseweb="input"]:focus-within {{ border: none !important; box-shadow: none !important; }}
div[class*="st-key-"][class*="city"] input, div[class*="st-key-"][class*="_area"] input {{ caret-color: transparent; cursor: pointer; }}
button:focus-visible, a:focus-visible, summary:focus-visible {{ outline: 2px solid {T["brand"]} !important; outline-offset: 2px; }}

@media (prefers-reduced-motion: reduce) {{
  .ex-live::before, .q-rent {{ animation: none; }}
  * {{ transition: none !important; }}
}}
@media (max-width: 1100px) {{
  /* the hero's two columns stack before the estimator gets too narrow for its controls */
  div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"] div[class*="st-key-herotool"]) {{ flex-wrap: wrap; gap: 1.5rem; }}
  div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"] div[class*="st-key-herotool"]) > div[data-testid="stColumn"] {{
    flex: 1 1 100% !important; width: 100% !important; }}
  .hero-left {{ padding-bottom: 0; }}
  .hero-sub {{ max-width: 60ch; }}
  .trust {{ grid-template-columns: 1fr; gap: 1.8rem; }}
  .mk {{ grid-template-columns: 1fr; gap: 2rem; }}
  .vs {{ grid-template-columns: 1fr; gap: 1rem; }}
  .faq-side {{ position: static; }}
}}
@media (max-width: 900px) {{
  .facts, .tiles-4 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .steps {{ grid-template-columns: 1fr; gap: 2rem; }}
  .step {{ padding-top: 1.2rem; }}
  a.tl-row {{ grid-template-columns: 2.6rem minmax(0, 1fr) 2.4rem; gap: 1rem 1.2rem; }}
  .tl-out {{ grid-column: 2 / 3; grid-row: 2; border-left: none; padding-left: 0; }}
  .tl-go {{ grid-column: 3; grid-row: 1; }}
  .foot-grid {{ grid-template-columns: 1fr 1fr; gap: 1.6rem; }}
  .foot-brand {{ grid-column: 1 / -1; }}
  .mk-row {{ grid-template-columns: 8.6rem minmax(0, 1fr) 4rem 9.4rem; }}
  .mk-row .n {{ display: none; }}
}}
@media (max-width: 700px) {{
  .hero-h {{ font-size: 2.55rem; }}
  .hero-sub {{ font-size: 1.1rem; }}
  .proof {{ gap: 0 1.4rem; }}
  .proof b {{ font-size: 1.15rem; }}
  .proof div {{ font-size: .84rem; }}
  .q-range {{ text-align: left; }}
  .sec-aside {{ text-align: left; max-width: none; }}
  a.tl-row {{ grid-template-columns: minmax(0, 1fr) 2.4rem; padding: 1.4rem 0; }}
  a.tl-row:hover {{ padding-left: 0; padding-right: 0; background: transparent; }}
  .tl-ic {{ display: none; }}
  .tl-out {{ grid-column: 1; }}
  .tl-go {{ grid-column: 2; }}
  .sec {{ margin-top: 3.6rem; }}
  .band {{ padding: 3rem 0; }}
  .cmp-row {{ grid-template-columns: 8rem 1fr 4.4rem; gap: .6rem; font-size: .92rem; }}
  .foot-grid {{ grid-template-columns: 1fr 1fr; }}
  .res-rent {{ font-size: 2.3rem; }}
  .res-tiles {{ grid-template-columns: 1fr; }}
  .bars-row {{ grid-template-columns: 7.5rem 1fr 3.8rem; }}
  .lb-row {{ grid-template-columns: 1.2rem 1fr 5.8rem 3.9rem; gap: .5rem; font-size: .87rem; }}
  .lb-track, .lb-barh {{ display: none; }}
  .lb-head {{ font-size: .76rem; }}
  .lb-val small {{ display: none; }}
  .tiles {{ grid-template-columns: 1fr; }}
  .facts {{ grid-template-columns: 1fr; }}
  .mk-row {{ grid-template-columns: 6.2rem minmax(0, 1fr) 3.6rem; gap: .7rem; font-size: .94rem; }}
  .mk-row .at {{ display: none; }}
  .mk-row.me .city::after {{ display: none; }}
  .mk-call b {{ font-size: 2.1rem; }}
  .ex-mini-row {{ grid-template-columns: 6.8rem 1fr 3rem; }}
  .vs-row {{ grid-template-columns: 4rem minmax(0, 1fr) 5.4rem; gap: .6rem; }}
  .ph-aside {{ display: none; }}
  div[class*="st-key-card"] {{ padding: 18px 16px; }}
  div[class*="st-key-herotool"] {{ padding: 1rem 1rem 1.1rem; }}
  .qt-head {{ margin: -1rem -1rem .1rem; padding: .65rem 1rem; }}
}}
</style>
"""


# ----------------------------- pieces -----------------------------
def page_head(title, lead="", aside=None):
    """The heading of a tool page: where you are and the question the page answers. aside: [(value, label)]."""
    side = ""
    if aside:
        side = '<div class="ph-aside">' + "".join(f"<div><b>{esc(v)}</b>{esc(k)}</div>" for v, k in aside) + "</div>"
    sub = f"<p>{lead}</p>" if lead else ""
    return f'<div class="ph"><div><h1>{esc(title)}</h1>{sub}</div>{side}</div>'


def card_head(icon_name, title, subtitle="", aside=""):
    sub = f'<p class="card-sub">{subtitle}</p>' if subtitle else ""
    side = f'<div class="card-aside">{aside}</div>' if aside else ""
    return f'<div class="card-head"><div><p class="card-title">{icon(icon_name)}{esc(title)}</p>{sub}</div>{side}</div>'


def card_title(icon_name, title, aside=""):
    side = f'<span class="card-aside">{aside}</span>' if aside else ""
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap">'
        f'<p class="card-title">{icon(icon_name)}{esc(title)}</p>{side}</div>'
    )


def tip(text):
    """A small (i) that shows `text` on hover or keyboard focus."""
    return (
        f'<span class="tip" tabindex="0" role="note" aria-label="{esc(text)}">i'
        f'<span class="tip-text" aria-hidden="true">{esc(text)}</span></span>'
    )


def field_label(text, hint="", required=False, info=None):
    star = '<span class="req" aria-label="required">*</span>' if required else ""
    extra = tip(info) if info else ""
    return f'<div class="fl"><span>{esc(text)}{star}{extra}</span><span class="hint">{esc(hint)}</span></div>'


def field_note(text):
    return f'<p class="field-note">{text}</p>'


def where(listing):
    loc = listing.get("Area Locality")
    return f"{loc}, {listing['City']}" if loc and loc != "unknown" else listing["City"]


def describe(listing):
    """'2 BHK · Andheri West, Mumbai · 850 sq ft · Semi-Furnished'"""
    bits = [f"{int(listing['BHK'])} BHK", where(listing), f"{float(listing['Size']):,.0f} sq ft"]
    if listing.get("Furnishing Status"):
        bits.append(listing["Furnishing Status"])
    return " · ".join(bits)


def status_pill(stale, when=None):
    if stale:
        pill = f'<span class="status old">{icon("alert", 13, T["signal_dark"])}Outdated — details changed</span>'
        sub = f"last estimated {esc(when)}" if when else ""
    else:
        pill = f'<span class="status ok">{icon("check", 13, T["brand_dark"])}Up to date</span>'
        sub = f"estimated {esc(when)}" if when else ""
    return f'<div>{pill}<span class="status-when">{sub}</span></div>'


def result_card(result, when=None, stale=False, compact=False):
    """The main result: monthly rent, the likely range (lowest to highest of the three seed models) and rent per
    sq ft. `when` is the time it was calculated; `stale` marks it as no longer matching the form."""
    listing = result["listing"]
    cells = [
        ("hl", "Likely range", f"{inr_short(result['seed_low'])} – {inr_short(result['seed_high'])}"),
        ("", "Rent per sq ft", inr(result["rent_per_sqft"])),
    ]
    tiles_html = "".join(f'<div class="tile {c}"><small>{esc(a)}</small><b>{esc(b)}</b></div>' for c, a, b in cells)
    return f"""
<div class="{"res-mini" if compact else "res"}">
  <div class="res-top">
    <div><div class="kicker">Estimated monthly rent</div>
      <div class="res-rent">{inr(result["rent"], 100)}<span> / month</span></div>
      <div class="res-sub">{esc(describe(listing))}</div></div>
    {status_pill(stale, when)}
  </div>
  <div class="res-tiles">{tiles_html}</div>
</div>"""


def compare_bars(result):
    """Rent per sq ft: this home vs the locality and city medians, with one sentence (or an empty state)."""
    market = result.get("market") or {}
    city, loc, listing, me = market.get("city"), market.get("locality"), result["listing"], result["rent_per_sqft"]
    if not city:
        return empty(
            "No neighbourhood comparison is available: the saved model files have no market medians. "
            "Run <code>python run_all.py</code> to rebuild them."
        )
    rows = [("This home (estimate)", me, "me")]
    if loc:
        rows.append((f"{listing['Area Locality']} median", loc["median_rps"], ""))
    rows.append((f"{listing['City']} median", city["median_rps"], ""))
    top = max(v for _, v, _ in rows) * 1.06
    bars = "".join(
        f'<div class="bars-row {c}"><span>{esc(label)}</span><div class="bars-track" aria-hidden="true">'
        f'<div class="bars-fill" style="width:{100 * v / top:.1f}%"></div></div><span class="bars-val">{inr(v)}</span></div>'
        for label, v, c in rows
    )
    ref, name = (loc["median_rps"], listing["Area Locality"]) if loc else (city["median_rps"], listing["City"])
    gap = me / ref - 1
    sentence = (
        f"About the same per sq ft as the typical home in {esc(name)}."
        if abs(gap) < 0.05
        else f"{abs(gap):.0%} {'above' if gap > 0 else 'below'} the typical rent per sq ft in {esc(name)}."
    )
    basis = (
        f" Locality median from {int(loc['n'])} listings."
        if loc
        else " Too few listings in this locality for a local median, so the city median is shown."
    )
    size = float(listing["Size"])
    at_rate = f" At the {esc(name)} median rate, this {size:,.0f} sq ft home would rent for about {inr(ref * size, 100)} a month."
    return f'{bars}<p class="muted" style="margin-top:.6rem">{sentence}{basis}{at_rate}</p>'


def budget_summary(b):
    """Headline of the affordability calculator: share of income going to housing, a verdict in words (not only
    colour) and a bar with the chosen limit and the limit + 10 points marked."""
    color = {"ok": T["brand_dark"], "stretched": T["signal_dark"], "heavy": T["signal_dark"]}[b["level"]]
    mark = icon({"ok": "check", "stretched": "alert", "heavy": "x"}[b["level"]], 17, color)
    t, t2 = b["target"] * 100, min(b["target"] * 100 + 10, 100)
    width = min(b["share"], 1.0) * 100
    return f"""
<div class="aff-big">{b["share"]:.0%} <small>of take-home income goes to housing</small></div>
<div class="verdict-line" style="color:{color}">{mark}<span>{esc(b["verdict"])}</span></div>
<div class="aff-track" role="img" aria-label="{b["share"]:.0%} of income; marks at {t:.0f}% and {t2:.0f}%">
  <div class="aff-fill" style="width:{width:.1f}%;background:{color}"></div>
  <div class="aff-mark" style="left:{t:.0f}%"></div><div class="aff-mark" style="left:{t2:.0f}%"></div>
</div>
<div class="aff-scale"><span style="left:0;transform:none">0%</span><span style="left:{t:.0f}%">{t:.0f}%</span>
  <span style="left:{t2:.0f}%">{t2:.0f}%</span><span style="right:0;left:auto;transform:none">100%</span></div>"""


def leaderboard(rows, chosen, rule=None):
    """rows: [(model, rmse_mean, rmse_std, median_pct_error)] best first. Bars start at zero; whiskers show ±1 std
    over the seeds; the chosen model and the rule-of-thumb baseline are tagged in words, not only in colour."""
    top = max(m + s for _, m, s, _ in rows) * 1.04
    head = (
        '<div class="lb-row lb-head"><span></span><span>Model</span><span class="lb-barh">Test RMSE (bars start at ₹0)</span>'
        '<span class="lb-val">RMSE ± seeds</span><span class="lb-val">Median % error</span></div>'
    )
    out = []
    for i, (name, m, s, pct) in enumerate(rows, 1):
        win, base = name == chosen, name == rule
        tag = (
            '<span class="tag">Chosen</span>' if win else '<span class="tag grey">Rule of thumb</span>' if base else ""
        )
        out.append(
            f'<div class="lb-row{" win" if win else ""}{" base" if base else ""}"><span class="lb-rank">{i}</span>'
            f"<span>{esc(name)}{tag}</span>"
            f'<div class="lb-track" aria-hidden="true"><div class="lb-fill" style="width:{100 * m / top:.1f}%"></div>'
            f'<div class="lb-err" style="left:{100 * (m - s) / top:.1f}%;width:{100 * 2 * s / top:.1f}%"></div></div>'
            f'<span class="lb-val">{inr(m)} <small>± {inr(s)}</small></span>'
            f'<span class="lb-val">{"–" if pct is None else f"{pct:.1f}%"}</span></div>'
        )
    return '<div class="lb">' + head + "".join(out) + "</div>"


def sr_table(caption, columns, rows):
    """A table only screen readers see — the values behind a chart."""
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in r) + "</tr>" for r in rows)
    # the wrapper (not the table) is clipped: a table ignores height/overflow and would stretch the page
    return f'<div class="sr-only"><table><caption>{esc(caption)}</caption><tr>{head}</tr>{body}</table></div>'


def notes(messages):
    if not messages:
        return ""
    return '<div class="notes">' + "".join(f"<p>{esc(m)}</p>" for m in messages) + "</div>"


def problems(messages, heading="This home can't be estimated yet:", footer=""):
    body = "".join(f"<p>{esc(m)}</p>" for m in messages)
    foot = f'<p class="muted" style="margin-top:.5rem">{footer}</p>' if footer else ""
    return f'<div class="problems" role="alert"><p><strong>{esc(heading)}</strong></p>{body}{foot}</div>'


def banner(text):
    return f'<div class="banner" role="status">{text}</div>'


def empty(text):
    return f'<div class="empty">{text}</div>'


def tiles(items, cls=""):
    """[(label, value, note)] -> metric tiles (3 or 4 in a row); `cls` adds a layout class."""
    cells = "".join(
        f"<div class='tile'><small>{esc(a)}</small><b>{esc(b)}</b><span>{esc(c)}</span></div>" for a, b, c in items
    )
    return f'<div class="tiles{" tiles-4" if len(items) == 4 else ""}{f" {cls}" if cls else ""}">{cells}</div>'


def facts(items):
    """[(label, value, note)] -> a compact row of study facts."""
    cells = "".join(f"<div><small>{esc(a)}</small><b>{esc(b)}</b><span>{esc(c)}</span></div>" for a, b, c in items)
    return f'<div class="facts">{cells}</div>'
