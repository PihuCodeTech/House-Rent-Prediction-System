"""Look and feel of the Kiraya app: design tokens, page CSS, rupee formatting, icons and the HTML pieces
(result card, notes bar, comparison bars, problems, metric tiles)."""

import base64
import html
import re

T = {  # design tokens
    "bg": "#F7F4EF",
    "card": "#FFFFFF",
    "line": "#E9E3DA",
    "field": "#F2EDE6",
    "ink": "#221E1A",
    "ink2": "#5C544C",  # muted text; ≥ 6:1 contrast on every surface
    "rust": "#B5472A",
    "rust_dark": "#9C3B21",
    "rust_soft": "#FBEEE9",
    "green": "#3F7D5A",
    "note": "#F1ECE4",
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
}


def icon(name, size=20, color=None):
    """A line icon as an <img> (inline <svg> is removed by Streamlit's HTML sanitiser)."""
    color = color or T["rust"]
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
SERIF = "'Times New Roman', Times, 'Liberation Serif', serif"  # headings and running text
SANS = (
    "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"  # UI + numbers
)

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown {{ font-family: {SERIF}; }}
/* sans-serif for controls, labels, numbers and tables */
button, input, textarea, select, [data-baseweb], [data-testid="stWidgetLabel"] p, .fl, .kicker, .num, .res-rent,
.chip, .tile b, .tile small, .bars-val, .status, .lb-row, .aff-big, .facts b, .field-note {{ font-family: {SANS}; }}
.num, .res-rent, .tile b, .bars-val, .lb-val, .aff-big {{ font-variant-numeric: tabular-nums lining-nums; }}
.stApp {{ background: {T["bg"]}; color: {T["ink"]}; }}
.block-container {{ padding-top: 5.2rem; padding-bottom: 3rem; max-width: 1240px; }}
div[data-testid="stElementContainer"]:has(div.sr-only):not(:has(p)) {{ position: absolute; width: 1px; height: 1px; overflow: hidden; }}
.sr-only {{ position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0);
  white-space: nowrap; border: 0; }}

/* top bar */
header[data-testid="stHeader"] {{ background: {T["card"]}; border-bottom: 1px solid {T["line"]}; height: 4rem; }}
img[data-testid="stHeaderLogo"] {{ height: 2.2rem; max-width: 22rem; }}
a[data-testid="stTopNavLink"] {{ background: transparent !important; border-radius: 0; padding: .5rem .2rem; margin: 0 .7rem;
  border-bottom: 2px solid transparent; }}
a[data-testid="stTopNavLink"] p {{ font-family: {SERIF}; font-weight: 600; font-size: .98rem; }}
a[data-testid="stTopNavLink"][aria-current="page"] {{ border-bottom-color: {T["rust"]}; }}
a[data-testid="stTopNavLink"][aria-current="page"] p, a[data-testid="stTopNavLink"][aria-current="page"] [data-testid="stIconMaterial"] {{ color: {T["rust"]} !important; }}
.ico {{ display: inline-block; vertical-align: middle; flex: 0 0 auto; }}

/* cards: any container with a key starting with card */
div[class*="st-key-card"] {{
  background: {T["card"]}; border: 1px solid {T["line"]}; border-radius: 16px;
  padding: 26px 28px; box-shadow: 0 1px 2px rgba(34,30,26,.04), 0 10px 28px rgba(34,30,26,.05);
  transition: opacity .2s ease, filter .2s ease;
}}
/* an estimate that no longer matches the form: faded until it is recalculated */
div[class*="st-key-card"][class*="_old"] {{ opacity: .45; filter: grayscale(.7); }}
.card-title {{ display: flex; align-items: center; gap: .55rem; font-size: 1.15rem; font-weight: 600; color: {T["ink"]}; margin: 0; }}
.card-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem;
  border-bottom: 1px solid {T["line"]}; padding-bottom: .9rem; margin-bottom: .2rem; }}
.card-sub {{ color: {T["ink2"]}; font-size: .92rem; margin: .3rem 0 0; line-height: 1.45; }}
.card-aside {{ color: {T["ink2"]}; font-size: .85rem; white-space: nowrap; font-family: {SANS}; }}
.card-aside b {{ color: {T["rust"]}; }}

/* rows of cards line up: columns already stretch to the tallest one; the last card in each column fills the rest */
div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] > div[data-testid="stLayoutWrapper"]:last-child:has(> div[class*="st-key-card"]) {{
  flex: 1 1 auto; display: flex; flex-direction: column; }}
div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] > div[data-testid="stLayoutWrapper"]:last-child > div[class*="st-key-card"] {{
  flex: 1 1 auto; }}

/* hover / focus info tips */
.tip {{ position: relative; display: inline-flex; align-items: center; justify-content: center; width: 15px; height: 15px;
  margin-left: .35rem; border-radius: 50%; border: 1.3px solid {T["ink2"]}; color: {T["ink2"]}; font-size: 10px; font-weight: 700;
  font-style: italic; font-family: Georgia, serif; cursor: help; vertical-align: 1px; line-height: 1; }}
.fl .tip {{ font-size: 10px; font-weight: 700; color: {T["ink2"]}; }}
.tip:hover, .tip:focus, .fl .tip:hover, .fl .tip:focus {{ border-color: {T["rust"]}; color: {T["rust"]}; outline: none; }}
.tip .tip-text, .fl .tip .tip-text {{ visibility: hidden; opacity: 0; position: absolute; left: 50%; bottom: calc(100% + 8px); transform: translateX(-50%);
  width: 240px; background: {T["ink"]}; color: #FFFFFF; font: 400 .8rem/1.45 {SANS}; font-style: normal; text-align: left;
  border-radius: 8px; padding: .55rem .7rem; box-shadow: 0 8px 20px rgba(0,0,0,.18); z-index: 1000; transition: opacity .12s ease; }}
.tip .tip-text::after {{ content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px; border: 5px solid transparent;
  border-top-color: {T["ink"]}; }}
.tip:hover .tip-text, .tip:focus .tip-text, .fl .tip:hover .tip-text, .fl .tip:focus .tip-text {{ visibility: visible; opacity: 1; }}

/* field labels with hints on the right */
.fl {{ display: flex; justify-content: space-between; align-items: baseline; gap: .6rem; margin: .35rem 0 -.55rem; }}
.fl span:first-child {{ font-weight: 600; font-size: .86rem; color: {T["ink"]}; }}
.fl .req {{ color: {T["rust"]}; margin-left: .15rem; }}
.fl .hint {{ font-size: .76rem; color: {T["ink2"]}; text-align: right; }}
.field-note {{ font-size: .78rem; color: {T["ink2"]}; line-height: 1.45; margin: -.35rem 0 .2rem; }}
.field-note b {{ color: {T["ink"]}; font-weight: 600; }}
[data-testid="stWidgetLabel"] p {{ font-size: .86rem !important; font-weight: 600; }}

/* inputs */
div[data-baseweb="select"] > div, div[data-testid="stNumberInputContainer"], div[data-baseweb="input"] {{
  background: {T["field"]} !important; border-color: {T["line"]} !important; border-radius: 10px !important;
}}
div[data-testid="stNumberInputContainer"] input, div[data-baseweb="select"] input, input::placeholder {{
  font-family: {SANS} !important; font-weight: 500; }}
input::placeholder {{ font-weight: 400; color: {T["ink2"]} !important; opacity: .8; }}
div[data-baseweb="select"] > div {{ min-height: 46px; }}
div[data-testid="stNumberInput"] button {{ background: transparent; }}

/* listing-details expander as an inset card */
div[data-testid="stExpander"] details {{ background: {T["note"]}; border: 1px solid {T["line"]}; border-radius: 12px; }}
div[data-testid="stExpander"] summary p {{ font-weight: 600; }}

/* buttons */
button[data-testid="stBaseButton-primary"] {{
  background: {T["rust"]}; border-color: {T["rust"]}; border-radius: 10px; min-height: 52px; font-weight: 600;
  box-shadow: 0 6px 16px rgba(181,71,42,.22);
}}
button[data-testid="stBaseButton-primary"]:hover {{ background: {T["rust_dark"]}; border-color: {T["rust_dark"]}; }}
button[data-testid="stBaseButton-primary"] p {{ font-size: 1rem; font-weight: 600; }}
button[data-testid="stBaseButton-secondary"] {{ border-radius: 10px; min-height: 52px; border-color: {T["line"]}; background: {T["card"]}; }}
button[data-testid="stBaseButton-secondary"]:hover {{ border-color: {T["rust"]}; color: {T["rust"]}; }}
button[data-testid="stBaseButton-tertiary"] p {{ font-size: .85rem; color: {T["ink2"]}; }}

/* result card */
.kicker {{ font-size: .74rem; font-weight: 600; letter-spacing: .09em; text-transform: uppercase; color: {T["ink2"]}; }}
.res-top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; flex-wrap: wrap; }}
.res-rent {{ font-weight: 700; font-size: 3.2rem; line-height: 1.1; letter-spacing: -.03em; color: {T["ink"]}; margin-top: .2rem; }}
.res-rent span {{ font-size: 1.15rem; font-weight: 500; color: {T["ink2"]}; letter-spacing: 0; }}
.res-sub {{ color: {T["ink2"]}; font-size: .95rem; margin-top: .35rem; }}
.status {{ display: inline-flex; align-items: center; gap: .4rem; font-size: .78rem; font-weight: 600; border-radius: 999px;
  padding: .3rem .7rem; white-space: nowrap; border: 1px solid; }}
.status.ok {{ color: {T["green"]}; background: #EEF5F0; border-color: #CFE3D6; }}
.status.old {{ color: {T["rust_dark"]}; background: {T["rust_soft"]}; border-color: #F1D3C8; }}
.status-when {{ display: block; text-align: right; font-family: {SANS}; font-size: .74rem; color: {T["ink2"]}; margin-top: .35rem; }}
.res-tiles {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .8rem; margin-top: 1.1rem;
  border-top: 1px solid {T["line"]}; padding-top: 1.1rem; }}
.res-tiles .tile b {{ font-size: 1.35rem; }}
.res-tiles .tile.hl {{ background: {T["rust_soft"]}; border-color: #F1D3C8; }}
.res-tiles .tile.hl b {{ color: {T["rust_dark"]}; }}
.res-mini .res-rent {{ font-size: 2.5rem; }}
.res-mini .res-tiles {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}

/* bars comparing rent per sq ft */
.bars-row {{ display: grid; grid-template-columns: 12rem 1fr 4.5rem; gap: .7rem; align-items: center; margin: .9rem 0; font-size: .92rem; }}
.bars-track {{ background: {T["field"]}; border-radius: 7px; height: 14px; }}
.bars-fill {{ height: 14px; border-radius: 7px; background: #CBBFB1; }}
.bars-row.me {{ font-weight: 600; }}
.bars-row.me .bars-fill {{ background: {T["rust"]}; }}
.bars-val {{ text-align: right; }}

/* leaderboard */
.lb-row {{ display: grid; grid-template-columns: 1.6rem 15rem 1fr 9.5rem 8.5rem; gap: .8rem; align-items: center; padding: .42rem 0;
  border-bottom: 1px solid {T["line"]}; font-size: .86rem; }}
.lb-row:last-child {{ border-bottom: none; }}
.lb-rank {{ color: {T["ink2"]}; text-align: right; }}
.lb-track {{ background: {T["field"]}; border-radius: 6px; height: 10px; position: relative; }}
.lb-fill {{ height: 10px; border-radius: 6px; background: #CBBFB1; }}
.lb-err {{ position: absolute; top: -3px; height: 16px; border-left: 1.5px solid {T["ink2"]}; border-right: 1.5px solid {T["ink2"]}; }}
.lb-val {{ text-align: right; }}
.lb-val small {{ color: {T["ink2"]}; }}
.lb-row.win {{ font-weight: 600; }}
.lb-row.win .lb-fill {{ background: {T["rust"]}; }}
.lb-row.base {{ color: {T["ink2"]}; }}
.lb-row.base .lb-fill {{ background: repeating-linear-gradient(135deg, #D9D1C6 0 6px, #E9E3DA 6px 12px); }}
.lb-head {{ font-size: .7rem; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; color: {T["ink2"]}; padding-top: 0; }}
.tag.grey {{ color: {T["ink2"]}; background: {T["note"]}; border-color: {T["line"]}; }}
.tag {{ display: inline-block; font-size: .68rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; color: {T["rust_dark"]};
  background: {T["rust_soft"]}; border: 1px solid #F1D3C8; border-radius: 6px; padding: .05rem .35rem; margin-left: .35rem; vertical-align: 1px; }}

/* affordability */
.aff-big {{ font-size: 2rem; font-weight: 700; line-height: 1.1; }}
.aff-big small {{ font-size: .95rem; font-weight: 500; color: {T["ink2"]}; }}
.aff-track {{ position: relative; height: 14px; border-radius: 7px; background: {T["field"]}; margin: .9rem 0 .3rem; overflow: hidden; }}
.aff-fill {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 7px; }}
.aff-mark {{ position: absolute; top: -2px; bottom: -2px; border-left: 2px dashed {T["ink2"]}; }}
.aff-scale {{ position: relative; height: 1.1rem; font-family: {SANS}; font-size: .72rem; color: {T["ink2"]}; }}
.aff-scale span {{ position: absolute; transform: translateX(-50%); }}
.verdict-line {{ display: flex; gap: .5rem; align-items: baseline; font-weight: 600; margin: .6rem 0 .2rem; }}

/* notes, banners, problems */
.notes {{ background: {T["note"]}; border: 1px solid {T["line"]}; border-radius: 12px; padding: .8rem 1.1rem; }}
.notes p {{ display: flex; gap: .65rem; margin: .25rem 0; font-size: .92rem; color: {T["ink"]}; line-height: 1.45; }}
.notes p::before {{ content: ""; flex: 0 0 7px; height: 7px; margin-top: .45em; border-radius: 50%; background: {T["rust"]}; }}
.banner {{ background: {T["rust_soft"]}; border: 1px solid #F1D3C8; color: {T["ink"]}; border-radius: 12px; padding: .7rem 1rem; font-size: .92rem; }}
.problems {{ background: #FCEDEA; border: 1px solid #F0C9C0; border-radius: 12px; padding: .9rem 1.1rem; }}
.problems p {{ margin: .25rem 0; font-size: .93rem; }}
.empty {{ color: {T["ink2"]}; background: {T["note"]}; border: 1px dashed {T["line"]}; border-radius: 12px; padding: 1rem 1.1rem;
  font-size: .9rem; line-height: 1.5; }}

/* compare + about */
.diff-big {{ font-size: 1.35rem; font-weight: 700; color: {T["ink"]}; }}
.tiles {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin: 1rem 0 .4rem; }}
.tiles-4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
.tile {{ border: 1px solid {T["line"]}; border-radius: 12px; padding: .85rem 1rem; background: #FCFBF9; min-width: 0; }}
.tile small {{ display: block; color: {T["ink2"]}; font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }}
.tile b {{ display: block; font-size: 1.5rem; font-weight: 700; margin: .3rem 0 .15rem; letter-spacing: -.01em; }}
.tile span {{ display: block; color: {T["ink2"]}; font-size: .84rem; line-height: 1.4; }}
.facts {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .8rem; margin: 1.1rem 0 0; }}
.facts div {{ border-left: 2px solid {T["line"]}; padding-left: .7rem; }}
.facts small {{ display: block; font-family: {SANS}; font-size: .7rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: {T["ink2"]}; }}
.facts b {{ display: block; font-size: .98rem; margin-top: .2rem; }}
.facts span {{ display: block; font-size: .84rem; color: {T["ink2"]}; line-height: 1.35; margin-top: .1rem; }}
.verdict {{ background: {T["rust_soft"]}; border: 1px solid #F1D3C8; border-radius: 12px; padding: .8rem 1rem;
  margin: 1.1rem 0 0; font-size: .97rem; line-height: 1.5; color: {T["ink"]}; }}
.muted {{ color: {T["ink2"]}; font-size: .9rem; line-height: 1.55; }}
.lead {{ font-size: .96rem; line-height: 1.55; color: {T["ink"]}; }}

/* focus: no outline box inside text fields; the whole field gets a rust ring instead */
input, input:focus, input:focus-visible, textarea:focus, textarea:focus-visible {{ outline: none !important; box-shadow: none !important; }}
div[data-baseweb="select"] > div:focus-within, div[data-testid="stNumberInputContainer"]:focus-within {{
  border-color: {T["rust"]} !important; box-shadow: 0 0 0 3px #F3D6CC !important;
}}
div[data-testid="stNumberInputContainer"] div[data-baseweb="input"],
div[data-testid="stNumberInputContainer"] div[data-baseweb="input"]:focus-within {{ border: none !important; box-shadow: none !important; }}
/* fixed lists (city, size measured as) are picked, not typed: no blinking text cursor */
div[class*="st-key-"][class*="city"] input, div[class*="st-key-"][class*="_area"] input {{ caret-color: transparent; cursor: pointer; }}
button:focus-visible, a:focus-visible, summary:focus-visible {{ outline: 2px solid {T["rust"]} !important; outline-offset: 2px; }}

@media (max-width: 900px) {{
  .facts, .tiles-4 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
@media (max-width: 700px) {{
  .res-rent {{ font-size: 2.5rem; }}
  .res-tiles, .res-mini .res-tiles {{ grid-template-columns: 1fr; }}
  .bars-row {{ grid-template-columns: 7.5rem 1fr 3.8rem; }}
  .lb-row {{ grid-template-columns: 1.2rem 1fr 5.4rem 3.8rem; gap: .5rem; font-size: .8rem; }}
  .lb-track, .lb-barh {{ display: none; }}
  .lb-head {{ font-size: .62rem; }}
  .lb-val small {{ display: none; }}
  .tiles {{ grid-template-columns: 1fr; }}
  .facts {{ grid-template-columns: 1fr; }}
  div[class*="st-key-card"] {{ padding: 18px 16px; }}
}}
</style>
"""


# ----------------------------- pieces -----------------------------
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
        pill = '<span class="status old">⚠ Outdated — details changed</span>'
        sub = f"last estimated {esc(when)}" if when else ""
    else:
        pill = '<span class="status ok">✓ Up to date</span>'
        sub = f"estimated {esc(when)}" if when else ""
    return f'<div>{pill}<span class="status-when">{sub}</span></div>'


def result_card(result, when=None, stale=False, compact=False):
    """The main result: monthly rent, the likely range, rent per sq ft and (full card) the three seed models'
    agreement. `when` is the time it was calculated; `stale` marks it as no longer matching the form."""
    listing = result["listing"]
    cells = [
        ("hl", "Likely range", f"{inr_short(result['low'])} – {inr_short(result['high'])}"),
        ("", "Rent per sq ft", inr(result["rent_per_sqft"])),
    ]
    if not compact:
        cells.append(("", "Model agreement", f"{inr_short(result['seed_low'])} – {inr_short(result['seed_high'])}"))
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
    color = {"ok": T["green"], "stretched": "#9A6A12", "heavy": T["rust_dark"]}[b["level"]]
    mark = {"ok": "✓", "stretched": "!", "heavy": "✕"}[b["level"]]
    t, t2 = b["target"] * 100, min(b["target"] * 100 + 10, 100)
    width = min(b["share"], 1.0) * 100
    return f"""
<div class="aff-big">{b["share"]:.0%} <small>of take-home income goes to housing</small></div>
<div class="verdict-line" style="color:{color}"><span aria-hidden="true">{mark}</span><span>{esc(b["verdict"])}</span></div>
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


def tiles(items):
    """[(label, value, note)] -> metric tiles (3 or 4 in a row)."""
    cells = "".join(
        f"<div class='tile'><small>{esc(a)}</small><b>{esc(b)}</b><span>{esc(c)}</span></div>" for a, b, c in items
    )
    return f'<div class="tiles{" tiles-4" if len(items) == 4 else ""}">{cells}</div>'


def facts(items):
    """[(label, value, note)] -> a compact row of study facts."""
    cells = "".join(f"<div><small>{esc(a)}</small><b>{esc(b)}</b><span>{esc(c)}</span></div>" for a, b, c in items)
    return f'<div class="facts">{cells}</div>'
