"""
JC Municipal Bond Sentiment — v3.0
- Prominent credit recommendation cards
- Evidence panel with sourced signals
- Annotated trend chart with risk flag markers
- Period trajectory indicator (improving/deteriorating/stable)
- Richer meeting-level detail
Run: streamlit run app.py
"""

import os
import io
import json
import warnings
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="JC Municipal Bond Sentiment",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

.block-container { padding-top: 3.5rem; padding-bottom: 5rem; max-width: 1200px; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }

.paper-title {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 34px; font-weight: 500; line-height: 1.2;
    color: var(--text-color); margin-bottom: 0.35rem; letter-spacing: -0.02em;
}
.paper-byline {
    font-family: 'DM Sans', sans-serif;
    font-size: 12.5px; color: var(--text-color); opacity: 0.45;
    margin-bottom: 1.5rem; letter-spacing: 0.02em;
}
.abstract-box {
    border-top: 1px solid rgba(128,128,128,0.2);
    border-bottom: 1px solid rgba(128,128,128,0.2);
    padding: 1.25rem 0; margin-bottom: 2rem;
}
.abstract-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 9.5px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; opacity: 0.38; color: var(--text-color); margin-bottom: 8px;
}
.abstract-text {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 15.5px; line-height: 1.85; color: var(--text-color); max-width: 920px;
}
.sec-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 9.5px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--text-color); opacity: 0.38;
    margin: 2.75rem 0 0.85rem; padding-bottom: 6px;
    border-bottom: 1px solid rgba(128,128,128,0.13);
}
.insight-box {
    border-left: 2px solid #1a4f82;
    padding: 0.6rem 1rem; margin: 0.75rem 0 1.25rem;
    background: rgba(26,79,130,0.04);
}
.insight-text {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 14.5px; line-height: 1.75; color: var(--text-color); opacity: 0.88;
}
.kpi-card {
    border: 0.75px solid rgba(128,128,128,0.18);
    border-radius: 2px; padding: 0.9rem 1rem;
    background: rgba(128,128,128,0.025); margin-bottom: 0.5rem;
}
.kpi-card-highlight {
    border: 0.75px solid rgba(26,79,130,0.35);
    border-radius: 2px; padding: 0.9rem 1rem;
    background: rgba(26,79,130,0.04); margin-bottom: 0.5rem;
}
.kpi-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 9px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; opacity: 0.4; color: var(--text-color); margin-bottom: 5px;
}
.kpi-value {
    font-family: 'DM Sans', sans-serif;
    font-size: 24px; font-weight: 500; line-height: 1.1; color: var(--text-color);
}
.kpi-value-lg {
    font-family: 'DM Sans', sans-serif;
    font-size: 28px; font-weight: 600; line-height: 1.1;
}
.kpi-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 11.5px; margin-top: 3px; opacity: 0.5; color: var(--text-color);
}
.kpi-delta {
    font-family: 'DM Mono', monospace;
    font-size: 10.5px; margin-top: 4px; letter-spacing: 0.02em;
}
.kpi-rationale {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 13px; line-height: 1.7; margin-top: 6px;
    color: var(--text-color); opacity: 0.75;
}

/* ── Recommendation cards ── */
.rec-overweight {
    border: 1.5px solid rgba(46,125,79,0.4);
    border-radius: 2px; padding: 1rem 1.1rem; margin-bottom: 0.5rem;
    background: rgba(46,125,79,0.05);
}
.rec-underweight {
    border: 1.5px solid rgba(185,64,64,0.4);
    border-radius: 2px; padding: 1rem 1.1rem; margin-bottom: 0.5rem;
    background: rgba(185,64,64,0.04);
}
.rec-marketweight {
    border: 1.5px solid rgba(26,79,130,0.3);
    border-radius: 2px; padding: 1rem 1.1rem; margin-bottom: 0.5rem;
    background: rgba(26,79,130,0.03);
}
.rec-monitor {
    border: 1.5px solid rgba(196,122,0,0.35);
    border-radius: 2px; padding: 1rem 1.1rem; margin-bottom: 0.5rem;
    background: rgba(196,122,0,0.03);
}

/* ── Evidence items ── */
.evidence-item {
    border-left: 2px solid rgba(128,128,128,0.2);
    padding: 0.45rem 0.85rem; margin-bottom: 0.5rem;
}
.evidence-item-neg { border-left-color: #b94040; }
.evidence-item-pos { border-left-color: #2e7d4f; }
.evidence-item-neut { border-left-color: #3d7ab5; }
.evidence-signal {
    font-family: 'DM Sans', sans-serif;
    font-size: 10.5px; font-weight: 600; letter-spacing: 0.03em;
    text-transform: uppercase; margin-bottom: 3px;
}
.evidence-detail {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 14px; line-height: 1.7; color: var(--text-color); opacity: 0.82;
}

.stress-card {
    border: 0.75px solid rgba(185,64,64,0.25);
    border-radius: 2px; padding: 0.85rem 1rem; margin-bottom: 0.5rem;
    background: rgba(185,64,64,0.02);
}
.stress-card-mild {
    border: 0.75px solid rgba(196,122,0,0.25);
    border-radius: 2px; padding: 0.85rem 1rem; margin-bottom: 0.5rem;
    background: rgba(196,122,0,0.02);
}
.pos  { color: #2e7d4f !important; }
.neg  { color: #b94040 !important; }
.neut { color: #1a4f82 !important; }
.warn { color: #c47a00 !important; }
.fig-caption {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 13.5px; line-height: 1.8; color: var(--text-color);
    opacity: 0.78; margin-top: 0.1rem; margin-bottom: 1.5rem; font-style: italic;
}
.fig-caption b { font-style: normal; font-weight: 600; color: var(--text-color); }
.explainer-body {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 15px; line-height: 1.85; color: var(--text-color); opacity: 0.84;
    margin-bottom: 0.75rem;
}
.appendix-term {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 15px; line-height: 1.88; color: var(--text-color); margin-bottom: 0.65rem;
}
.appendix-term b { font-weight: 600; font-style: normal; }
.paper-footer {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 12px; color: var(--text-color); opacity: 0.32;
    margin-top: 4rem; padding-top: 1rem;
    border-top: 1px solid rgba(128,128,128,0.12); line-height: 1.75;
}
label, .stSlider label, .stNumberInput label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 11px !important; font-weight: 500 !important;
    letter-spacing: 0.02em; opacity: 0.65;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# CHART CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

CHART_BG = "#F5F0E8"
FONT_CH  = dict(size=12, color="#1a1a1a", family="DM Sans, Arial, sans-serif")
LEGEND   = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(size=11, color="#1a1a1a"), bgcolor="rgba(0,0,0,0)")
BASE     = dict(plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG, font=FONT_CH,
                margin=dict(l=8, r=8, t=20, b=8), legend=LEGEND)

def layout(**overrides):
    d = dict(BASE); d.update(overrides); return d

BLUE   = "#1a4f82"; LBLUE  = "#3d7ab5"
GREEN  = "#2e7d4f"; LGREEN = "#6ab06a"
RED    = "#b94040"
GOLD   = "#c47a00"; GRAY   = "#888888"
PURPLE = "#5c3d82"

def ax(title="", grid=True, pct=False, suffix=""):
    d = dict(
        title=dict(text=title, font=dict(size=11, color="#555555")),
        tickfont=dict(size=11, color="#444444"),
        gridcolor="#e8e0d0" if grid else "rgba(0,0,0,0)",
        linecolor="#d4c9b8", linewidth=1, showline=True,
        showgrid=grid, zeroline=False, ticks="outside", ticklen=3,
    )
    if pct:    d["tickformat"] = ".1%"
    if suffix: d["ticksuffix"] = suffix
    return d

CAT_KEYS   = ["fiscal_stress", "pilot", "pension", "political_cohesion", "positive"]
CAT_LABELS = ["Fiscal Stress", "PILOT", "Pension", "Political Cohesion", "Positive"]
CAT_COLORS = [RED, GOLD, PURPLE, LBLUE, GREEN]

SIGNAL_CSS = {"Bullish": "pos", "Neutral": "warn", "Bearish": "neg"}

REC_CSS = {
    "Overweight":   "rec-overweight",
    "Market Weight":"rec-marketweight",
    "Underweight":  "rec-underweight",
    "Monitor":      "rec-monitor",
}
REC_COLOR = {
    "Overweight":   "#2e7d4f",
    "Market Weight":"#1a4f82",
    "Underweight":  "#b94040",
    "Monitor":      "#c47a00",
}
REC_ICON = {
    "Overweight":   "↑",
    "Market Weight":"→",
    "Underweight":  "↓",
    "Monitor":      "◎",
}

EVIDENCE_CSS = {"negative": "evidence-item-neg", "positive": "evidence-item-pos",
                "neutral": "evidence-item-neut"}
EVIDENCE_COLOR = {"negative": "#b94040", "positive": "#2e7d4f", "neutral": "#3d7ab5"}

# ══════════════════════════════════════════════════════════════════════════
# SEED DATA
# ══════════════════════════════════════════════════════════════════════════

SEED_PATH = os.path.join(os.path.dirname(__file__), "data", "seed_data.json")

@st.cache_data(show_spinner=False)
def load_seed_data(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.sort(key=lambda r: r.get("date", ""), reverse=True)
        return data
    except Exception as e:
        st.error(f"Could not load seed data: {e}")
        return []

SEED = load_seed_data(SEED_PATH)

# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def get_api_key() -> str:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY", "")


def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    raw  = uploaded_file.read()
    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(raw))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e:
            st.error(f"PDF read error: {e}"); return ""
    if name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            st.error(f"DOCX read error: {e}"); return ""
    if name.endswith((".html", ".htm")):
        try:
            from bs4 import BeautifulSoup
            return BeautifulSoup(raw, "html.parser").get_text(separator="\n")
        except Exception as e:
            st.error(f"HTML parse error: {e}"); return ""
    return raw.decode("utf-8", errors="ignore")


SYSTEM_PROMPT = """You are a senior municipal bond credit analyst with deep expertise in
New Jersey local government finance, writing for an institutional fixed-income audience.

Jersey City fiscal context:
- PILOT revenue ~30%+ of tax levy; expiration/challenge = material revenue risk
- Moody's A2. Watch: PFRS unfunded ~$1.2B, tax appeals ~$380M vs ~$22M reserve
- PFRS/PERS contribution deferral = direct credit-negative trigger
- Vote splits on fiscal items = governance execution risk

Return ONLY valid JSON, no markdown fences:
{
  "score": <float -1.0 to +1.0>,
  "signal": <"Bullish"|"Neutral"|"Bearish">,
  "credit_recommendation": <"Overweight"|"Market Weight"|"Underweight"|"Monitor">,
  "recommendation_rationale": <1-2 sentences on bond positioning>,
  "summary": <3-4 sentence analyst prose>,
  "credit_implications": <2-3 sentences on spread/rating/debt service implications>,
  "categories": {"fiscal_stress":int,"pilot":int,"pension":int,"political_cohesion":int,"positive":int},
  "evidence": [up to 5 objects: {"category":str,"signal":str,"detail":str,"direction":"negative"|"positive"|"neutral"}],
  "leading_indicators": [3 strings],
  "key_items": [4 strings],
  "risk_flags": [0-3 strings]
}
Rec calibration: Overweight=score>+0.20 no major flags; Underweight=score<-0.25 or 2+ flags; Monitor=mixed/near-zero with watch items; Market Weight=otherwise."""


def run_analysis(text: str, api_key: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user",
                   "content": f"Council minutes for analysis:\n\n{text[:60_000]}"}],
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    return json.loads(raw)


def compute_trajectory(seed: list[dict]) -> tuple[str, str, str]:
    """Return (label, css_class, description) for period credit trajectory."""
    if len(seed) < 3:
        return "Insufficient data", "neut", "Need at least 3 meetings to assess trajectory."
    scores = [r["score"] for r in sorted(seed, key=lambda x: x.get("date",""))]
    # Simple linear regression slope
    x = np.arange(len(scores))
    slope = np.polyfit(x, scores, 1)[0]
    recent_avg  = np.mean(scores[-3:])
    earlier_avg = np.mean(scores[:3])
    diff = recent_avg - earlier_avg
    if slope > 0.03 and diff > 0.05:
        return "Improving", "pos", f"Scores trending up; recent 3-meeting avg {recent_avg:+.2f} vs earlier {earlier_avg:+.2f}."
    elif slope < -0.03 and diff < -0.05:
        return "Deteriorating", "neg", f"Scores trending down; recent 3-meeting avg {recent_avg:+.2f} vs earlier {earlier_avg:+.2f}."
    else:
        return "Stable", "neut", f"No material directional trend; recent 3-meeting avg {recent_avg:+.2f}."


def rec_card(rec: str, rationale: str, score: float | None = None):
    """Render a prominent credit recommendation card."""
    css   = REC_CSS.get(rec, "rec-monitor")
    color = REC_COLOR.get(rec, GRAY)
    icon  = REC_ICON.get(rec, "◎")
    score_str = f"Score: {score:+.2f} · " if score is not None else ""
    st.markdown(f"""<div class="{css}">
<div class="kpi-label">Credit Recommendation</div>
<div class="kpi-value-lg" style="color:{color};">{icon} {rec}</div>
<div class="kpi-rationale">{score_str}{rationale}</div>
</div>""", unsafe_allow_html=True)


def render_evidence(evidence: list[dict]):
    """Render sourced signal evidence panel."""
    if not evidence:
        return
    st.markdown('<div class="sec-header" style="margin-top:1.25rem;">Signal evidence</div>',
                unsafe_allow_html=True)
    for ev in evidence:
        direction = ev.get("direction", "neutral")
        ev_css    = EVIDENCE_CSS.get(direction, "evidence-item-neut")
        ev_color  = EVIDENCE_COLOR.get(direction, GRAY)
        cat_label = ev.get("category", "").replace("_", " ").title()
        signal    = ev.get("signal", "")
        detail    = ev.get("detail", "")
        st.markdown(f"""<div class="evidence-item {ev_css}">
<div class="evidence-signal" style="color:{ev_color};">{cat_label} — {signal}</div>
<div class="evidence-detail">{detail}</div>
</div>""", unsafe_allow_html=True)


def render_meeting_detail(r: dict, fig_num: int | None = None):
    """Full meeting detail panel used inside expanders."""
    sig   = r["signal"]
    score = r["score"]
    rec   = r.get("credit_recommendation", "Monitor")
    cats  = r["categories"]

    # Abstract summary
    st.markdown(f"""
<div class="abstract-box" style="margin-top:0.5rem; margin-bottom:1rem;">
  <div class="abstract-label">Analyst summary — {r['date']}</div>
  <div class="abstract-text">{r['summary']}</div>
</div>""", unsafe_allow_html=True)

    # Recommendation + credit implications side by side
    top_c1, top_c2 = st.columns([1, 1])
    with top_c1:
        rec_card(rec, r.get("recommendation_rationale", ""), score)
    with top_c2:
        ci = r.get("credit_implications", "")
        if ci:
            st.markdown(f"""<div class="kpi-card-highlight">
<div class="kpi-label">Credit Implications</div>
<div class="kpi-rationale" style="font-size:14px; margin-top:4px;">{ci}</div>
</div>""", unsafe_allow_html=True)

    # Category signal counts
    st.markdown('<div class="sec-header" style="margin-top:1.25rem;">Signal counts by category</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    cat_map = [
        (c1, "fiscal_stress",      "Fiscal Stress",     "neg"),
        (c2, "pilot",              "PILOT",             "neut"),
        (c3, "pension",            "Pension",
         "neg" if cats.get("pension", 0) > 0 else "neut"),
        (c4, "political_cohesion", "Political Cohesion","neut"),
        (c5, "positive",           "Positive",          "pos"),
    ]
    for col, k, label_txt, col_cls in cat_map:
        with col:
            st.markdown(f"""<div class="kpi-card">
<div class="kpi-label">{label_txt}</div>
<div class="kpi-value {col_cls}">{cats.get(k, 0)}</div>
</div>""", unsafe_allow_html=True)

    # Evidence
    render_evidence(r.get("evidence", []))

    # Key items + leading indicators
    kc1, kc2 = st.columns(2)
    with kc1:
        if r.get("key_items"):
            st.markdown('<div class="sec-header" style="margin-top:1.25rem;">Key agenda items</div>',
                        unsafe_allow_html=True)
            for item in r["key_items"]:
                st.markdown(f'<div class="appendix-term">— {item}</div>',
                            unsafe_allow_html=True)
    with kc2:
        if r.get("leading_indicators"):
            st.markdown('<div class="sec-header" style="margin-top:1.25rem;">Leading indicators to watch</div>',
                        unsafe_allow_html=True)
            for li in r["leading_indicators"]:
                st.markdown(f'<div class="appendix-term">→ {li}</div>',
                            unsafe_allow_html=True)

    # Risk flags
    if r.get("risk_flags"):
        st.markdown('<div class="sec-header" style="margin-top:1.25rem;">Risk flags</div>',
                    unsafe_allow_html=True)
        for flag in r["risk_flags"]:
            st.markdown(f"""<div class="stress-card">
<div class="kpi-label">⚑ Credit negative — requires analyst review</div>
<div class="insight-text">{flag}</div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="paper-title">Jersey City Municipal Bond Sentiment</div>
<div class="paper-byline">
NLP-powered fiscal signal extraction · credit recommendation · bond market implications ·
signal evidence · leading indicator tracking — Jersey City, NJ municipal credit research
</div>
""", unsafe_allow_html=True)

n_seed = len(SEED)
date_range = ""
if n_seed > 0:
    dates = sorted(r.get("date", "") for r in SEED)
    date_range = f"{dates[0]} through {dates[-1]}"

st.markdown(f"""
<div class="abstract-box">
  <div class="abstract-label">Overview</div>
  <div class="abstract-text">
    This tool extracts structured bond-relevant signals from Jersey City Municipal Council
    meeting minutes and translates them into explicit credit recommendations — Overweight,
    Market Weight, Underweight, or Monitor — with supporting evidence sourced directly from
    the agenda text. Signals are classified across five categories (fiscal stress, PILOT/abatement
    activity, pension contribution status, political cohesion, and positive credit events) and
    scored on a continuous −1.0 to +1.0 scale. Each analysis includes a bond market implications
    statement connecting meeting findings to spread direction, rating trajectory, and debt service
    coverage. Jersey City-specific credit context is embedded: ~30%+ PILOT revenue dependency,
    Moody's A2, PFRS unfunded liability (~$1.2B), tax appeal exposure (~$380M vs ~$22M reserve).
    {"<strong>" + str(n_seed) + " pre-analyzed meetings</strong> (" + date_range + ") provide baseline context. " if n_seed > 0 else ""}Upload
    a new document in Section 2 to run a live analysis.
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — SIGNAL HISTORY
# ══════════════════════════════════════════════════════════════════════════

st.markdown(
    f'<div class="sec-header">1. Signal history'
    f'{" — " + str(n_seed) + " meetings (" + date_range + ")" if n_seed > 0 else ""}'
    f'</div>',
    unsafe_allow_html=True,
)

if not SEED:
    st.markdown("""<div class="stress-card-mild">
<div class="kpi-label">No seed data found</div>
<div class="insight-text">Run <code>python preprocess.py</code> to generate <code>data/seed_data.json</code>.</div>
</div>""", unsafe_allow_html=True)

else:
    # ── Trajectory ────────────────────────────────────────────────────────
    traj_label, traj_cls, traj_desc = compute_trajectory(SEED)

    avg_score   = sum(r["score"] for r in SEED) / n_seed
    latest      = SEED[0]
    n_bullish   = sum(1 for r in SEED if r["signal"] == "Bullish")
    n_bearish   = sum(1 for r in SEED if r["signal"] == "Bearish")
    n_neutral   = n_seed - n_bullish - n_bearish
    total_flags = sum(len(r.get("risk_flags", [])) for r in SEED)
    score_cls   = "pos" if avg_score > 0.05 else "neg" if avg_score < -0.05 else "warn"

    # Most common recommendation
    recs = [r.get("credit_recommendation", "Monitor") for r in SEED]
    rec_counts = {r: recs.count(r) for r in set(recs)}
    dominant_rec = max(rec_counts, key=rec_counts.get)

    # ── KPI row ───────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_data = [
        (k1, "Latest Signal",       latest["signal"],
         f"Score: {latest['score']:+.2f}",
         latest["date"],
         SIGNAL_CSS[latest["signal"]]),
        (k2, "Period Avg Score",    f"{avg_score:+.2f}",
         f"{n_seed} meetings",
         f"{n_bullish}B / {n_bearish}Br / {n_neutral}N",
         score_cls),
        (k3, "Credit Trajectory",   traj_label,
         traj_desc,
         date_range,
         traj_cls),
        (k4, "Dominant Rec.",       dominant_rec,
         f"Most frequent over period",
         f"{rec_counts[dominant_rec]}/{n_seed} meetings",
         REC_COLOR.get(dominant_rec, GRAY).replace("#", "").join(["color:#", " !important"]
         ).replace("color:", "") if False else
         ("pos" if dominant_rec == "Overweight"
          else "neg" if dominant_rec == "Underweight"
          else "warn" if dominant_rec == "Monitor"
          else "neut")),
        (k5, "Risk Flags Raised",   str(total_flags),
         "Across all sessions",
         "Require analyst review",
         "neg" if total_flags > 0 else "pos"),
    ]
    for col, label, value, sub, delta, cls in kpi_data:
        with col:
            st.markdown(f"""<div class="kpi-card">
<div class="kpi-label">{label}</div>
<div class="kpi-value {cls}">{value}</div>
<div class="kpi-sub">{sub}</div>
<div class="kpi-delta {cls}">{delta}</div>
</div>""", unsafe_allow_html=True)

    # ── Trend chart — annotated ───────────────────────────────────────────
    trend_df = pd.DataFrame([
        {"Date": r["date"], "Score": r["score"], "Signal": r["signal"],
         "Rec": r.get("credit_recommendation", "Monitor"),
         "Flags": len(r.get("risk_flags", [])),
         "Summary": r.get("summary", "")[:120] + "…",
         "File": r.get("filename", "")}
        for r in SEED
    ]).sort_values("Date")

    bar_colors = [GREEN if s == "Bullish" else RED if s == "Bearish" else GOLD
                  for s in trend_df["Signal"]]

    hover_texts = [
        f"<b>{row['Date']}</b><br>"
        f"Signal: {row['Signal']} ({row['Score']:+.2f})<br>"
        f"Rec: {row['Rec']}<br>"
        f"Flags: {row['Flags']}<br>"
        f"<i>{row['Summary']}</i>"
        for _, row in trend_df.iterrows()
    ]

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=trend_df["Date"], y=trend_df["Score"],
        marker_color=bar_colors, opacity=0.75,
        text=[f"{s:+.2f}" for s in trend_df["Score"]],
        textposition="outside",
        textfont=dict(size=11, color="#333"),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
    ))

    # Annotate risk flag meetings with a marker
    flag_rows = trend_df[trend_df["Flags"] > 0]
    if not flag_rows.empty:
        fig_trend.add_trace(go.Scatter(
            x=flag_rows["Date"],
            y=flag_rows["Score"] + (0.07 * np.sign(flag_rows["Score"].values + 0.001)),
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=10, color=RED, opacity=0.9),
            text=[f"⚑{int(f)}" for f in flag_rows["Flags"]],
            textposition="top center",
            textfont=dict(size=9, color=RED),
            hovertemplate="Risk flags: %{text}<extra></extra>",
            showlegend=False,
        ))

    # Trend line
    if len(trend_df) >= 3:
        scores_arr = trend_df["Score"].values
        x_arr      = np.arange(len(scores_arr))
        slope, intercept = np.polyfit(x_arr, scores_arr, 1)
        trend_y = slope * x_arr + intercept
        trend_color = GREEN if slope > 0.01 else RED if slope < -0.01 else GRAY
        fig_trend.add_trace(go.Scatter(
            x=trend_df["Date"], y=trend_y,
            mode="lines",
            line=dict(color=trend_color, width=1.5, dash="dot"),
            name=f"Trend ({slope:+.3f}/meeting)",
            hoverinfo="skip",
        ))

    fig_trend.add_hline(y=0, line_dash="dot", line_color="#d4c9b8", line_width=1)
    fig_trend.add_hline(y=avg_score, line_dash="dash", line_color=GRAY, line_width=1.2,
                        annotation_text=f"Period avg ({avg_score:+.2f})",
                        annotation_font=dict(size=9, color=GRAY),
                        annotation_position="top right")
    # Threshold bands
    fig_trend.add_hrect(y0=0.15, y1=1.1, fillcolor=GREEN, opacity=0.03, line_width=0)
    fig_trend.add_hrect(y0=-1.1, y1=-0.25, fillcolor=RED, opacity=0.03, line_width=0)

    fig_trend.update_layout(
        **layout(height=300, margin=dict(l=8, r=100, t=20, b=8)),
        xaxis=dict(**ax("Meeting date")),
        yaxis=dict(**ax("Sentiment score"), range=[-1.1, 1.1]),
        showlegend=True,
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown(f"""<div class="fig-caption">
<b>Figure 1.</b> Net sentiment score per meeting. Green = Bullish; red = Bearish; amber = Neutral.
Red triangles (⚑) mark meetings with risk flags; hover for full detail.
Dotted line shows linear trend ({("improving" if traj_label == "Improving" else "deteriorating" if traj_label == "Deteriorating" else "stable")}).
Light green band = Overweight zone (score > +0.15); light red band = Underweight zone (score < −0.25).
</div>""", unsafe_allow_html=True)

    # ── Aggregate category chart ──────────────────────────────────────────
    st.markdown('<div class="sec-header" style="margin-top:1.5rem;">Aggregate signal counts by category</div>',
                unsafe_allow_html=True)
    agg_cats = {k: sum(r["categories"].get(k, 0) for r in SEED) for k in CAT_KEYS}

    fig_cats = go.Figure(go.Bar(
        x=CAT_LABELS,
        y=[agg_cats[k] for k in CAT_KEYS],
        marker_color=CAT_COLORS, opacity=0.78,
        text=[str(agg_cats[k]) for k in CAT_KEYS],
        textposition="outside",
        textfont=dict(size=12, color="#333"),
        hovertemplate="%{x}: %{y} signals<extra></extra>",
    ))
    fig_cats.update_layout(
        **layout(height=280, margin=dict(l=8, r=8, t=20, b=8)),
        xaxis=dict(**ax("Signal category")),
        yaxis=dict(**ax("Total signal count"), dtick=2),
        showlegend=False,
    )
    st.plotly_chart(fig_cats, use_container_width=True)
    st.markdown(f"""<div class="fig-caption">
<b>Figure 2.</b> Aggregate signal counts across all {n_seed} meetings.
Fiscal Stress and Pension = direct credit-negative. PILOT requires directional interpretation.
Political Cohesion = governance execution risk.
</div>""", unsafe_allow_html=True)

    # ── Recommendation distribution ───────────────────────────────────────
    st.markdown('<div class="sec-header" style="margin-top:1.5rem;">Credit recommendation distribution</div>',
                unsafe_allow_html=True)

    rec_order  = ["Overweight", "Market Weight", "Monitor", "Underweight"]
    rec_vals   = [rec_counts.get(r, 0) for r in rec_order]
    rec_colors = [REC_COLOR[r] for r in rec_order]

    fig_rec = go.Figure(go.Bar(
        x=rec_order, y=rec_vals,
        marker_color=rec_colors, opacity=0.75,
        text=[str(v) for v in rec_vals],
        textposition="outside",
        textfont=dict(size=13, color="#333"),
        hovertemplate="%{x}: %{y} meetings<extra></extra>",
    ))
    fig_rec.update_layout(
        **layout(height=220, margin=dict(l=8, r=8, t=20, b=8)),
        xaxis=dict(**ax("")),
        yaxis=dict(**ax("Meetings"), dtick=1),
        showlegend=False,
    )
    st.plotly_chart(fig_rec, use_container_width=True)
    st.markdown(f"""<div class="fig-caption">
<b>Figure 3.</b> Distribution of credit recommendations across all {n_seed} analyzed meetings.
Dominant recommendation over the period: <b>{dominant_rec}</b> ({rec_counts[dominant_rec]}/{n_seed} meetings).
</div>""", unsafe_allow_html=True)

    # ── Per-meeting expandable detail ─────────────────────────────────────
    st.markdown('<div class="sec-header" style="margin-top:1.5rem;">Meeting-level detail</div>',
                unsafe_allow_html=True)

    for r in SEED:
        rec   = r.get("credit_recommendation", "Monitor")
        flags = len(r.get("risk_flags", []))
        flag_str = f"  ·  ⚑ {flags} flag(s)" if flags else ""
        with st.expander(
            f"{r['date']}  ·  {r['signal']}  ({r['score']:+.2f})  ·  {rec}{flag_str}  —  {r.get('filename','')}"
        ):
            render_meeting_detail(r)

    # ── Tabular summary ───────────────────────────────────────────────────
    st.markdown('<div class="sec-header" style="margin-top:1.5rem;">Tabular summary</div>',
                unsafe_allow_html=True)

    summary_rows = []
    for r in SEED:
        cats = r["categories"]
        summary_rows.append({
            "Date":          r["date"],
            "Signal":        r["signal"],
            "Score":         f"{r['score']:+.2f}",
            "Recommendation":r.get("credit_recommendation", "—"),
            "Fiscal Stress": cats.get("fiscal_stress", 0),
            "PILOT":         cats.get("pilot", 0),
            "Pension":       cats.get("pension", 0),
            "Pol. Cohesion": cats.get("political_cohesion", 0),
            "Positive":      cats.get("positive", 0),
            "Risk Flags":    len(r.get("risk_flags", [])),
            "File":          r.get("filename", ""),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.markdown(f"""<div class="fig-caption">
<b>Table 1.</b> Per-meeting summary for all {n_seed} analyzed meetings.
Recommendation column reflects model's explicit bond positioning guidance per session.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — UPLOAD & ANALYZE
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="sec-header">2. Upload and analyze a new council meeting</div>',
            unsafe_allow_html=True)

st.markdown("""<div class="explainer-body">
Upload a Jersey City council meeting minutes document to run a live analysis. The model
returns a credit recommendation, bond market implications statement, sourced signal evidence,
leading indicators, and risk flags — all grounded in Jersey City-specific fiscal context.
Accepted formats: PDF (text-layer), DOCX, TXT, HTML.
</div>""", unsafe_allow_html=True)

api_key = get_api_key()
inp_c1, inp_c2 = st.columns([2, 1])
with inp_c1:
    uploaded = st.file_uploader("Council minutes document",
                                type=["pdf", "docx", "txt", "html", "htm"])
with inp_c2:
    if not api_key:
        api_key = st.text_input("Anthropic API Key", type="password",
                                help="Or set ANTHROPIC_API_KEY in .streamlit/secrets.toml")
    else:
        st.markdown("""<div class="kpi-card">
<div class="kpi-label">API Key</div>
<div class="kpi-value pos" style="font-size:16px;">✓ Configured</div>
<div class="kpi-sub">From secrets / environment</div>
</div>""", unsafe_allow_html=True)

if uploaded:
    if not api_key:
        st.markdown("""<div class="stress-card-mild">
<div class="kpi-label">API key required</div>
<div class="insight-text">Enter an Anthropic API key above to run analysis.</div>
</div>""", unsafe_allow_html=True)
    else:
        with st.spinner("Extracting text…"):
            text = extract_text(uploaded)
        if not text.strip():
            st.error("Could not extract text. Try a text-based PDF or a different format.")
        else:
            word_count = len(text.split())
            st.markdown(f"""<div class="insight-box">
<div class="insight-text">Extracted <strong>{word_count:,} words</strong> from <em>{uploaded.name}</em>.</div>
</div>""", unsafe_allow_html=True)

            if st.button("▶ Run analysis", type="primary"):
                with st.spinner("Analyzing with Claude…"):
                    try:
                        result = run_analysis(text, api_key)
                    except json.JSONDecodeError as e:
                        st.error(f"JSON parse error: {e}"); result = None
                    except Exception as e:
                        st.error(f"Analysis failed: {e}"); result = None

                if result:
                    result["date"]     = uploaded.name
                    result["filename"] = uploaded.name
                    st.markdown(f"""
<div class="abstract-box" style="margin-top:1.5rem;">
  <div class="abstract-label">Analysis result — {uploaded.name}</div>
  <div class="abstract-text">{result.get('summary','')}</div>
</div>""", unsafe_allow_html=True)
                    render_meeting_detail(result)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — JC CREDIT CONTEXT
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="sec-header">3. Jersey City credit context</div>',
            unsafe_allow_html=True)

cx_c1, cx_c2 = st.columns(2)
context_l = [
    ("Moody's Rating", "A2",
     "Current issuer rating. Watch items: pension trajectory, tax appeal reserve adequacy."),
    ("PILOT Revenue Dependency", "~30%+ of tax levy",
     "Concentration and expiration risk. Journal Square submarket is the primary concentration."),
    ("PFRS Unfunded Liability", "~$1.2B",
     "Contribution deferrals are a direct credit-negative trigger. Full ADEC compliance is baseline."),
]
context_r = [
    ("Tax Appeal Exposure", "~$380M pending",
     "Pending appeals vs. ~$22M reserve — ~6% coverage. Settlement overruns erode fiscal flexibility."),
    ("Peer Universe", "NJ Investment-Grade GO",
     "Comparison universe: NJ GO issuers rated A2/A or better."),
    ("Recommendation Thresholds", "±0.15 / ±0.25",
     "Monitor: score near 0 with watch items. Market Weight: −0.15 to +0.20. "
     "Overweight: >+0.20, no major flags. Underweight: <−0.25 or 2+ flags."),
]
for col, items in [(cx_c1, context_l), (cx_c2, context_r)]:
    with col:
        for label, value, desc in items:
            st.markdown(f"""<div class="kpi-card" style="margin-bottom:0.75rem;">
<div class="kpi-label">{label}</div>
<div class="kpi-value neut" style="font-size:18px;">{value}</div>
<div class="kpi-sub" style="font-size:11px; line-height:1.6; margin-top:6px;">{desc}</div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="sec-header">4. Methodology</div>', unsafe_allow_html=True)

with st.expander("Signal extraction, scoring, and recommendation methodology"):
    st.markdown("""
<div class="appendix-term"><b>Sentiment score.</b>
Continuous value on [−1.0, +1.0]. Weights signals by materiality — a reserve drawdown
scores more negatively than a routine housekeeping resolution. Not a keyword count.</div>

<div class="appendix-term"><b>Credit recommendation.</b>
Explicit bond positioning guidance derived from the score and risk flag combination:
Overweight (score > +0.20, no material flags), Market Weight (−0.15 to +0.20),
Monitor (near-zero with specific watch items), Underweight (score < −0.25 or 2+ flags).</div>

<div class="appendix-term"><b>Credit implications.</b>
A 2–3 sentence statement connecting meeting findings to bond market consequences —
spread direction, rating trajectory, debt service coverage, or liquidity. Written for
an institutional fixed-income audience.</div>

<div class="appendix-term"><b>Signal evidence.</b>
Up to 5 sourced signal items per meeting, each citing the specific agenda item or
discussion that generated the signal. Directional classification: negative, positive, neutral.</div>

<div class="appendix-term"><b>Credit trajectory.</b>
Linear regression slope across all meeting scores. Improving = slope > 0.03 and recent
3-meeting avg exceeds earlier avg by >0.05. Deteriorating = slope < −0.03 and similar
threshold. Stable = no material directional trend.</div>

<div class="appendix-term"><b>Risk flags.</b>
Discrete credit-negative items warranting analyst escalation — distinct from category
counts. Named events (e.g., settlement exceeding reserve by a material amount) that
a credit analyst would place on formal watch.</div>
""", unsafe_allow_html=True)

with st.expander("How to obtain council minutes and update seed data"):
    st.markdown("""
<div class="appendix-term"><b>CivicWeb Portal (free, HTML).</b>
Navigate to <em>cityofjerseycity.civicweb.net/portal/</em> → Meetings → Regular Meeting
of Municipal Council → select date → Minutes. Save page as HTML (Ctrl+S) and upload.</div>

<div class="appendix-term"><b>OPRA Request (PDF).</b>
Contact the City Clerk at (201) 547-5150. Text-layer PDFs work well.</div>

<div class="appendix-term"><b>Adding new meetings to seed data.</b>
Place new PDFs in <code>data/pdfs/</code>, then run:
<code>python preprocess.py --resume</code>
The <code>--resume</code> flag skips already-processed filenames and appends new results.
Commit the updated <code>data/seed_data.json</code> to redeploy.</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="paper-footer">
JC Municipal Bond Sentiment · v3.0 · Manohar Analytics · NLP-powered fiscal signal extraction ·
For research and informational purposes only. Not investment advice. Municipal bond trading
involves risk; conduct independent due diligence. Credit recommendations are model outputs
and should not be the sole basis for investment decisions. · Powered by Anthropic Claude.
</div>
""", unsafe_allow_html=True)
