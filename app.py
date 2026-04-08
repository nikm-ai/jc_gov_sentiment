"""
JC Municipal Bond Sentiment — v2.0
Style: matches portfolio-risk-dashboard editorial aesthetic
Run: streamlit run app.py
"""

import os
import io
import json
import math
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="JC Municipal Bond Sentiment",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════
# CSS — editorial, data-ink aesthetic (mirrors portfolio-risk-dashboard)
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

.block-container { padding-top: 3.5rem; padding-bottom: 5rem; max-width: 1200px; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }

/* ── Typography system ── */
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

/* ── KPI cards ── */
.kpi-card {
    border: 0.75px solid rgba(128,128,128,0.18);
    border-radius: 2px; padding: 0.9rem 1rem;
    background: rgba(128,128,128,0.025); margin-bottom: 0.5rem;
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
.kpi-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 11.5px; margin-top: 3px; opacity: 0.5; color: var(--text-color);
}
.kpi-delta {
    font-family: 'DM Mono', monospace;
    font-size: 10.5px; margin-top: 4px; letter-spacing: 0.02em;
}

/* ── Stress / warning card variants ── */
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

/* ── Signal badge ── */
.signal-badge {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 10px; font-weight: 600; letter-spacing: 0.09em;
    text-transform: uppercase; padding: 2px 9px; border-radius: 2px; margin-right: 6px;
}
.signal-bullish { background: rgba(46,125,79,0.12); color: #2e7d4f; }
.signal-bearish { background: rgba(185,64,64,0.12); color: #b94040; }
.signal-neutral  { background: rgba(196,122,0,0.12);  color: #c47a00; }

/* ── Colour helpers ── */
.pos  { color: #2e7d4f !important; }
.neg  { color: #b94040 !important; }
.neut { color: #1a4f82 !important; }
.warn { color: #c47a00 !important; }
.mono { font-family: 'DM Mono', monospace !important; font-size: 13px !important; }

/* ── Captions ── */
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
# CHART CONSTANTS (identical to portfolio-risk-dashboard)
# ══════════════════════════════════════════════════════════════════════════

CHART_BG = "#F5F0E8"
FONT_CH   = dict(size=12, color="#1a1a1a", family="DM Sans, Arial, sans-serif")
LEGEND    = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                 font=dict(size=11, color="#1a1a1a"), bgcolor="rgba(0,0,0,0)")
BASE      = dict(plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG, font=FONT_CH,
                 margin=dict(l=8, r=8, t=20, b=8), legend=LEGEND)

def layout(**overrides):
    d = dict(BASE); d.update(overrides); return d

BLUE   = "#1a4f82"; LBLUE  = "#3d7ab5"; LLBLUE = "#b8cfe0"
GREEN  = "#2e7d4f"; LGREEN = "#6ab06a"; LLGREEN = "#c8e6c9"
RED    = "#b94040"; LRED   = "#c47a7a"; LLRED   = "#f5c6c6"
GOLD   = "#c47a00"; LGOLD  = "#e8a020"
GRAY   = "#888888"; LGRAY  = "#cccccc"
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
New Jersey local government finance.

Jersey City fiscal context:
- PILOT (Payment in Lieu of Taxes) revenue constitutes approximately 30%+ of total tax levy;
  expiration or court challenge of major agreements represents meaningful revenue risk.
- Moody's current rating: A2. Key watch items: PFRS pension unfunded liability (~$1.2B),
  tax appeal reserve adequacy (~$380M pending vs ~$22M reserve), and abatement pipeline.
- PFRS and PERS contributions must remain current; any deferral triggers negative watch.
- Political cohesion (vote splits on fiscal items) signals execution risk on budget plans.

Your task: analyze the provided council meeting minutes and extract structured bond-relevant
signals. Return ONLY valid JSON — no markdown fences, no preamble — with this exact shape:

{
  "score": <float, -1.0 to +1.0, where -1 is severely bearish and +1 is strongly bullish>,
  "signal": <"Bullish" | "Neutral" | "Bearish">,
  "summary": <string, 3-4 sentences in precise analyst prose — what happened, what it means
               for credit, and what to monitor>,
  "categories": {
    "fiscal_stress": <int, count of distinct stress signals detected>,
    "pilot":         <int, count of PILOT/abatement-related signals>,
    "pension":       <int, count of pension/PFRS/PERS signals>,
    "political_cohesion": <int, count of governance/vote signals>,
    "positive":      <int, count of positive credit signals>
  },
  "leading_indicators": [<3 concise strings — forward-looking items to monitor>],
  "key_items": [<4 concise strings — most notable agenda items>],
  "risk_flags": [<0–3 strings — specific credit-negative items requiring attention, empty list if none>]
}

Score calibration: 0.0 = no net signal; ±0.1-0.3 = mild; ±0.3-0.6 = moderate; ±0.6+ = strong."""


def run_analysis(text: str, api_key: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Council minutes for analysis:\n\n{text[:60_000]}"}],
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    return json.loads(raw)

# ══════════════════════════════════════════════════════════════════════════
# SEED DATA — 3 pre-analyzed meetings (Jul–Dec 2025)
# ══════════════════════════════════════════════════════════════════════════

SEED = [
    {
        "date": "2025-12-10", "score": 0.05, "signal": "Neutral",
        "categories": {"fiscal_stress": 1, "pilot": 4, "pension": 0,
                        "political_cohesion": 4, "positive": 4},
        "summary": (
            "End-of-year session was predominantly housekeeping, with capital carry-forward "
            "resolutions and a Journal Square Transit Village Phase II PILOT introduced on first "
            "reading. Pension contributions were confirmed on schedule — a key watch item given "
            "PFRS's $1.2B unfunded liability. No material fiscal stress signals were detected; "
            "the PILOT pipeline continues to expand, though concentration risk merits monitoring."
        ),
        "leading_indicators": [
            "Journal Square PILOT concentration: 4 active agreements in single submarket",
            "Tax appeal reserve adequacy vs. ~$380M pending docket",
            "2026 PFRS contribution schedule — any deferral triggers negative watch",
        ],
        "key_items": [
            "Journal Square Phase II PILOT — first reading",
            "FY2025 capital carry-forward resolutions ($12.4M)",
            "Pension contribution confirmation (PFRS on schedule)",
            "IT infrastructure budget amendment — $1.1M",
        ],
        "risk_flags": [],
    },
    {
        "date": "2025-11-19", "score": -0.18, "signal": "Bearish",
        "categories": {"fiscal_stress": 3, "pilot": 2, "pension": 1,
                        "political_cohesion": 2, "positive": 2},
        "summary": (
            "A contested budget amendment for emergency infrastructure repair drew two dissenting "
            "votes — the most significant governance fracture observed in recent sessions. More "
            "consequentially, a tax appeal settlement was approved at $4.2M above the reserve "
            "estimate, surfacing potential reserve adequacy risk given the $380M pending docket. "
            "One PILOT renewal (Liberty Harbor) was advanced, partially offsetting stress signals, "
            "but the combination of split votes on fiscal items and reserve pressure warrants "
            "a cautious near-term credit view."
        ),
        "leading_indicators": [
            "Tax appeal reserve vs. estimated $380M pipeline — CFO update due Q1 2026",
            "Split 7-2 vote pattern on fiscal items: governance execution risk",
            "Emergency capital spending: potential for unbudgeted appropriations",
        ],
        "key_items": [
            "Tax appeal settlement — $4.2M above reserve estimate",
            "Emergency road repair budget amendment ($3.8M)",
            "7-2 split vote on reserve transfer authorization",
            "Liberty Harbor PILOT renewal — approved",
        ],
        "risk_flags": [
            "Tax appeal settlement exceeded reserve by $4.2M — reserve adequacy at risk",
            "Split council vote signals reduced fiscal consensus",
        ],
    },
    {
        "date": "2025-10-15", "score": 0.24, "signal": "Bullish",
        "categories": {"fiscal_stress": 0, "pilot": 3, "pension": 0,
                        "political_cohesion": 5, "positive": 6},
        "summary": (
            "A uniformly constructive session: unanimous passage of a new mixed-use development "
            "PILOT at McGinley Square, a $2.1M NJ DOT transit infrastructure grant award, and "
            "formal certification of the FY2025 budget surplus. No fiscal stress signals were "
            "detected. The strength of political cohesion (no dissents on any fiscal item) and "
            "the active new-development pipeline suggest improved near-term revenue visibility "
            "and positive momentum for the credit."
        ),
        "leading_indicators": [
            "McGinley Square PILOT buildout timeline and abatement revenue ramp",
            "NJ DOT grant disbursement schedule and compliance requirements",
            "Affordable housing bond issuance size and debt service impact",
        ],
        "key_items": [
            "McGinley Square mixed-use PILOT — unanimously approved",
            "$2.1M NJ DOT transit infrastructure grant",
            "FY2025 budget surplus certification — unanimous",
            "Affordable housing bond resolution — first reading",
        ],
        "risk_flags": [],
    },
]

CAT_KEYS   = ["fiscal_stress", "pilot", "pension", "political_cohesion", "positive"]
CAT_LABELS = ["Fiscal Stress", "PILOT", "Pension", "Political Cohesion", "Positive"]
CAT_COLORS = [RED, GOLD, PURPLE, LBLUE, GREEN]

SIGNAL_CSS = {"Bullish": "signal-bullish", "Neutral": "signal-neutral", "Bearish": "signal-bearish"}
SCORE_CLS  = {"Bullish": "pos", "Neutral": "warn", "Bearish": "neg"}

# ══════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="paper-title">Jersey City Municipal Bond Sentiment</div>
<div class="paper-byline">
NLP-powered fiscal signal extraction from public council records ·
signal categorization · credit risk indicators · leading indicator tracking —
Jersey City, NJ municipal credit research
</div>
""", unsafe_allow_html=True)

# ── Overview abstract ────────────────────────────────────────────────────
st.markdown("""
<div class="abstract-box">
  <div class="abstract-label">Overview</div>
  <div class="abstract-text">
    This tool ingests Jersey City Municipal Council meeting minutes — uploaded as PDF, DOCX,
    HTML, or plain text — and uses Claude to extract structured bond-relevant signals across
    five categories: fiscal stress, PILOT/abatement activity, pension contribution status,
    political cohesion, and positive credit events. Each document is scored on a continuous
    scale from −1.0 (strongly bearish) to +1.0 (strongly bullish), with an accompanying
    plain-language analyst summary and forward-looking indicator set. Jersey City-specific
    credit context is embedded in the model: ~30%+ PILOT revenue dependency, Moody's A2
    rating, PFRS unfunded liability (~$1.2B), and pending tax appeal exposure (~$380M).
    Three pre-analyzed seed meetings from the second half of 2025 are included as baseline
    reference. Upload a new document in Section 2 to run a live analysis.
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — SIGNAL HISTORY (seed data)
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="sec-header">1. Signal history — pre-loaded meetings (Jul–Dec 2025)</div>',
            unsafe_allow_html=True)

st.markdown("""<div class="explainer-body">
The three meetings below represent analyzed seed data covering the second half of 2025.
They establish a baseline sentiment trajectory and demonstrate the signal extraction methodology.
Scores are continuous (not bucketed) and reflect the net balance of credit-positive and
credit-negative signals within each session. A score of 0.0 implies no net directional signal,
not an absence of material content.
</div>""", unsafe_allow_html=True)

# ── Summary KPI row ───────────────────────────────────────────────────────
avg_score  = sum(r["score"] for r in SEED) / len(SEED)
latest     = SEED[0]
n_bullish  = sum(1 for r in SEED if r["signal"] == "Bullish")
n_bearish  = sum(1 for r in SEED if r["signal"] == "Bearish")
total_flags = sum(len(r["risk_flags"]) for r in SEED)

score_cls = "pos" if avg_score > 0.05 else "neg" if avg_score < -0.05 else "warn"
latest_cls = SCORE_CLS[latest["signal"]]

kpi_col = st.columns(4)
kpi_data = [
    ("Latest signal",      latest["signal"],   f"Score: {latest['score']:+.2f}",
     latest["date"],       latest_cls),
    ("Avg sentiment score", f"{avg_score:+.2f}", f"{len(SEED)} meetings analyzed",
     f"{n_bullish}B / {n_bearish}Br / {len(SEED)-n_bullish-n_bearish}N",  score_cls),
    ("Signal range",        f"{min(r['score'] for r in SEED):+.2f} → {max(r['score'] for r in SEED):+.2f}",
     "Min → max over period", "Jul–Dec 2025", "neut"),
    ("Risk flags raised",   str(total_flags),   "Across all sessions",
     "Require analyst review", "neg" if total_flags > 0 else "pos"),
]
for col, (label, value, sub, delta, cls) in zip(kpi_col, kpi_data):
    with col:
        st.markdown(f"""<div class="kpi-card">
<div class="kpi-label">{label}</div>
<div class="kpi-value {cls}">{value}</div>
<div class="kpi-sub">{sub}</div>
<div class="kpi-delta {cls}">{delta}</div>
</div>""", unsafe_allow_html=True)

# ── Score trend chart ─────────────────────────────────────────────────────
trend_df = pd.DataFrame([
    {"Date": r["date"], "Score": r["score"], "Signal": r["signal"]} for r in SEED
]).sort_values("Date")

bar_colors = [GREEN if s == "Bullish" else RED if s == "Bearish" else GOLD
              for s in trend_df["Signal"]]

fig_trend = go.Figure()
fig_trend.add_trace(go.Bar(
    x=trend_df["Date"], y=trend_df["Score"],
    marker_color=bar_colors, opacity=0.75,
    text=[f"{s:+.2f}" for s in trend_df["Score"]],
    textposition="outside",
    textfont=dict(size=11, color="#333"),
    hovertemplate="Date: %{x}<br>Score: %{y:+.2f}<extra></extra>",
))
fig_trend.add_hline(y=0, line_dash="dot", line_color="#d4c9b8", line_width=1)
fig_trend.add_hline(y=avg_score, line_dash="dash", line_color=GRAY, line_width=1.2,
                    annotation_text=f"Period avg ({avg_score:+.2f})",
                    annotation_font=dict(size=9, color=GRAY),
                    annotation_position="top right")
fig_trend.update_layout(
    **layout(height=260, margin=dict(l=8, r=80, t=20, b=8)),
    xaxis=dict(**ax("Meeting date")),
    yaxis=dict(**ax("Sentiment score"), range=[-0.55, 0.55]),
    showlegend=False,
)
st.plotly_chart(fig_trend, use_container_width=True)
st.markdown("""<div class="fig-caption">
<b>Figure 1.</b> Net sentiment score per meeting session. Green bars indicate a net bullish
reading; red bars a net bearish reading; amber indicates neutral. Dashed line marks the
period average. Scores reflect the model's assessment of the balance of credit-positive
vs. credit-negative signals detected in the minutes text.
</div>""", unsafe_allow_html=True)

# ── Aggregate category signal chart ──────────────────────────────────────
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
    **layout(height=300, margin=dict(l=8, r=8, t=20, b=8)),
    xaxis=dict(**ax("Signal category")),
    yaxis=dict(**ax("Total signal count"), dtick=1),
    showlegend=False,
)
st.plotly_chart(fig_cats, use_container_width=True)
st.markdown("""<div class="fig-caption">
<b>Figure 2.</b> Aggregate signal counts across all three pre-loaded meetings.
Fiscal Stress and Pension signals are direct credit-negative indicators.
PILOT signals require directional interpretation (new agreements = positive;
expirations/challenges = negative). Political Cohesion captures governance risk.
Positive signals offset stress signals in the net score calculation.
</div>""", unsafe_allow_html=True)

# ── Per-meeting expandable detail ─────────────────────────────────────────
st.markdown('<div class="sec-header" style="margin-top:1.5rem;">Meeting-level detail</div>',
            unsafe_allow_html=True)

for i, r in enumerate(SEED):
    sig_badge = f'<span class="signal-badge {SIGNAL_CSS[r["signal"]]}">{r["signal"]}</span>'
    with st.expander(f"{r['date']}  ·  {r['signal']}  ·  Score {r['score']:+.2f}"):
        st.markdown(f"""
<div class="abstract-box" style="margin-top:0.5rem; margin-bottom:1rem;">
  <div class="abstract-label">Signal summary — {r['date']}</div>
  <div class="abstract-text">{r['summary']}</div>
</div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        cats = r["categories"]
        for col, k, label, col_cls in [
            (c1, "fiscal_stress",   "Fiscal Stress Signals",    "neg"),
            (c1, "pension",         "Pension Signals",           "neg" if cats.get("pension",0) > 0 else "neut"),
            (c2, "pilot",           "PILOT Signals",             "neut"),
            (c2, "political_cohesion", "Political Cohesion",     "neut"),
            (c3, "positive",        "Positive Signals",          "pos"),
        ]:
            with col:
                st.markdown(f"""<div class="kpi-card">
<div class="kpi-label">{label}</div>
<div class="kpi-value {col_cls}">{cats.get(k,0)}</div>
</div>""", unsafe_allow_html=True)

        kc1, kc2 = st.columns(2)
        with kc1:
            if r.get("key_items"):
                st.markdown('<div class="sec-header" style="margin-top:1rem;">Key agenda items</div>',
                            unsafe_allow_html=True)
                for item in r["key_items"]:
                    st.markdown(f'<div class="appendix-term">— {item}</div>', unsafe_allow_html=True)
        with kc2:
            if r.get("leading_indicators"):
                st.markdown('<div class="sec-header" style="margin-top:1rem;">Leading indicators to watch</div>',
                            unsafe_allow_html=True)
                for li in r["leading_indicators"]:
                    st.markdown(f'<div class="appendix-term">→ {li}</div>', unsafe_allow_html=True)

        if r.get("risk_flags"):
            st.markdown('<div class="sec-header" style="margin-top:1rem;">Risk flags</div>',
                        unsafe_allow_html=True)
            for flag in r["risk_flags"]:
                st.markdown(f"""<div class="stress-card">
<div class="kpi-label" style="margin-bottom:4px;">⚑ Credit Negative</div>
<div class="insight-text">{flag}</div>
</div>""", unsafe_allow_html=True)

# ── Summary table ─────────────────────────────────────────────────────────
st.markdown('<div class="sec-header" style="margin-top:1.5rem;">Signal history — tabular summary</div>',
            unsafe_allow_html=True)

summary_rows = []
for r in SEED:
    cats = r["categories"]
    summary_rows.append({
        "Date": r["date"],
        "Signal": r["signal"],
        "Score": f"{r['score']:+.2f}",
        "Fiscal Stress": cats.get("fiscal_stress", 0),
        "PILOT": cats.get("pilot", 0),
        "Pension": cats.get("pension", 0),
        "Pol. Cohesion": cats.get("political_cohesion", 0),
        "Positive": cats.get("positive", 0),
        "Risk Flags": len(r.get("risk_flags", [])),
    })

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
st.markdown("""<div class="fig-caption">
<b>Table 1.</b> Per-meeting signal counts across all five categories.
Risk Flags column counts distinct credit-negative items flagged for analyst review.
Scores are continuous; the categorical signal label is assigned at thresholds of ±0.05.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — UPLOAD & ANALYZE
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="sec-header">2. Upload and analyze a new council meeting</div>',
            unsafe_allow_html=True)

st.markdown("""<div class="explainer-body">
Upload a Jersey City council meeting minutes document to run a live analysis. The model
is prompted with full Jersey City credit context and returns a structured JSON result
containing a continuous sentiment score, five-category signal counts, a 3–4 sentence
analyst summary, leading indicators, and explicit risk flags. Accepted formats: PDF (text-layer),
DOCX, TXT, and saved HTML (from the CivicWeb portal). For best results, use text-based PDFs
rather than scanned images.
</div>""", unsafe_allow_html=True)

api_key = get_api_key()

inp_c1, inp_c2 = st.columns([2, 1])
with inp_c1:
    uploaded = st.file_uploader(
        "Council minutes document",
        type=["pdf", "docx", "txt", "html", "htm"],
        help="PDF (text layer), DOCX, TXT, or HTML from CivicWeb portal",
    )
with inp_c2:
    if not api_key:
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Or set ANTHROPIC_API_KEY in .streamlit/secrets.toml",
        )
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
        with st.spinner("Extracting text from document…"):
            text = extract_text(uploaded)

        if not text.strip():
            st.error("Could not extract text. Try a different format or a text-based PDF.")
        else:
            word_count = len(text.split())
            char_count = len(text)
            st.markdown(f"""<div class="insight-box">
<div class="insight-text">
Extracted <strong>{word_count:,} words</strong> ({char_count:,} characters) from
<em>{uploaded.name}</em>. Text will be truncated to 60,000 characters if needed.
</div>
</div>""", unsafe_allow_html=True)

            if st.button("▶ Run analysis", type="primary"):
                with st.spinner("Analyzing with Claude — extracting bond-relevant signals…"):
                    try:
                        result = run_analysis(text, api_key)
                    except json.JSONDecodeError as e:
                        st.error(f"Model response could not be parsed as JSON: {e}")
                        result = None
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        result = None

                if result:
                    sig   = result["signal"]
                    score = result["score"]
                    cats  = result["categories"]
                    score_color = GREEN if score > 0.05 else RED if score < -0.05 else GOLD

                    st.markdown(f"""
<div class="abstract-box" style="margin-top:1.5rem;">
  <div class="abstract-label">Analysis result — {uploaded.name}</div>
  <div class="abstract-text">{result['summary']}</div>
</div>""", unsafe_allow_html=True)

                    # ── Result KPI row ────────────────────────────────────
                    r_cols = st.columns(4)
                    r_kpis = [
                        ("Sentiment signal",  sig,
                         f"Score: {score:+.2f}",
                         f"Range: −1.0 (bearish) → +1.0 (bullish)",
                         SCORE_CLS[sig]),
                        ("Fiscal stress signals", str(cats.get("fiscal_stress", 0)),
                         "Direct credit-negative indicators",
                         "Shortfalls, reserve drawdowns, BANs",
                         "neg" if cats.get("fiscal_stress", 0) > 1 else "neut"),
                        ("PILOT signals",     str(cats.get("pilot", 0)),
                         "Abatement activity",
                         "New agreements / expirations / risks",
                         "neut"),
                        ("Positive signals",  str(cats.get("positive", 0)),
                         "Credit-constructive indicators",
                         "Grants, surplus, rating events",
                         "pos" if cats.get("positive", 0) > 0 else "neut"),
                    ]
                    for col, (label, value, sub, delta, cls) in zip(r_cols, r_kpis):
                        with col:
                            st.markdown(f"""<div class="kpi-card">
<div class="kpi-label">{label}</div>
<div class="kpi-value {cls}">{value}</div>
<div class="kpi-sub">{sub}</div>
<div class="kpi-delta {cls}">{delta}</div>
</div>""", unsafe_allow_html=True)

                    # ── Category bar chart ────────────────────────────────
                    r_counts = [cats.get(k, 0) for k in CAT_KEYS]
                    fig_r = go.Figure(go.Bar(
                        x=CAT_LABELS, y=r_counts,
                        marker_color=CAT_COLORS, opacity=0.78,
                        text=[str(v) for v in r_counts],
                        textposition="outside",
                        textfont=dict(size=12, color="#333"),
                        hovertemplate="%{x}: %{y}<extra></extra>",
                    ))
                    fig_r.update_layout(
                        **layout(height=280, margin=dict(l=8, r=8, t=20, b=8)),
                        xaxis=dict(**ax("Signal category")),
                        yaxis=dict(**ax("Signal count"), dtick=1),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_r, use_container_width=True)
                    st.markdown(f"""<div class="fig-caption">
<b>Figure 3.</b> Signal category counts for {uploaded.name}.
Net sentiment score of {score:+.2f} ({sig}) reflects the balance of positive
vs. stress signals weighted by category significance.
</div>""", unsafe_allow_html=True)

                    # ── Indicators and risk flags ─────────────────────────
                    det_c1, det_c2 = st.columns(2)
                    with det_c1:
                        if result.get("key_items"):
                            st.markdown('<div class="sec-header">Key agenda items</div>',
                                        unsafe_allow_html=True)
                            for item in result["key_items"]:
                                st.markdown(f'<div class="appendix-term">— {item}</div>',
                                            unsafe_allow_html=True)
                        if result.get("leading_indicators"):
                            st.markdown('<div class="sec-header" style="margin-top:1.5rem;">Leading indicators to watch</div>',
                                        unsafe_allow_html=True)
                            for li in result["leading_indicators"]:
                                st.markdown(f'<div class="appendix-term">→ {li}</div>',
                                            unsafe_allow_html=True)
                    with det_c2:
                        if result.get("risk_flags"):
                            st.markdown('<div class="sec-header">Risk flags</div>',
                                        unsafe_allow_html=True)
                            for flag in result["risk_flags"]:
                                st.markdown(f"""<div class="stress-card">
<div class="kpi-label">⚑ Credit negative</div>
<div class="insight-text">{flag}</div>
</div>""", unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="sec-header">Risk flags</div>',
                                        unsafe_allow_html=True)
                            st.markdown("""<div class="kpi-card">
<div class="kpi-label">Status</div>
<div class="kpi-value pos" style="font-size:16px;">No flags raised</div>
<div class="kpi-sub">No distinct credit-negative items detected</div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — JC CREDIT CONTEXT
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="sec-header">3. Jersey City credit context</div>',
            unsafe_allow_html=True)

st.markdown("""<div class="explainer-body">
The model is primed with the following Jersey City-specific structural credit context,
which informs how signals are weighted and summarized. This context is not inferred
from meeting content — it is treated as standing background knowledge for the analyst.
</div>""", unsafe_allow_html=True)

cx_c1, cx_c2 = st.columns(2)

context_items_l = [
    ("Moody's Rating", "A2",
     "Current issuer rating. Watch items: pension trajectory, tax appeal reserve adequacy."),
    ("PILOT Revenue Dependency", "~30%+ of tax levy",
     "Concentration and expiration risk. Any court challenge to major PILOT agreements "
     "represents material near-term revenue risk. Journal Square submarket is the primary concentration."),
    ("PFRS Unfunded Liability", "~$1.2B",
     "Police and Fire Retirement System. Contribution deferrals are a direct credit-negative trigger. "
     "Full ADEC compliance is the baseline positive assumption embedded in the model."),
]
context_items_r = [
    ("Tax Appeal Exposure", "~$380M pending",
     "Pending tax appeals vs. ~$22M reserve — a reserve-to-exposure ratio of approximately 6%. "
     "Settlement outcomes above reserve estimates erode fiscal flexibility. CFO monitoring of "
     "reserve adequacy is a key leading indicator."),
    ("Benchmark", "S&P 500 Muni GO Index",
     "Peer comparison universe: NJ investment-grade GO issuers rated A2/A or better."),
    ("Signal Threshold", "±0.05",
     "Scores within ±0.05 of zero are classified Neutral. Scores outside ±0.30 are classified "
     "as materially directional. Scores outside ±0.60 represent strong credit signals."),
]

for col, items in [(cx_c1, context_items_l), (cx_c2, context_items_r)]:
    with col:
        for label, value, desc in items:
            st.markdown(f"""<div class="kpi-card" style="margin-bottom:0.75rem;">
<div class="kpi-label">{label}</div>
<div class="kpi-value neut" style="font-size:18px;">{value}</div>
<div class="kpi-sub" style="font-size:11px; line-height:1.6; margin-top:6px;">{desc}</div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — METHODOLOGY APPENDIX
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="sec-header">4. Methodology</div>', unsafe_allow_html=True)

with st.expander("Signal extraction and scoring methodology"):
    st.markdown("""
<div class="appendix-term"><b>Sentiment score.</b>
A continuous value on [−1.0, +1.0] produced by Claude after reviewing the full text of
council minutes. The model is given Jersey City-specific credit context and instructed to
weight signals by materiality: a budget amendment that draws down reserves scores more
negatively than a routine housekeeping resolution. The score is not a keyword count — it
reflects qualitative assessment of fiscal implications.</div>

<div class="appendix-term"><b>Fiscal Stress signals.</b>
Counted instances of: revenue shortfall disclosures, unbudgeted appropriations, reserve
drawdown authorizations, Bond Anticipation Note (BAN) issuances, emergency declarations
with fiscal implications, or negative audit findings referenced in discussion.</div>

<div class="appendix-term"><b>PILOT / Abatement signals.</b>
Counted references to Payment in Lieu of Taxes agreements — new agreements (positive),
renewals (neutral-to-positive), expirations or non-renewals (negative), court challenges
(negative), and Phase introductions on first reading (forward-looking positive).</div>

<div class="appendix-term"><b>Pension signals.</b>
Counted references to PFRS/PERS contribution status, unfunded liability updates,
actuarial assumption changes, or contribution deferrals. Confirmation of on-schedule
contributions is logged as a positive signal; any deferral discussion is a strong negative.</div>

<div class="appendix-term"><b>Political Cohesion signals.</b>
Counted instances of split votes (negative), unanimous passage of fiscal resolutions
(positive), contested budget items (negative), or council member dissents on revenue
or spending matters. High cohesion implies reduced execution risk on fiscal plans.</div>

<div class="appendix-term"><b>Positive signals.</b>
Counted credit-constructive events: new development approvals, state or federal grant
awards, surplus certifications, rating upgrades or affirmations, debt payoff confirmations,
and reserve increases.</div>

<div class="appendix-term"><b>Risk flags.</b>
Discrete credit-negative items that warrant explicit analyst attention — distinct from
aggregate category counts. A flag is raised when the model identifies a specific,
named event (e.g., a settlement exceeding reserve estimates by a material amount) that
a credit analyst would escalate to a watch item.</div>

<div class="appendix-term"><b>Leading indicators.</b>
Forward-looking items identified within the minutes that have potential credit implications
at future sessions — items to monitor in subsequent meetings, not current-period signals.</div>
""", unsafe_allow_html=True)

with st.expander("Data sources and how to obtain council minutes"):
    st.markdown("""
<div class="appendix-term"><b>CivicWeb Portal (free, HTML).</b>
Navigate to <em>cityofjerseycity.civicweb.net/portal/</em> → Meetings → Regular Meeting of
Municipal Council → select date → Minutes. Save page as HTML (Ctrl+S) and upload here.
This is the fastest path and generally produces clean text extraction.</div>

<div class="appendix-term"><b>OPRA Request (PDF).</b>
Contact the City Clerk at (201) 547-5150 to request minutes for a specific date range
under the Open Public Records Act. PDF files with text layers work well; scanned PDFs
require OCR pre-processing before upload.</div>

<div class="appendix-term"><b>City website (PDF).</b>
Recent meeting agendas and minutes are sometimes posted directly at
<em>jerseycitynj.gov</em> under the Council section, though coverage is less consistent
than the CivicWeb portal.</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="paper-footer">
JC Municipal Bond Sentiment · v2.0 · Manohar Analytics · NLP-powered fiscal signal extraction ·
For research and informational purposes only. Not investment advice. Municipal bond trading
involves risk; conduct independent due diligence. Signal scores represent model assessments
and should not be the sole basis for investment decisions. · Powered by Anthropic Claude.
</div>
""", unsafe_allow_html=True)
