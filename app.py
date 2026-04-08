"""
JC Municipal Bond Sentiment — simplified single-page app
Run: streamlit run app.py
"""

import os
import sys
import io
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JC Municipal Bond Sentiment",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .block-container { max-width: 860px; padding-top: 2rem; }
  .metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
  h1 { font-size: 1.7rem !important; }
  .stAlert p { font-size: 0.9rem; }
  .signal-pill {
      display: inline-block; padding: 2px 10px;
      border-radius: 12px; font-size: 0.78rem; font-weight: 600;
  }
  .pill-bullish  { background:#d1fae5; color:#065f46; }
  .pill-neutral  { background:#fef3c7; color:#92400e; }
  .pill-bearish  { background:#fee2e2; color:#991b1b; }
</style>
""", unsafe_allow_html=True)

# ── helpers ────────────────────────────────────────────────────────────────────
def get_api_key() -> str:
    """Return API key from st.secrets or environment, or empty string."""
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY", "")


def extract_text(uploaded_file) -> str:
    """Extract plain text from PDF, DOCX, TXT, or HTML uploads."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(raw))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e:
            st.error(f"PDF read error: {e}")
            return ""

    if name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            st.error(f"DOCX read error: {e}")
            return ""

    if name.endswith(".html") or name.endswith(".htm"):
        try:
            from bs4 import BeautifulSoup
            return BeautifulSoup(raw, "html.parser").get_text(separator="\n")
        except Exception as e:
            st.error(f"HTML parse error: {e}")
            return ""

    return raw.decode("utf-8", errors="ignore")


SYSTEM_PROMPT = """You are a municipal bond credit analyst specializing in New Jersey local government.
Jersey City context:
- Relies ~30%+ on PILOT (Payment in Lieu of Taxes) revenue; concentration risk is high
- Moody's A2 rated; watch items: pension (PFRS unfunded ~$1.2B), tax appeals (~$380M pending)
- PFRS/PERS contributions must stay current to avoid credit negative events

Analyze the provided council meeting minutes and return ONLY valid JSON (no markdown fences) with this exact shape:
{
  "score": <float -1.0 to +1.0>,
  "signal": <"Bullish" | "Neutral" | "Bearish">,
  "summary": <2-3 sentence plain-English summary for a bond analyst>,
  "categories": {
    "fiscal_stress": <int count of signals>,
    "pilot": <int>,
    "pension": <int>,
    "political_cohesion": <int>,
    "positive": <int>
  },
  "leading_indicators": [<up to 3 short bullet strings to watch>],
  "key_items": [<up to 4 notable agenda items as short strings>]
}"""


def run_analysis(text: str, api_key: str) -> dict:
    """Call Claude and return parsed JSON result."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # Truncate to ~60k chars to stay within context
    truncated = text[:60_000]

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Council minutes:\n\n{truncated}"}],
    )
    raw = message.content[0].text.strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


# ── seed / demo data ──────────────────────────────────────────────────────────
SEED_RESULTS = [
    {"date": "2025-12-10", "score": 0.05,  "signal": "Neutral",
     "categories": {"fiscal_stress": 1, "pilot": 4, "pension": 0, "political_cohesion": 4, "positive": 4},
     "summary": "End-of-year meeting focused on capital carry-forwards and routine resolutions. New PILOT agreement for Journal Square Transit Village Phase II introduced on first reading. Pension contributions confirmed on schedule.",
     "leading_indicators": ["Journal Square PILOT concentration risk growing", "Tax appeal reserve adequacy", "PFRS contribution schedule 2026"],
     "key_items": ["Journal Square Phase II PILOT (1st reading)", "Capital carry-forward resolutions", "Pension contribution confirmation", "Budget amendment — IT infrastructure"]},
    {"date": "2025-11-19", "score": -0.12, "signal": "Bearish",
     "categories": {"fiscal_stress": 3, "pilot": 2, "pension": 1, "political_cohesion": 2, "positive": 2},
     "summary": "Contested budget amendment for emergency infrastructure repair drew two dissenting votes. Tax appeal settlement approved at $4.2M above reserve estimate, signaling potential reserve shortfall.",
     "leading_indicators": ["Tax appeal reserve under pressure", "Split votes on fiscal items", "Emergency capital spending"],
     "key_items": ["Tax appeal settlement $4.2M", "Emergency road repair budget amendment", "Split 7-2 vote on reserve transfer", "PILOT renewal — Liberty Harbor"]},
    {"date": "2025-10-15", "score": 0.22, "signal": "Bullish",
     "categories": {"fiscal_stress": 0, "pilot": 3, "pension": 0, "political_cohesion": 5, "positive": 6},
     "summary": "Strong meeting with unanimous passage of new mixed-use development PILOT and a $2.1M state grant award for transit infrastructure. No fiscal stress signals detected.",
     "leading_indicators": ["New PILOT pipeline healthy", "State grant momentum", "Construction activity supporting ratables"],
     "key_items": ["Mixed-use PILOT — McGinley Square", "$2.1M NJ DOT transit grant", "Unanimous budget surplus certification", "Affordable housing bond resolution"]},
]


# ── signal color helpers ───────────────────────────────────────────────────────
SIGNAL_COLOR = {"Bullish": "#16a34a", "Neutral": "#d97706", "Bearish": "#dc2626"}
SIGNAL_BG    = {"Bullish": "#d1fae5", "Neutral": "#fef3c7", "Bearish": "#fee2e2"}
SIGNAL_FG    = {"Bullish": "#065f46", "Neutral": "#92400e", "Bearish": "#991b1b"}

CAT_LABELS = ["Fiscal Stress", "PILOT", "Pension", "Political Cohesion", "Positive"]
CAT_KEYS   = ["fiscal_stress", "pilot", "pension", "political_cohesion", "positive"]
CAT_COLORS = ["#dc2626", "#f97316", "#a855f7", "#3b82f6", "#16a34a"]


def render_result(result: dict, label: str = ""):
    """Render a single analysis result card."""
    sig = result["signal"]
    score = result["score"]
    pill_cls = f"pill-{sig.lower()}"

    if label:
        st.markdown(f"#### {label}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Signal", sig)
    col2.metric("Score", f"{score:+.2f}")
    col3.metric("Date", result.get("date", "—"))

    st.markdown(f"**Summary:** {result['summary']}")

    # Category bar chart
    cats = result["categories"]
    counts = [cats.get(k, 0) for k in CAT_KEYS]
    fig = go.Figure(go.Bar(
        x=CAT_LABELS, y=counts,
        marker_color=CAT_COLORS,
        text=counts, textposition="outside",
    ))
    fig.update_layout(
        height=280, margin=dict(t=20, b=20, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=False, showticklabels=False),
        xaxis=dict(tickfont=dict(size=12)),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if result.get("leading_indicators"):
            st.markdown("**⚠️ Leading Indicators**")
            for li in result["leading_indicators"]:
                st.markdown(f"- {li}")
    with c2:
        if result.get("key_items"):
            st.markdown("**📋 Key Agenda Items**")
            for ki in result["key_items"]:
                st.markdown(f"- {ki}")


# ── main layout ────────────────────────────────────────────────────────────────
st.markdown("# 🏦 Jersey City Municipal Bond Sentiment")
st.caption("NLP-powered fiscal signal extraction from public council records · Manohar Analytics")
st.divider()

# ── Section 1: Seed data overview ─────────────────────────────────────────────
st.subheader("📊 Pre-Loaded Signal History  (Jul–Dec 2025)")

# Summary metrics across seed data
avg_score = sum(r["score"] for r in SEED_RESULTS) / len(SEED_RESULTS)
latest = SEED_RESULTS[0]
latest_sig = latest["signal"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest Signal", latest_sig, f"{latest['score']:+.2f}")
m2.metric("Meetings Analyzed", len(SEED_RESULTS))
m3.metric("Avg Sentiment", f"{avg_score:+.2f}")
m4.metric("Political Cohesion", "High", "Latest: 0.78")

# Mini trend table
trend_df = pd.DataFrame([
    {"Date": r["date"], "Signal": r["signal"], "Score": f"{r['score']:+.2f}"}
    for r in SEED_RESULTS
])
st.dataframe(trend_df, use_container_width=True, hide_index=True)

# Expandable detail for each seed meeting
for r in SEED_RESULTS:
    with st.expander(f"{r['date']} — **{r['signal']}** ({r['score']:+.2f})"):
        render_result(r)

st.divider()

# ── Section 2: Upload & analyze ────────────────────────────────────────────────
st.subheader("📄 Analyze a New Council Meeting")

api_key = get_api_key()
if not api_key:
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Add to .streamlit/secrets.toml as ANTHROPIC_API_KEY to avoid entering each time.",
    )

uploaded = st.file_uploader(
    "Upload council minutes (PDF, DOCX, TXT, or HTML)",
    type=["pdf", "docx", "txt", "html", "htm"],
)

if uploaded:
    if not api_key:
        st.warning("Enter your Anthropic API key above to run analysis.")
    else:
        with st.spinner("Extracting text…"):
            text = extract_text(uploaded)

        if not text.strip():
            st.error("Could not extract text from this file. Try a different format.")
        else:
            word_count = len(text.split())
            st.caption(f"Extracted {word_count:,} words from {uploaded.name}")

            if st.button("▶ Run Analysis", type="primary"):
                with st.spinner("Analyzing with Claude…"):
                    try:
                        result = run_analysis(text, api_key)
                        result["date"] = "just now"
                        st.success("Analysis complete.")
                        render_result(result, label=f"Results — {uploaded.name}")
                    except json.JSONDecodeError as e:
                        st.error(f"Could not parse Claude's response as JSON: {e}")
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

st.divider()
st.caption(
    "**Disclaimer:** For research and informational purposes only. Not investment advice. "
    "Municipal bond trading involves risk. Conduct independent due diligence. · v2.0 · Manohar Analytics"
)
