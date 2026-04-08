# JC Municipal Bond Sentiment

[![Live Demo](https://img.shields.io/badge/Live%20Demo-jcgovsentiment.streamlit.app-1a4f82?style=flat-square)](https://jcgovsentiment.streamlit.app/)
[![Built with Claude](https://img.shields.io/badge/Powered%20by-Anthropic%20Claude-5c3d82?style=flat-square)](https://anthropic.com)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square)](https://streamlit.io)

NLP-powered fiscal signal extraction from Jersey City public council records, built for municipal bond research.

---

## What is this?

Jersey City carries **~$1.2B in unfunded pension liability** and relies on tax abatement payments (PILOTs) for nearly **30% of its total budget**. When the city council meets, the agenda text contains early signals about whether that credit picture is improving or deteriorating.

This tool reads council meeting minutes and translates them into an explicit bond positioning recommendation — **Overweight / Market Weight / Underweight / Monitor** — with sourced evidence. The same analytical workflow a muni credit analyst would follow, automated.

---

## What it produces

For each council meeting document, the model returns:

| Output | Description |
|---|---|
| **Sentiment score** | Continuous −1.0 (bearish) to +1.0 (bullish), weighted by credit materiality |
| **Credit recommendation** | Overweight / Market Weight / Underweight / Monitor |
| **Score breakdown** | Per-category contribution table showing exactly how the score was derived |
| **Analyst summary** | 3–4 sentence prose connecting meeting content to credit trajectory |
| **Credit implications** | Spread direction, rating trajectory, and debt service coverage statement |
| **Signal evidence** | Up to 5 sourced signals, each citing the specific agenda item |
| **Risk flags** | Discrete credit-negative events requiring analyst escalation |
| **Leading indicators** | 3 forward-looking items to monitor at future meetings |

---

## Signal categories and weights

| Category | Per-Signal Weight | Credit Direction |
|---|---|---|
| Fiscal Stress | −0.20 | Negative |
| Pension / PFRS / PERS | −0.15 | Negative |
| Political Cohesion | −0.08 | Negative |
| PILOT / Abatement | −0.10 | Directional |
| Positive Events | +0.15 | Positive |
| Risk Flag Penalty | −0.10 each | Negative |

Scores are computed as a weighted sum of signal counts, then clamped to [−1.0, +1.0].

**Recommendation thresholds:**
- **Overweight:** score > +0.20, no material flags
- **Market Weight:** −0.15 to +0.20
- **Monitor:** near-zero with specific watch items
- **Underweight:** score < −0.25 or 2+ risk flags

---

## Jersey City credit context (embedded in every analysis)

- **Moody's A2** issuer rating
- **~$1.2B PFRS unfunded liability** — contribution deferrals = direct credit-negative trigger
- **~$380M pending tax appeals** vs. ~$22M reserve (~6% coverage ratio)
- **~30%+ PILOT revenue dependency** — concentration and expiration risk
- Peer universe: NJ investment-grade GO issuers rated A2/A or better

---

## Getting started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Pre-process meeting PDFs

1. Place council meeting PDFs in `data/pdfs/`
2. Set your API key:
   ```bash
   export ANTHROPIC_API_KEY=your_key_here
   ```
3. Run preprocessing:
   ```bash
   python preprocess.py
   ```
4. To add new meetings without reprocessing existing ones:
   ```bash
   python preprocess.py --resume
   ```
5. To re-run any meetings where pension signals were undercounted:
   ```bash
   python preprocess.py --reprocess-zero-pension
   ```

### Run the app locally

```bash
streamlit run app.py
```

For the live analysis feature, add your key to `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "your_key_here"
```

### Obtaining council minutes

**CivicWeb Portal (free, HTML):**
Navigate to `cityofjerseycity.civicweb.net/portal/` → Meetings → Regular Meeting of Municipal Council → select date → Minutes. Save as HTML.

**OPRA Request (PDF):**
Contact the City Clerk at (201) 547-5150. Text-layer PDFs work best.

---

## Project structure

```
jc_gov_sentiment/
├── app.py              # Streamlit application
├── preprocess.py       # Batch PDF analysis script
├── requirements.txt
├── data/
│   ├── pdfs/           # Source meeting documents
│   └── seed_data.json  # Pre-analyzed meeting history
└── .streamlit/
    └── secrets.toml    # API key (not committed)
```

---

## Deployment (Streamlit Cloud)

1. Push the repo to GitHub (ensure `data/seed_data.json` is committed)
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Set `ANTHROPIC_API_KEY` in the app's Secrets panel
4. The pre-analyzed history loads automatically; live analysis is enabled for authenticated sessions

---

---

*For research and informational purposes only. Not investment advice. Municipal bond trading involves risk; conduct independent due diligence. Credit recommendations are model outputs and should not be the sole basis for investment decisions.*
