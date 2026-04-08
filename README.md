# JC Municipal Bond Sentiment

NLP-powered fiscal signal extraction from Jersey City public council records, built for municipal bond research.

## What it does

Upload a Jersey City council meeting minutes file (PDF, DOCX, TXT, or HTML) and get an instant structured bond sentiment signal powered by Claude:

- **Score** (-1.0 bearish → +1.0 bullish) with signal label
- **Category breakdown** across 5 signal types: Fiscal Stress, PILOT, Pension, Political Cohesion, Positive
- **Leading indicators** to watch
- **Key agenda items** summary

Pre-loaded seed data (Jul–Dec 2025) is included for context and demo purposes.

## Signal categories

| Category | What it captures |
|---|---|
| Fiscal Stress | Shortfalls, budget amendments, reserve drawdowns, BAN issuances |
| PILOT / Abatements | New agreements, expirations, revenue concentration risk |
| Pension | PFRS/PERS contribution status, unfunded liability trends |
| Political Cohesion | Split votes, dissents, contested budget items |
| Positive | New development, surplus, rating upgrades, grant awards |

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set your API key in `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```
Or enter it directly in the app UI.

## Getting council minutes

1. **CivicWeb Portal** (free, HTML): https://cityofjerseycity.civicweb.net/portal/
   - Meetings → Regular Meeting of Municipal Council → select date → Minutes → Save page as HTML
2. **OPRA Request** (PDF): Contact City Clerk at (201) 547-5150

## JC bond context

- Relies ~30%+ on PILOT revenue — concentration and expiration risk
- Moody's A2 rated
- PFRS unfunded liability ~$1.2B; tax appeals ~$380M pending vs ~$22M reserve

## Disclaimer

For research and informational purposes only. Not investment advice.
