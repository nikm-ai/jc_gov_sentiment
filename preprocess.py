"""
preprocess.py — One-time script to analyze JC council meeting PDFs and
write data/seed_data.json for use as pre-loaded seed data in app.py.

Usage:
    1. Place your PDFs in data/pdfs/
    2. Set ANTHROPIC_API_KEY in your environment or .env file
    3. Run: python preprocess.py
    4. To add new files later: python preprocess.py --resume

Cost estimate: ~$0.05–0.08 per meeting (claude-sonnet), ~$0.50–0.80 for 10 meetings.
"""

import os
import io
import re
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

import anthropic
import PyPDF2
from bs4 import BeautifulSoup

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
PDF_DIR = ROOT / "data" / "pdfs"
OUT_FILE = ROOT / "data" / "seed_data.json"

# ══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — richer schema with recommendation, evidence, implications
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a senior municipal bond credit analyst with deep expertise in
New Jersey local government finance, writing for an institutional fixed-income audience.

Jersey City fiscal context (treat as standing background knowledge):
- PILOT (Payment in Lieu of Taxes) revenue constitutes approximately 30%+ of total tax levy.
  Expiration, court challenge, or non-renewal of major PILOT agreements = meaningful revenue risk.
- Moody's current rating: A2. Key watch items: PFRS pension unfunded liability (~$1.2B),
  tax appeal reserve adequacy (~$380M pending vs. ~$22M reserve), and abatement pipeline.
- PFRS and PERS contributions must remain current; any deferral = direct credit-negative trigger.
- Political cohesion (vote splits on fiscal items) signals execution risk on budget plans.
- Tax appeal reserve-to-exposure ratio of ~6% is thin; settlements above reserve estimates
  erode fiscal flexibility immediately.

Analyze the provided council meeting minutes in full detail. Return ONLY valid JSON —
no markdown fences, no preamble, no trailing text. Use this exact shape:

{
  "score": <float, -1.0 to +1.0>,
  "signal": <"Bullish" | "Neutral" | "Bearish">,

  "credit_recommendation": <"Overweight" | "Market Weight" | "Underweight" | "Monitor">,
  "recommendation_rationale": <string, 1-2 sentences explaining the recommendation directly
                                in terms of bond positioning — mention spread, duration,
                                or relative value where relevant>,

  "summary": <string, 3-4 sentences of precise analyst prose: what happened at this meeting,
               what it means for the credit, and what the key forward risks are>,

  "credit_implications": <string, 2-3 sentences explicitly connecting the meeting's findings
                           to bond market implications — spread direction, rating trajectory,
                           debt service coverage, or liquidity. Be specific and direct.>,

  "categories": {
    "fiscal_stress":       <int, count of distinct fiscal stress signals>,
    "pilot":               <int, count of PILOT/abatement signals>,
    "pension":             <int, count of pension/PFRS/PERS signals>,
    "political_cohesion":  <int, count of governance/vote-quality signals>,
    "positive":            <int, count of positive credit signals>
  },

  "evidence": [
    <up to 5 objects, each with:
      "category": <one of "fiscal_stress"|"pilot"|"pension"|"political_cohesion"|"positive">,
      "signal":   <short label, e.g. "Split vote on reserve transfer">,
      "detail":   <1-2 sentences quoting or paraphrasing the specific agenda item or
                   discussion that generated this signal — be concrete>,
      "direction": <"negative" | "positive" | "neutral">
    >
  ],

  "leading_indicators": [<3 concise strings — specific forward-looking items to monitor
                           at future meetings, with explicit credit relevance>],

  "key_items": [<4 concise strings — most notable agenda items by credit significance>],

  "risk_flags": [<0–3 strings — discrete credit-negative items requiring analyst escalation.
                  Empty list if none. Each flag should name the specific item and its
                  credit implication.>]
}

Score calibration:
  0.0          = no net signal (housekeeping session)
  ±0.05–0.15   = mild directional signal
  ±0.15–0.35   = moderate — warrants monitoring
  ±0.35–0.60   = strong — warrants positioning review
  ±0.60+       = very strong — material credit event

Credit recommendation calibration:
  Overweight   = score > +0.20 and no material risk flags
  Market Weight = score -0.15 to +0.20, or flags present but offset by positives
  Underweight  = score < -0.25 or 2+ risk flags with no strong offsets
  Monitor      = mixed signals, score near zero but with specific watch items"""


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages = [reader.pages[i].extract_text() or "" for i in range(len(reader.pages))]
        text = "\n".join(pages).strip()
        if not text:
            log.warning(f"  No text extracted from {path.name} — may be a scanned PDF.")
        return text
    except Exception as e:
        log.error(f"  PDF read error for {path.name}: {e}")
        return ""

def extract_text_from_html(path: Path) -> str:
    try:
        return BeautifulSoup(path.read_bytes(), "html.parser").get_text(separator="\n")
    except Exception as e:
        log.error(f"  HTML read error for {path.name}: {e}")
        return ""

def extract_text_from_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        log.error(f"  TXT read error for {path.name}: {e}")
        return ""

def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":         return extract_text_from_pdf(path)
    elif suffix in (".html", ".htm"): return extract_text_from_html(path)
    elif suffix == ".txt":       return extract_text_from_txt(path)
    else:
        log.warning(f"  Unsupported file type: {path.name}")
        return ""


# ── Date extraction ───────────────────────────────────────────────────────────

def extract_date_from_filename(path: Path) -> str:
    name = path.stem
    # YYYY-MM-DD or YYYY_MM_DD
    m = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # YYYYMMDD
    m = re.search(r"(\d{8})", name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
    # MM.DD.YYYY or MM-DD-YYYY or MM/DD/YYYY
    m = re.search(r"(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})", name)
    if m:
        try:
            return f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"
        except Exception:
            pass
    return name


# ── Validation & defaults ─────────────────────────────────────────────────────

REQUIRED_KEYS = {
    "score", "signal", "credit_recommendation", "recommendation_rationale",
    "summary", "credit_implications", "categories",
    "evidence", "leading_indicators", "key_items", "risk_flags"
}

DEFAULTS = {
    "credit_recommendation": "Monitor",
    "recommendation_rationale": "Insufficient signal clarity to make a directional recommendation.",
    "credit_implications": "No material credit implications identified in this session.",
    "evidence": [],
    "leading_indicators": [],
    "key_items": [],
    "risk_flags": [],
    "categories": {
        "fiscal_stress": 0, "pilot": 0, "pension": 0,
        "political_cohesion": 0, "positive": 0
    },
}

def validate_result(result: dict) -> dict:
    """Fill missing keys with defaults and clamp score."""
    for k, v in DEFAULTS.items():
        if k not in result:
            log.warning(f"  Missing key '{k}' — using default.")
            result[k] = v
    result["score"] = max(-1.0, min(1.0, float(result.get("score", 0.0))))
    if result.get("signal") not in ("Bullish", "Neutral", "Bearish"):
        result["signal"] = "Neutral"
    if result.get("credit_recommendation") not in ("Overweight", "Market Weight", "Underweight", "Monitor"):
        result["credit_recommendation"] = "Monitor"
    return result


# ── Claude call ───────────────────────────────────────────────────────────────

def analyze_document(client: anthropic.Anthropic, text: str,
                     retries: int = 3, delay: float = 8.0) -> dict | None:
    truncated = text[:60_000]
    for attempt in range(1, retries + 1):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Council minutes for analysis:\n\n{truncated}"
                }],
            )
            raw = message.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"): raw = raw[4:]
            result = json.loads(raw)
            return validate_result(result)

        except json.JSONDecodeError as e:
            log.error(f"  JSON parse error (attempt {attempt}/{retries}): {e}")
            if attempt == retries: return None
            time.sleep(delay)

        except anthropic.RateLimitError:
            wait = delay * attempt
            log.warning(f"  Rate limited. Waiting {wait:.0f}s…")
            time.sleep(wait)

        except anthropic.APIStatusError as e:
            log.error(f"  API error {e.status_code} (attempt {attempt}/{retries})")
            if attempt == retries: return None
            time.sleep(delay)

        except Exception as e:
            log.error(f"  Unexpected error (attempt {attempt}/{retries}): {e}")
            if attempt == retries: return None
            time.sleep(delay)

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-process JC council meeting PDFs.")
    parser.add_argument("--pdf-dir",  default=str(PDF_DIR))
    parser.add_argument("--out-file", default=str(OUT_FILE))
    parser.add_argument("--delay",    type=float, default=4.0,
                        help="Seconds between API calls (default 4).")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip files already in output JSON.")
    args = parser.parse_args()

    pdf_dir  = Path(args.pdf_dir)
    out_file = Path(args.out_file)

    # API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set.")
        raise SystemExit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Discover files
    if not pdf_dir.exists():
        log.error(f"Directory not found: {pdf_dir}")
        raise SystemExit(1)

    patterns = ["*.pdf", "*.html", "*.htm", "*.txt"]
    files = sorted({
        path for pat in patterns
        for path in pdf_dir.glob(pat)
    }, key=lambda p: p.name.lower())

    if not files:
        log.error(f"No files found in {pdf_dir}")
        raise SystemExit(1)

    log.info(f"Found {len(files)} unique file(s) in {pdf_dir}")

    # Load existing if resuming
    existing: dict[str, dict] = {}
    if args.resume and out_file.exists():
        try:
            existing_list = json.loads(out_file.read_text())
            existing = {r["filename"]: r for r in existing_list if "filename" in r}
            log.info(f"Resuming: {len(existing)} file(s) already processed.")
        except Exception as e:
            log.warning(f"Could not load existing output: {e}")

    results = []
    skipped = failed = 0

    for i, path in enumerate(files, 1):
        log.info(f"[{i}/{len(files)}] {path.name}")

        if args.resume and path.name in existing:
            log.info(f"  → Already processed. Skipping.")
            results.append(existing[path.name])
            skipped += 1
            continue

        log.info(f"  Extracting text…")
        text = extract_text(path)
        word_count = len(text.split())

        if not text.strip():
            log.error(f"  No text extracted. Skipping.")
            failed += 1
            continue

        log.info(f"  Extracted {word_count:,} words.")
        date_str = extract_date_from_filename(path)
        log.info(f"  Date: {date_str}")
        log.info(f"  Calling Claude…")

        result = analyze_document(client, text)
        if result is None:
            log.error(f"  Analysis failed. Skipping.")
            failed += 1
            continue

        result["date"]     = date_str
        result["filename"] = path.name
        result["words"]    = word_count

        rec   = result.get("credit_recommendation", "?")
        sig   = result.get("signal", "?")
        score = result.get("score", 0.0)
        flags = len(result.get("risk_flags", []))
        log.info(f"  ✓ {sig} ({score:+.2f}) | {rec} | {flags} flag(s)")

        results.append(result)

        # Save incrementally
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        if i < len(files):
            log.info(f"  Waiting {args.delay:.0f}s…")
            time.sleep(args.delay)

    # Sort newest-first
    results.sort(key=lambda r: r.get("date", ""), reverse=True)
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    log.info("─" * 60)
    log.info(f"Done.  Processed: {len(results)-skipped}  |  Skipped: {skipped}  |  Failed: {failed}")
    log.info(f"Output: {out_file}")
    log.info("─" * 60)

    if results:
        avg = sum(r.get("score", 0) for r in results) / len(results)
        log.info(f"Period avg score: {avg:+.2f}")
        for r in results:
            flags    = len(r.get("risk_flags", []))
            flag_str = f"  ⚑ {flags}" if flags else ""
            rec      = r.get("credit_recommendation", "?")
            log.info(f"  {r['date']}  {r['signal']:8s}  {r.get('score',0):+.2f}  [{rec:13s}]{flag_str}  — {r['filename']}")


if __name__ == "__main__":
    main()
