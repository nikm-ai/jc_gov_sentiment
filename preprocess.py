"""
preprocess.py — v4.0
Analyze JC council meeting PDFs and write data/seed_data.json.

Upgrades from v3:
- More aggressive pension signal extraction (PFRS, PERS, ADEC, any pension-adjacent items)
- score_breakdown field in output (per-category contributions for methodology transparency)
- Improved validation with pension floor warning

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT    = Path(__file__).parent
PDF_DIR = ROOT / "data" / "pdfs"
OUT_FILE = ROOT / "data" / "seed_data.json"

# ══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — v4.0: aggressive pension extraction + score_breakdown
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

═══════════════════════════════════════════════════
PENSION SIGNAL EXTRACTION — READ THIS CAREFULLY
═══════════════════════════════════════════════════
Count a pension signal for ANY of the following items in the meeting minutes:
  - Any explicit mention of PFRS, PERS, pension, retirement, actuarial, ADEC, unfunded liability
  - Any budget appropriation or resolution that includes pension-related line items
  - Any ordinance appropriating funds that could include employee benefit obligations
  - Any discussion of employee compensation packages, deferred compensation, or retirement costs
  - Any reference to state-mandated contribution schedules or pension reform measures
  - Any emergency or special appropriation that may cover shortfalls including pension-related
  - Budget amendments or temporary emergency appropriations (these typically contain pension line items)
  - Salary ordinances or compensation schedules (include pension-related payroll costs)
  - Any resolution referencing the state Division of Local Government Services in budget context
  - "Salary and wages" appropriations in any budget resolution (pension costs attach to payroll)

IMPORTANT: Jersey City's pension obligations are embedded throughout its budget. Almost every
regular council meeting that includes any budget appropriation or salary/compensation item will
contain at least one pension-adjacent signal. Return 0 ONLY if the meeting is purely ceremonial
(proclamations, tributes) with zero fiscal content.

═══════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════
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
    "pension":             <int, count of pension/PFRS/PERS/retirement-adjacent signals
                            — see detailed counting instructions above>,
    "political_cohesion":  <int, count of governance/vote-quality signals>,
    "positive":            <int, count of positive credit signals>
  },

  "score_breakdown": {
    "fiscal_contribution":    <float, fiscal_stress count * -0.20>,
    "pilot_contribution":     <float, pilot count * -0.10>,
    "pension_contribution":   <float, pension count * -0.15>,
    "cohesion_contribution":  <float, political_cohesion count * -0.08>,
    "positive_contribution":  <float, positive count * +0.15>,
    "flag_penalty":           <float, -0.10 per risk flag, 0 if none>,
    "raw_sum":                <float, sum of all contributions above>,
    "clamped_score":          <float, raw_sum clamped to [-1.0, +1.0]>
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
    if suffix == ".pdf":               return extract_text_from_pdf(path)
    elif suffix in (".html", ".htm"):  return extract_text_from_html(path)
    elif suffix == ".txt":             return extract_text_from_txt(path)
    else:
        log.warning(f"  Unsupported file type: {path.name}")
        return ""


# ── Date extraction ───────────────────────────────────────────────────────────

def extract_date_from_filename(path: Path) -> str:
    name = path.stem
    m = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"(\d{8})", name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
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
    "summary", "credit_implications", "categories", "score_breakdown",
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
    "score_breakdown": {
        "fiscal_contribution": 0.0, "pilot_contribution": 0.0,
        "pension_contribution": 0.0, "cohesion_contribution": 0.0,
        "positive_contribution": 0.0, "flag_penalty": 0.0,
        "raw_sum": 0.0, "clamped_score": 0.0,
    },
}

def validate_result(result: dict) -> dict:
    """Fill missing keys with defaults, clamp score, warn on suspiciously zero pension."""
    for k, v in DEFAULTS.items():
        if k not in result:
            log.warning(f"  Missing key '{k}' — using default.")
            result[k] = v

    result["score"] = max(-1.0, min(1.0, float(result.get("score", 0.0))))

    if result.get("signal") not in ("Bullish", "Neutral", "Bearish"):
        result["signal"] = "Neutral"
    if result.get("credit_recommendation") not in ("Overweight", "Market Weight", "Underweight", "Monitor"):
        result["credit_recommendation"] = "Monitor"

    # Warn if pension is zero — this is often a prompt miss for JC
    cats = result.get("categories", {})
    if cats.get("pension", 0) == 0:
        log.warning(
            "  ⚠ Pension signal count = 0. Verify: does this meeting contain any budget "
            "appropriations, salary items, or compensation resolutions? Consider re-running."
        )

    # Ensure score_breakdown is present and complete
    if "score_breakdown" not in result or not isinstance(result["score_breakdown"], dict):
        # Reconstruct from categories if breakdown missing
        cats = result.get("categories", {})
        flags = len(result.get("risk_flags", []))
        fc  = cats.get("fiscal_stress", 0)      * -0.20
        pc  = cats.get("pilot", 0)               * -0.10
        pen = cats.get("pension", 0)             * -0.15
        cc  = cats.get("political_cohesion", 0)  * -0.08
        pos = cats.get("positive", 0)            * +0.15
        fp  = flags * -0.10
        raw = fc + pc + pen + cc + pos + fp
        result["score_breakdown"] = {
            "fiscal_contribution": round(fc, 3),
            "pilot_contribution": round(pc, 3),
            "pension_contribution": round(pen, 3),
            "cohesion_contribution": round(cc, 3),
            "positive_contribution": round(pos, 3),
            "flag_penalty": round(fp, 3),
            "raw_sum": round(raw, 3),
            "clamped_score": round(max(-1.0, min(1.0, raw)), 3),
        }
    return result


# ── Claude call ───────────────────────────────────────────────────────────────

def analyze_document(client: anthropic.Anthropic, text: str,
                     retries: int = 3, delay: float = 8.0) -> dict | None:
    truncated = text[:60_000]
    for attempt in range(1, retries + 1):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2500,
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
    parser.add_argument("--reprocess-zero-pension", action="store_true",
                        help="Re-run analysis for any meeting with pension count = 0.")
    args = parser.parse_args()

    pdf_dir  = Path(args.pdf_dir)
    out_file = Path(args.out_file)

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

    existing: dict[str, dict] = {}
    if (args.resume or args.reprocess_zero_pension) and out_file.exists():
        try:
            existing_list = json.loads(out_file.read_text())
            existing = {r["filename"]: r for r in existing_list if "filename" in r}
            log.info(f"Loaded {len(existing)} existing result(s).")
        except Exception as e:
            log.warning(f"Could not load existing output: {e}")

    results = []
    skipped = failed = reprocessed = 0

    for i, path in enumerate(files, 1):
        log.info(f"[{i}/{len(files)}] {path.name}")

        should_skip = args.resume and path.name in existing
        if args.reprocess_zero_pension and path.name in existing:
            pension_count = existing[path.name].get("categories", {}).get("pension", 0)
            if pension_count == 0:
                log.info(f"  → pension=0, reprocessing.")
                should_skip = False
                reprocessed += 1
            else:
                should_skip = args.resume  # respect --resume for non-zero pension

        if should_skip:
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
        log.info(f"  Date: {date_str} | Calling Claude…")

        result = analyze_document(client, text)
        if result is None:
            log.error(f"  Analysis failed. Skipping.")
            failed += 1
            continue

        result["date"]     = date_str
        result["filename"] = path.name
        result["words"]    = word_count

        rec    = result.get("credit_recommendation", "?")
        sig    = result.get("signal", "?")
        score  = result.get("score", 0.0)
        flags  = len(result.get("risk_flags", []))
        pension = result.get("categories", {}).get("pension", 0)
        log.info(f"  ✓ {sig} ({score:+.2f}) | {rec} | {flags} flag(s) | pension={pension}")

        results.append(result)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        if i < len(files):
            log.info(f"  Waiting {args.delay:.0f}s…")
            time.sleep(args.delay)

    results.sort(key=lambda r: r.get("date", ""), reverse=True)
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    log.info("─" * 60)
    log.info(f"Done.  Processed: {len(results)-skipped}  |  Skipped: {skipped}  |  Failed: {failed}  |  Reprocessed: {reprocessed}")
    log.info(f"Output: {out_file}")
    log.info("─" * 60)

    if results:
        avg = sum(r.get("score", 0) for r in results) / len(results)
        pension_zeros = sum(1 for r in results if r.get("categories", {}).get("pension", 0) == 0)
        log.info(f"Period avg score: {avg:+.2f}  |  Pension=0 count: {pension_zeros}/{len(results)}")
        for r in results:
            flags    = len(r.get("risk_flags", []))
            flag_str = f"  ⚑ {flags}" if flags else ""
            rec      = r.get("credit_recommendation", "?")
            pension  = r.get("categories", {}).get("pension", 0)
            log.info(
                f"  {r['date']}  {r['signal']:8s}  {r.get('score',0):+.2f}  "
                f"[{rec:13s}]  pension={pension}{flag_str}  — {r['filename']}"
            )

        if pension_zeros > 0:
            log.warning(
                f"\n  ⚠ {pension_zeros} meeting(s) have pension=0. "
                f"Run with --reprocess-zero-pension to re-analyze these with the upgraded prompt."
            )


if __name__ == "__main__":
    main()
