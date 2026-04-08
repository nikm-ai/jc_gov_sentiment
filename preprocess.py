"""
preprocess.py — One-time script to analyze JC council meeting PDFs and
write data/seed_data.json for use as pre-loaded seed data in app.py.

Usage:
    1. Place your PDFs in data/pdfs/
    2. Set ANTHROPIC_API_KEY in your environment (or .env file)
    3. Run: python preprocess.py

Cost estimate: ~$0.02–0.05 per meeting (claude-sonnet), so ~$0.20–0.50 for 10 meetings.
Time estimate: ~2–4 minutes for 10 meetings (rate limit pauses included).
"""

import os
import io
import re
import json
import time
import glob
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
ROOT      = Path(__file__).parent
PDF_DIR   = ROOT / "data" / "pdfs"
OUT_FILE  = ROOT / "data" / "seed_data.json"

# ── Claude prompt (identical to app.py so results are consistent) ──────────────
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


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf(path: Path) -> str:
    """Extract text from a PDF file. Returns empty string on failure."""
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
    """Extract text from a saved HTML file."""
    try:
        raw = path.read_bytes()
        return BeautifulSoup(raw, "html.parser").get_text(separator="\n")
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
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    elif suffix in (".html", ".htm"):
        return extract_text_from_html(path)
    elif suffix == ".txt":
        return extract_text_from_txt(path)
    else:
        log.warning(f"  Unsupported file type: {path.name}")
        return ""


# ── Date extraction from filename ─────────────────────────────────────────────

def extract_date_from_filename(path: Path) -> str:
    """
    Try to find a YYYY-MM-DD or MM-DD-YYYY or YYYYMMDD pattern in the filename.
    Falls back to the stem (filename without extension) if nothing matches.
    """
    name = path.stem

    # YYYY-MM-DD
    m = re.search(r"(\d{4}[-_]\d{2}[-_]\d{2})", name)
    if m:
        return m.group(1).replace("_", "-")

    # YYYYMMDD
    m = re.search(r"(\d{8})", name)
    if m:
        s = m.group(1)
        try:
            dt = datetime.strptime(s, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # MM-DD-YYYY or MM/DD/YYYY
    m = re.search(r"(\d{1,2}[-_/]\d{1,2}[-_/]\d{4})", name)
    if m:
        raw = m.group(1).replace("_", "-").replace("/", "-")
        for fmt in ("%m-%d-%Y", "%-m-%-d-%Y"):
            try:
                dt = datetime.strptime(raw, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

    # Fall back to full stem
    return name


# ── Claude call ───────────────────────────────────────────────────────────────

def analyze_document(client: anthropic.Anthropic, text: str, filename: str,
                     retries: int = 3, delay: float = 5.0) -> dict | None:
    """
    Call Claude to extract bond signals. Retries on rate limit / server errors.
    Returns parsed dict or None on failure.
    """
    truncated = text[:60_000]
    for attempt in range(1, retries + 1):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1200,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Council minutes for analysis:\n\n{truncated}"
                }],
            )
            raw = message.content[0].text.strip()

            # Strip accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            result = json.loads(raw)

            # Basic validation
            required_keys = {"score", "signal", "summary", "categories",
                              "leading_indicators", "key_items", "risk_flags"}
            missing = required_keys - set(result.keys())
            if missing:
                log.warning(f"  Response missing keys: {missing}. Filling with defaults.")
                for k in missing:
                    result[k] = [] if k in ("leading_indicators", "key_items", "risk_flags") else \
                                 {} if k == "categories" else \
                                 0.0 if k == "score" else "Neutral"

            return result

        except json.JSONDecodeError as e:
            log.error(f"  JSON parse error (attempt {attempt}/{retries}): {e}")
            if attempt == retries:
                return None
            time.sleep(delay)

        except anthropic.RateLimitError:
            wait = delay * attempt
            log.warning(f"  Rate limited. Waiting {wait:.0f}s before retry {attempt}/{retries}…")
            time.sleep(wait)

        except anthropic.APIStatusError as e:
            log.error(f"  API error (attempt {attempt}/{retries}): {e.status_code} — {e.message}")
            if attempt == retries:
                return None
            time.sleep(delay)

        except Exception as e:
            log.error(f"  Unexpected error (attempt {attempt}/{retries}): {e}")
            if attempt == retries:
                return None
            time.sleep(delay)

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-process JC council meeting PDFs.")
    parser.add_argument("--pdf-dir",  default=str(PDF_DIR),
                        help="Directory containing PDF/HTML/TXT files.")
    parser.add_argument("--out-file", default=str(OUT_FILE),
                        help="Output JSON path.")
    parser.add_argument("--delay",    type=float, default=3.0,
                        help="Seconds to wait between API calls (default 3).")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip files already present in output JSON.")
    args = parser.parse_args()

    pdf_dir  = Path(args.pdf_dir)
    out_file = Path(args.out_file)

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        # Try loading from .env file if present
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not api_key:
        log.error("ANTHROPIC_API_KEY not found. Set it as an environment variable or in a .env file.")
        raise SystemExit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # ── Discover files ────────────────────────────────────────────────────────
    if not pdf_dir.exists():
        log.error(f"PDF directory not found: {pdf_dir}")
        log.info(f"Create it with: mkdir -p {pdf_dir}")
        raise SystemExit(1)

    patterns = ["*.pdf", "*.PDF", "*.html", "*.htm", "*.txt"]
    files = sorted(
        path for pat in patterns for path in pdf_dir.glob(pat)
    )

    if not files:
        log.error(f"No PDF/HTML/TXT files found in {pdf_dir}")
        raise SystemExit(1)

    log.info(f"Found {len(files)} file(s) in {pdf_dir}")

    # ── Load existing results if resuming ─────────────────────────────────────
    existing: dict[str, dict] = {}
    if args.resume and out_file.exists():
        try:
            existing_list = json.loads(out_file.read_text())
            existing = {r["filename"]: r for r in existing_list if "filename" in r}
            log.info(f"Resuming: {len(existing)} file(s) already processed.")
        except Exception as e:
            log.warning(f"Could not load existing output: {e}. Starting fresh.")

    # ── Process each file ─────────────────────────────────────────────────────
    results = []
    skipped = 0
    failed  = 0

    for i, path in enumerate(files, 1):
        log.info(f"[{i}/{len(files)}] {path.name}")

        if args.resume and path.name in existing:
            log.info(f"  → Already processed. Skipping.")
            results.append(existing[path.name])
            skipped += 1
            continue

        # Extract text
        log.info(f"  Extracting text…")
        text = extract_text(path)
        word_count = len(text.split())

        if not text.strip():
            log.error(f"  No text extracted. Skipping.")
            failed += 1
            continue

        log.info(f"  Extracted {word_count:,} words.")

        # Extract date
        date_str = extract_date_from_filename(path)
        log.info(f"  Date: {date_str}")

        # Analyze
        log.info(f"  Calling Claude…")
        result = analyze_document(client, text, path.name)

        if result is None:
            log.error(f"  Analysis failed. Skipping.")
            failed += 1
            continue

        # Attach metadata
        result["date"]     = date_str
        result["filename"] = path.name
        result["words"]    = word_count

        sig   = result.get("signal", "?")
        score = result.get("score", 0.0)
        flags = len(result.get("risk_flags", []))
        log.info(f"  ✓ {sig} ({score:+.2f}) — {flags} risk flag(s)")

        results.append(result)

        # Save incrementally after each file (safe against interruption)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        # Rate limit pause between calls
        if i < len(files):
            log.info(f"  Waiting {args.delay:.0f}s…")
            time.sleep(args.delay)

    # ── Sort by date descending ───────────────────────────────────────────────
    results.sort(key=lambda r: r.get("date", ""), reverse=True)
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("─" * 60)
    log.info(f"Done.  Processed: {len(results) - skipped}  |  Skipped: {skipped}  |  Failed: {failed}")
    log.info(f"Output written to: {out_file}")
    log.info("─" * 60)

    if results:
        avg = sum(r.get("score", 0) for r in results) / len(results)
        log.info(f"Period avg sentiment score: {avg:+.2f}")
        for r in results:
            flags = len(r.get("risk_flags", []))
            flag_str = f"  ⚑ {flags} flag(s)" if flags else ""
            log.info(f"  {r['date']}  {r['signal']:8s}  {r.get('score', 0):+.2f}{flag_str}  — {r['filename']}")


if __name__ == "__main__":
    main()
