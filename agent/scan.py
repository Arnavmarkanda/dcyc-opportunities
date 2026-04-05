"""
DCYC Opportunity Scanner
------------------------
Scans source URLs daily, extracts updated info using Claude API,
and rewrites the data array inside index.html if anything changed.
"""
 
import os
import json
import re
import time
import requests
import anthropic
from bs4 import BeautifulSoup
from datetime import datetime
 
# ── Config ──────────────────────────────────────────────────────────────────
SOURCES_FILE   = "agent/sources.json"
INDEX_FILE     = "index.html"
MAX_CHARS      = 8000   # Max page content sent to Claude per source
SLEEP_BETWEEN  = 2      # Seconds between requests (be polite to servers)
CHANGED        = False  # Global flag — did anything change?
 
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
 
# ── Helpers ──────────────────────────────────────────────────────────────────
def fetch_page(url: str) -> str:
    """Fetch a webpage and return cleaned text content."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (DCYC Scanner Bot; educational use)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove nav, footer, scripts, styles — keep main content
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        return text[:MAX_CHARS]
    except Exception as e:
        print(f"  ⚠️  Could not fetch {url}: {e}")
        return ""
 
 
def extract_with_claude(page_text: str, source: dict) -> dict | None:
    """Send page content to Claude and extract structured opportunity data."""
    if not page_text:
        return None
 
    prompt = f"""You are a data extraction agent for the Douglas County Youth Commission Opportunities Hub.
 
Below is the text content from the webpage for this opportunity:
Title: {source['title']}
Category: {source['category']}
URL: {source['url']}
 
PAGE CONTENT:
{page_text}
 
Extract the current details and return ONLY a valid JSON object with these exact fields.
Do not include any explanation, preamble, or markdown — just the raw JSON object.
 
{{
  "deadline": "Month D, YYYY or null if not found",
  "startDate": "Month D, YYYY or null if not found",
  "endDate": "Month D, YYYY or null if not found",
  "age": "e.g. 16-18 or 16+ or null",
  "cost": "Free or $X,XXX or Paid or null",
  "isPaid": true or false,
  "desc": "2-3 sentence description of the program in your own words. Max 300 chars.",
  "tags": ["tag1", "tag2", "tag3", "tag4"]
}}
 
Rules:
- If a field cannot be determined from the content, use null
- Keep desc under 300 characters
- Only include 3-5 relevant tags
- If the page appears broken or irrelevant, return {{"error": "invalid_source"}}
- Never hallucinate dates or details not present in the source"""
 
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r'^```(?:json)?\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        data = json.loads(raw)
        if "error" in data:
            print(f"  ⚠️  Claude flagged invalid source: {data['error']}")
            return None
        return data
    except Exception as e:
        print(f"  ⚠️  Claude extraction failed: {e}")
        return None
 
 
def read_current_entry(index_html: str, entry_id: str) -> dict | None:
    """Read the current values for an entry from the index.html JS array."""
    pattern = rf'\{{id:"{re.escape(entry_id)}",[^{{}}]*\}}'
    match = re.search(pattern, index_html)
    if not match:
        return None
    entry_str = match.group(0)
 
    def extract_field(field, text):
        m = re.search(rf'{field}:"([^"]*)"', text)
        return m.group(1) if m else None
 
    def extract_bool(field, text):
        m = re.search(rf'{field}:(true|false)', text)
        return m.group(1) == 'true' if m else None
 
    return {
        "raw":      entry_str,
        "deadline": extract_field("deadline", entry_str),
        "start":    extract_field("start", entry_str),
        "end":      extract_field("end", entry_str),
        "age":      extract_field("age", entry_str),
        "cost":     extract_field("cost", entry_str),
        "isPaid":   extract_bool("isPaid", entry_str),
        "desc":     extract_field("desc", entry_str),
    }
 
 
def build_updated_entry(old_entry_str: str, new_data: dict) -> str:
    """Patch the old entry string with updated values from Claude."""
    updated = old_entry_str
 
    def safe_replace(field, new_val):
        if new_val is None:
            return
        new_val_str = str(new_val).replace('"', '\\"')
        updated_local = re.sub(
            rf'{field}:"[^"]*"',
            f'{field}:"{new_val_str}"',
            updated
        )
        return updated_local
 
    mapping = {
        "deadline": new_data.get("deadline"),
        "start":    new_data.get("startDate"),
        "end":      new_data.get("endDate"),
        "age":      new_data.get("age"),
        "cost":     new_data.get("cost"),
        "desc":     new_data.get("desc"),
    }
 
    for field, val in mapping.items():
        if val is not None:
            result = safe_replace(field, val)
            if result:
                updated = result
 
    # Update isPaid boolean
    if new_data.get("isPaid") is not None:
        updated = re.sub(
            r'isPaid:(true|false)',
            f'isPaid:{str(new_data["isPaid"]).lower()}',
            updated
        )
 
    # Update tags
    if new_data.get("tags"):
        tags_str = "[" + ",".join(f'"{t}"' for t in new_data["tags"][:4]) + "]"
        updated = re.sub(r'tags:\[[^\]]*\]', f'tags:{tags_str}', updated)
 
    return updated
 
 
def has_meaningful_change(old: dict, new_data: dict) -> bool:
    """Return True if Claude found something meaningfully different."""
    checks = [
        (old.get("deadline"), new_data.get("deadline")),
        (old.get("start"),    new_data.get("startDate")),
        (old.get("end"),      new_data.get("endDate")),
        (old.get("cost"),     new_data.get("cost")),
    ]
    for old_val, new_val in checks:
        if new_val and old_val and old_val != "N/A" and new_val != old_val:
            return True
    return False
 
 
# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    global CHANGED
 
    print(f"\n{'='*60}")
    print(f"DCYC Scanner starting — {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}\n")
 
    # Load sources
    with open(SOURCES_FILE) as f:
        sources = json.load(f)
 
    # Load index.html
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        index_html = f.read()
 
    updated_html = index_html
 
    for source in sources:
        sid   = source["id"]
        url   = source["url"]
        title = source["title"]
 
        print(f"Scanning: {title}")
        print(f"  URL: {url}")
 
        # Fetch page
        page_text = fetch_page(url)
        if not page_text:
            print(f"  ⏭️  Skipping — could not fetch page\n")
            time.sleep(SLEEP_BETWEEN)
            continue
 
        # Extract with Claude
        new_data = extract_with_claude(page_text, source)
        if not new_data:
            print(f"  ⏭️  Skipping — Claude could not extract data\n")
            time.sleep(SLEEP_BETWEEN)
            continue
 
        # Read current entry from HTML
        current = read_current_entry(updated_html, sid)
        if not current:
            print(f"  ⚠️  Entry ID '{sid}' not found in index.html — skipping\n")
            time.sleep(SLEEP_BETWEEN)
            continue
 
        # Compare and update if changed
        if has_meaningful_change(current, new_data):
            print(f"  ✅  Change detected — updating entry")
            new_entry_str = build_updated_entry(current["raw"], new_data)
            updated_html = updated_html.replace(current["raw"], new_entry_str)
            CHANGED = True
        else:
            print(f"  ✓  No changes detected")
 
        print()
        time.sleep(SLEEP_BETWEEN)
 
    # Write updated index.html if anything changed
    if CHANGED:
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            f.write(updated_html)
        print(f"\n{'='*60}")
        print("✅  Changes found and written to index.html")
        print("    GitHub Actions will now commit and push.")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("✓  No changes detected across all sources.")
        print("   index.html was not modified.")
        print(f"{'='*60}\n")
 
 
if __name__ == "__main__":
    main()
