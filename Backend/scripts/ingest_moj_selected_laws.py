from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CLEAN_DIR = DATA_DIR / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://www.justice.gov.lb"
INDEX_URL = f"{BASE}/laws/2"

RAW_OUT = CLEAN_DIR / "moj_selected_laws_raw_pages.json"
FULL_OUT = CLEAN_DIR / "moj_selected_laws_articles_full.json"
NEW_ONLY_OUT = CLEAN_DIR / "moj_selected_laws_articles_new_only.json"
REPORT_OUT = CLEAN_DIR / "moj_selected_laws_ingestion_report.json"

EXISTING_CORPUS = CLEAN_DIR / "combined_legal_articles_retrieval.json"

AR_SELECTED_LAWS = "\u0642\u0648\u0627\u0646\u064a\u0646 \u0645\u062e\u062a\u0627\u0631\u0629"
AR_BACK = "\u0631\u062c\u0648\u0639"
AR_LAWS = "\u0627\u0644\u0642\u0648\u0627\u0646\u064a\u0646"
AR_LAW_NO = "\u0642\u0627\u0646\u0648\u0646 \u0631\u0642\u0645"
AR_SINGLE_ARTICLE = "\u0645\u0627\u062f\u0629 \u0648\u062d\u064a\u062f\u0629"
AR_ARTICLE = "\u0627\u0644\u0645\u0627\u062f\u0629"
AR_MADDA = "\u0645\u0627\u062f\u0629"
AR_FOOTER_MARKERS = [
    "\u062e\u062f\u0645\u0627\u062a \u0627\u0644\u0645\u0648\u0627\u0637\u0646\u064a\u0646",
    "\u0627\u0642\u062a\u0631\u0627\u062d\u0627\u062a",
    "\u0627\u062a\u0635\u0644 \u0628\u0646\u0627",
    "\u062d\u0642\u0648\u0642 \u0627\u0644\u062a\u0623\u0644\u064a\u0641",
]

DIGIT_TRANS = str.maketrans("\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669", "0123456789")

ORDINAL_MAP = {
    "\u0627\u0644\u0623\u0648\u0644\u0649": 1,
    "\u0627\u0644\u0627\u0648\u0644\u0649": 1,
    "\u0627\u0644\u062b\u0627\u0646\u064a\u0629": 2,
    "\u0627\u0644\u062b\u0627\u0644\u062b\u0629": 3,
    "\u0627\u0644\u0631\u0627\u0628\u0639\u0629": 4,
    "\u0627\u0644\u062e\u0627\u0645\u0633\u0629": 5,
    "\u0627\u0644\u0633\u0627\u062f\u0633\u0629": 6,
    "\u0627\u0644\u0633\u0627\u0628\u0639\u0629": 7,
    "\u0627\u0644\u062b\u0627\u0645\u0646\u0629": 8,
    "\u0627\u0644\u062a\u0627\u0633\u0639\u0629": 9,
    "\u0627\u0644\u0639\u0627\u0634\u0631\u0629": 10,
    "\u0648\u062d\u064a\u062f\u0629": 1,
}


@dataclass
class LawPage:
    law_id: str
    url: str
    title: str
    raw_text: str


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_digits(s: str) -> str:
    return s.translate(DIGIT_TRANS)


def clean_html_to_text(raw_html: str) -> str:
    text = raw_html
    for pat in [r"<script[\s\S]*?</script>", r"<style[\s\S]*?</style>"]:
        text = re.sub(pat, " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return normalize_ws(text)


def fetch_law_links(session: requests.Session) -> list[tuple[str, str]]:
    r = session.get(INDEX_URL, timeout=40)
    r.raise_for_status()
    links = sorted(set(re.findall(r'href="([^\"]*/law-details/(\d+)/2[^\"]*)"', r.text)))
    out: list[tuple[str, str]] = []
    for href, law_id in links:
        normalized_href = href.replace("/index.php/", "/")
        out.append((law_id, urljoin(INDEX_URL, normalized_href)))
    return out


def extract_title(text: str, law_id: str) -> str:
    pattern = re.escape(AR_SELECTED_LAWS) + r"\s*>\s*(.*?)\s+" + re.escape(AR_BACK)
    m = re.search(pattern, text)
    if m:
        return normalize_ws(m.group(1))

    pattern2 = re.escape(AR_LAWS) + r"\s+(.*?)\s+" + re.escape(AR_LAW_NO)
    m2 = re.search(pattern2, text)
    if m2:
        return normalize_ws(m2.group(1))

    return f"MoJ Selected Law {law_id}"


def extract_law_body(text: str) -> str:
    anchor_candidates = [
        AR_SINGLE_ARTICLE,
        f"{AR_ARTICLE} \u0627\u0644\u0623\u0648\u0644\u0649",
        f"{AR_ARTICLE} \u0627\u0644\u0627\u0648\u0644\u0649",
        f"{AR_ARTICLE} 1",
        f"{AR_ARTICLE} \u0661",
    ]
    starts = [text.find(a) for a in anchor_candidates if text.find(a) != -1]
    if not starts:
        return ""
    start = min(starts)

    end_positions = [text.find(m, start + 1) for m in AR_FOOTER_MARKERS if text.find(m, start + 1) != -1]
    end = min(end_positions) if end_positions else len(text)
    return normalize_ws(text[start:end])


def article_token_to_number(token: str) -> int | None:
    t = normalize_digits(token).strip()
    if t.isdigit():
        return int(t)
    return ORDINAL_MAP.get(t)


def split_articles(body: str) -> list[tuple[str, str]]:
    if not body:
        return []

    token_alt = r"[0-9\u0660-\u0669]+|[^\s:：-]+"
    pattern = re.compile(rf"(?:{re.escape(AR_ARTICLE)}|{re.escape(AR_MADDA)})\s*(?:{token_alt})\s*[:：-]?")
    matches = list(pattern.finditer(body))

    rows: list[tuple[str, str]] = []
    for idx, m in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        header = normalize_ws(m.group(0))
        content = normalize_ws(body[m.end() : end])
        rows.append((header, content))
    return rows


def parse_article_number(header: str) -> int | None:
    token_alt = r"[0-9\u0660-\u0669]+|\S+"
    m = re.search(rf"(?:{re.escape(AR_ARTICLE)}|{re.escape(AR_MADDA)})\s*({token_alt})", header)
    if not m:
        return None
    return article_token_to_number(m.group(1))


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_ws(text).encode("utf-8")).hexdigest()


def load_existing_hashes() -> set[str]:
    if not EXISTING_CORPUS.exists():
        return set()
    data = json.loads(EXISTING_CORPUS.read_text(encoding="utf-8"))
    return {text_hash(str(r.get("text", ""))) for r in data if r.get("text")}


def build_article_row(page: LawPage, idx: int, header: str, content: str, existing_hashes: set[str]) -> dict[str, Any]:
    number = parse_article_number(header)
    text = normalize_ws(f"{header} {content}")
    th = text_hash(text)
    duplicate = th in existing_hashes

    art_label = str(number) if number is not None else f"row{idx+1}"
    citation_tag = f"LB-MOJ-SL-L{page.law_id}-A{art_label}"

    return {
        "doc_id": f"moj_selected_{page.law_id}_{idx+1}",
        "law_code": "moj_selected_laws",
        "law_name": page.title,
        "source_file": "moj_selected_laws_web_ingest",
        "article_number": art_label,
        "article_number_normalized": number,
        "text": text,
        "is_repealed": False,
        "quality_flags": ["duplicate_text_in_existing_corpus"] if duplicate else [],
        "exclude_from_retrieval": duplicate or len(text) < 30,
        "citation_tag": citation_tag,
        "provenance": {
            "publisher": "Ministry of Justice - Republic of Lebanon",
            "publisher_short": "MoJ Lebanon",
            "official_collection": "Selected Laws",
            "jurisdiction": "Lebanon",
            "source_type": "government_website",
            "access_url": "https://www.justice.gov.lb/laws/2",
            "law_landing_url": page.url,
            "usage_scope": "educational_research_only",
            "attribution_required": True,
            "last_verified_utc": "2026-04-16",
        },
        "ingestion": {
            "law_id": page.law_id,
            "article_header": header,
            "text_hash": th,
            "duplicate_in_existing_corpus": duplicate,
        },
    }


def fetch_pages(session: requests.Session, law_links: list[tuple[str, str]]) -> list[LawPage]:
    pages: list[LawPage] = []
    for law_id, url in law_links:
        r = session.get(url, timeout=40)
        r.raise_for_status()
        text = clean_html_to_text(r.text)
        title = extract_title(text, law_id)
        body = extract_law_body(text)
        pages.append(LawPage(law_id=law_id, url=url, title=title, raw_text=body))
    return pages


def main() -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "LegalChatbot-ResearchIngest/1.0"})

    law_links = fetch_law_links(session)
    pages = fetch_pages(session, law_links)

    RAW_OUT.write_text(
        json.dumps(
            [
                {
                    "law_id": p.law_id,
                    "url": p.url,
                    "title": p.title,
                    "body_length": len(p.raw_text),
                    "preview": p.raw_text[:500],
                }
                for p in pages
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    existing_hashes = load_existing_hashes()
    all_rows: list[dict[str, Any]] = []
    for p in pages:
        for idx, (header, content) in enumerate(split_articles(p.raw_text)):
            all_rows.append(build_article_row(p, idx, header, content, existing_hashes))

    FULL_OUT.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    new_only = [r for r in all_rows if not r["exclude_from_retrieval"]]
    NEW_ONLY_OUT.write_text(json.dumps(new_only, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "source": INDEX_URL,
        "laws_found": len(law_links),
        "laws_fetched": len(pages),
        "articles_total": len(all_rows),
        "articles_new_only": len(new_only),
        "articles_duplicate_existing": sum(1 for r in all_rows if r["ingestion"]["duplicate_in_existing_corpus"]),
        "outputs": {
            "raw_pages": str(RAW_OUT),
            "articles_full": str(FULL_OUT),
            "articles_new_only": str(NEW_ONLY_OUT),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
