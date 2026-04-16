import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CLEAN_DIR = DATA_DIR / "cleaned"
REGISTRY_PATH = DATA_DIR / "trusted_sources_registry.json"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

DIGIT_TRANS = str.maketrans("\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669", "0123456789")
REPEALED_PATTERNS = [
    "\u0645\u0644\u063a\u0627\u0629",
    "\u0645\u0644\u063a\u0649",
    "\u0623\u0644\u063a\u064a\u062a",
    "\u0627\u0644\u063a\u064a",
    "\u0623\u0628\u0637\u0644",
    "\u0627\u0628\u0637\u0644\u062a",
    "\u0646\u0633\u062e\u062a",
]

LAW_CONFIGS: list[dict[str, Any]] = [
    {
        "code": "penal",
        "law_name": "Lebanese Penal Code",
        "source_file": "penal_code_articles_ocr.json",
        "expected_min": 1,
        "expected_max": 772,
        "registry_key": "lebanon_penal_code",
    },
    {
        "code": "commercial",
        "law_name": "Lebanese Commercial Code",
        "source_file": "tijara_code_articles_ocr.json",
        "expected_min": 1,
        "expected_max": 1000,
        "registry_key": "lebanon_commercial_code",
    },
    {
        "code": "civil_procedure",
        "law_name": "Code of Civil Procedure",
        "source_file": "muhakamat-madaniya_code_articles_ocr.json",
        "expected_min": 1,
        "expected_max": 1033,
        "registry_key": "lebanon_civil_procedure_code",
    },
    {
        "code": "civil",
        "law_name": "Lebanese Code of Obligations and Contracts",
        "source_file": "civil_code_articles_ocr.json",
        "expected_min": 1,
        "expected_max": 1105,
        "registry_key": "lebanon_obligations_contracts_code",
    },
]


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing trusted source registry: {REGISTRY_PATH}")
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    entries = payload.get("sources", [])
    by_key = {e["key"]: e for e in entries if isinstance(e, dict) and e.get("key")}
    if not by_key:
        raise ValueError("trusted_sources_registry.json has no valid 'sources' entries")
    return by_key


def normalize_digits(s: str) -> str:
    return s.translate(DIGIT_TRANS)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_int(raw: str) -> int | None:
    normalized = normalize_digits(raw).strip()
    if not normalized or not normalized.isdigit():
        return None
    return int(normalized)


def repair_outlier(raw: str, prev_norm: int | None, lo: int, hi: int) -> int | None:
    digits = normalize_digits(raw).strip()
    if not digits.isdigit() or len(digits) < 4:
        return None

    candidates: set[int] = set()
    for i in range(len(digits)):
        candidate = (digits[:i] + digits[i + 1 :]).lstrip("0")
        if candidate.isdigit():
            n = int(candidate)
            if lo <= n <= hi:
                candidates.add(n)

    if not candidates:
        return None
    if prev_norm is None:
        return min(candidates)
    return min(candidates, key=lambda x: abs(x - (prev_norm + 1)))


def is_repealed(text: str) -> bool:
    return any(p in text for p in REPEALED_PATTERNS)


def build_citation_tag(registry_entry: dict[str, Any], article_number: int | None, row_idx: int) -> str:
    base = registry_entry.get("citation_prefix", "LB-LAW")
    if isinstance(article_number, int):
        return f"{base}-A{article_number:04d}"
    return f"{base}-ROW{row_idx:05d}"


def process_law(cfg: dict[str, Any], registry_by_key: dict[str, Any]) -> dict[str, Any]:
    path = DATA_DIR / cfg["source_file"]
    rows = json.loads(path.read_text(encoding="utf-8"))
    registry_entry = registry_by_key[cfg["registry_key"]]

    processed: list[dict[str, Any]] = []
    prev_norm: int | None = None

    for idx, row in enumerate(rows):
        raw_number = str(row.get("article_number", "")).strip()
        text_clean = normalize_ws(str(row.get("text", "")))

        flags: list[str] = []
        n = parse_int(raw_number)

        if n is None:
            flags.append("non_numeric_article_number")
        else:
            if n < cfg["expected_min"] or n > cfg["expected_max"]:
                repaired = repair_outlier(raw_number, prev_norm, cfg["expected_min"], cfg["expected_max"])
                if repaired is not None:
                    n = repaired
                    flags.append("corrected_ocr_number")
                else:
                    flags.append("article_number_out_of_expected_range")

            if prev_norm is not None:
                if n <= 9 and prev_norm >= 100:
                    n = prev_norm + 1
                    flags.append("corrected_contextual_sequence_number")
                elif n < prev_norm - 50:
                    flags.append("sequence_reset_or_number_jump")

        if not text_clean:
            flags.append("empty_text")
        elif len(text_clean) < 30:
            flags.append("very_short_text")

        repealed = is_repealed(text_clean)
        normalized_for_id = str(n) if n is not None else normalize_digits(raw_number) or "unknown"
        doc_id = f"{cfg['code']}_{idx}_{normalized_for_id}"

        processed.append(
            {
                "doc_id": doc_id,
                "law_code": cfg["code"],
                "law_name": cfg["law_name"],
                "source_file": cfg["source_file"],
                "article_number": raw_number,
                "article_number_normalized": n,
                "text": text_clean,
                "is_repealed": repealed,
                "quality_flags": flags,
                "citation_tag": build_citation_tag(registry_entry, n, idx),
                "provenance": {
                    "publisher": registry_entry.get("publisher"),
                    "publisher_short": registry_entry.get("publisher_short"),
                    "official_collection": registry_entry.get("official_collection"),
                    "jurisdiction": registry_entry.get("jurisdiction", "Lebanon"),
                    "source_type": registry_entry.get("source_type", "government_website"),
                    "access_url": registry_entry.get("access_url"),
                    "law_landing_url": registry_entry.get("law_landing_url"),
                    "usage_scope": registry_entry.get("usage_scope", "educational_research_only"),
                    "attribution_required": bool(registry_entry.get("attribution_required", True)),
                    "last_verified_utc": registry_entry.get("last_verified_utc"),
                },
            }
        )

        if isinstance(n, int):
            prev_norm = n

    num_counter = Counter(
        r["article_number_normalized"]
        for r in processed
        if isinstance(r["article_number_normalized"], int)
    )
    text_counter = Counter(
        hashlib.md5(r["text"].encode("utf-8")).hexdigest() for r in processed if r["text"]
    )

    for r in processed:
        n = r["article_number_normalized"]
        if isinstance(n, int) and num_counter[n] > 1:
            r["quality_flags"].append("duplicate_article_number")

        text_hash = hashlib.md5(r["text"].encode("utf-8")).hexdigest() if r["text"] else ""
        if text_hash and text_counter[text_hash] > 1:
            r["quality_flags"].append("duplicate_text")

        r["quality_flags"] = sorted(set(r["quality_flags"]))
        severe = {"empty_text", "non_numeric_article_number", "article_number_out_of_expected_range"}
        r["exclude_from_retrieval"] = any(flag in severe for flag in r["quality_flags"])

    out_clean = CLEAN_DIR / cfg["source_file"].replace("_ocr.json", "_clean.json")
    out_clean.write_text(json.dumps(processed, ensure_ascii=False, indent=2), encoding="utf-8")

    retrieval = [
        r
        for r in processed
        if not r["exclude_from_retrieval"] and not r["is_repealed"] and len(r["text"]) >= 30
    ]
    out_retrieval = CLEAN_DIR / cfg["source_file"].replace("_ocr.json", "_retrieval.json")
    out_retrieval.write_text(json.dumps(retrieval, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "law_code": cfg["code"],
        "source_file": cfg["source_file"],
        "rows_total": len(processed),
        "rows_retrieval": len(retrieval),
        "repealed": sum(1 for r in processed if r["is_repealed"]),
        "excluded": sum(1 for r in processed if r["exclude_from_retrieval"]),
        "with_flags": sum(1 for r in processed if r["quality_flags"]),
    }


def main() -> None:
    registry_by_key = load_registry()
    report = [process_law(cfg, registry_by_key) for cfg in LAW_CONFIGS]

    combined_retrieval: list[dict[str, Any]] = []
    for cfg in LAW_CONFIGS:
        path = CLEAN_DIR / cfg["source_file"].replace("_ocr.json", "_retrieval.json")
        combined_retrieval.extend(json.loads(path.read_text(encoding="utf-8")))

    combined_path = CLEAN_DIR / "combined_legal_articles_retrieval.json"
    combined_path.write_text(json.dumps(combined_retrieval, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = CLEAN_DIR / "data_quality_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Data prep completed.")
    print(f"Combined retrieval rows: {len(combined_retrieval)}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
