import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CLEAN_DIR = DATA_DIR / "cleaned"
REGISTRY_PATH = DATA_DIR / "trusted_sources_registry.json"


def load_registry() -> dict[str, Any]:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {entry["key"]: entry for entry in payload.get("sources", []) if entry.get("key")}


def normalize_article_number(value: Any) -> int | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s.isdigit():
        return None
    return int(s)


def build_citation_tag(prefix: str, article_number: int | None, idx: int) -> str:
    if article_number is not None:
        return f"{prefix}-A{article_number:04d}"
    return f"{prefix}-ROW{idx:05d}"


def ingest(input_path: Path, source_key: str, output_name: str) -> Path:
    registry = load_registry()
    if source_key not in registry:
        raise ValueError(f"Unknown source_key '{source_key}'. Add it to trusted_sources_registry.json")
    entry = registry[source_key]

    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Input JSON must be a list of article objects")

    enriched: list[dict[str, Any]] = []
    prefix = entry.get("citation_prefix", "LB-LAW")
    law_code = entry.get("key", source_key)
    law_name = entry.get("official_collection", "Trusted Source")

    for idx, row in enumerate(rows):
        text = str(row.get("text", "")).strip()
        if not text:
            continue

        art_num = normalize_article_number(row.get("article_number"))
        doc_id = row.get("doc_id") or f"{law_code}_{idx}_{art_num if art_num is not None else 'na'}"

        enriched.append(
            {
                "doc_id": doc_id,
                "law_code": row.get("law_code", law_code),
                "law_name": row.get("law_name", law_name),
                "source_file": input_path.name,
                "article_number": str(row.get("article_number", "")).strip(),
                "article_number_normalized": art_num,
                "text": text,
                "is_repealed": bool(row.get("is_repealed", False)),
                "quality_flags": sorted(set(row.get("quality_flags", []))),
                "exclude_from_retrieval": bool(row.get("exclude_from_retrieval", False)),
                "citation_tag": row.get("citation_tag") or build_citation_tag(prefix, art_num, idx),
                "provenance": {
                    "publisher": entry.get("publisher"),
                    "publisher_short": entry.get("publisher_short"),
                    "official_collection": entry.get("official_collection"),
                    "jurisdiction": entry.get("jurisdiction", "Lebanon"),
                    "source_type": entry.get("source_type", "government_website"),
                    "access_url": row.get("source_url") or entry.get("access_url"),
                    "law_landing_url": row.get("law_landing_url") or entry.get("law_landing_url"),
                    "usage_scope": entry.get("usage_scope", "educational_research_only"),
                    "attribution_required": bool(entry.get("attribution_required", True)),
                    "last_verified_utc": entry.get("last_verified_utc") or date.today().isoformat(),
                },
            }
        )

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CLEAN_DIR / output_name
    out_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest trusted-source article JSON with provenance metadata.")
    parser.add_argument("--input", required=True, help="Path to raw/normalized article JSON")
    parser.add_argument("--source-key", required=True, help="Key from trusted_sources_registry.json")
    parser.add_argument("--output", required=True, help="Output filename under Backend/data/cleaned")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = ingest(Path(args.input), args.source_key, args.output)
    print(f"Ingested trusted source rows into: {out}")


if __name__ == "__main__":
    main()
