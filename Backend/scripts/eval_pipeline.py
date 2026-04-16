from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LegalChatbot retrieval/answer behavior.")
    parser.add_argument(
        "--dataset",
        default="Backend/data/eval/legal_eval_sample.jsonl",
        help="Path to JSONL evaluation file",
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000",
        help="Backend base URL",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="top_k sent to /search",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of rows to evaluate (0 = all rows).",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional output metrics JSON path.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def extract_citations(answer: str) -> list[str]:
    return re.findall(r"\[([A-Za-z0-9\-_]+)\]", answer or "")


def run_query(api_base: str, query: str, top_k: int) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/search"
    response = requests.post(url, json={"query": query, "top_k": top_k}, timeout=180)
    response.raise_for_status()
    return response.json()


def is_insufficient_answer(answer: str) -> bool:
    if not answer:
        return False
    markers = [
        "Insufficient",
        "insufficient legal basis",
        "لا يمكنني إعطاء حكم قانوني دقيق",
        "Not explicit in retrieved sources",
    ]
    lowered = answer.lower()
    return any(m.lower() in lowered for m in markers)


def evaluate(rows: list[dict[str, Any]], api_base: str, top_k: int) -> dict[str, Any]:
    total = len(rows)
    failures = 0
    insufficient = 0
    expected_tag_hit_answer = 0
    expected_tag_hit_sources = 0
    checked_with_expected = 0
    failure_messages: list[str] = []

    categories = {
        "covered": {"total": 0, "insufficient": 0, "failures": 0},
        "edge": {"total": 0, "insufficient": 0, "failures": 0},
        "uncovered": {"total": 0, "insufficient": 0, "failures": 0},
    }

    for i, row in enumerate(rows, start=1):
        query = str(row.get("query", "")).strip()
        if not query:
            continue

        category = str(row.get("category", "covered")).strip().lower()
        if category not in categories:
            category = "covered"
        categories[category]["total"] += 1

        expected_tags = set(row.get("expected_citation_tags", []) or [])

        try:
            payload = run_query(api_base, query, top_k)
        except Exception as exc:
            failures += 1
            categories[category]["failures"] += 1
            failure_messages.append(f"{i}. {query} -> {exc}")
            continue

        answer = str(payload.get("answer", ""))
        sources = payload.get("sources", []) or []
        cited = set(extract_citations(answer))
        source_tags = {s.get("citation_tag") for s in sources if isinstance(s, dict) and s.get("citation_tag")}

        if is_insufficient_answer(answer):
            insufficient += 1
            categories[category]["insufficient"] += 1

        if expected_tags:
            checked_with_expected += 1
            if cited & expected_tags:
                expected_tag_hit_answer += 1
            if source_tags & expected_tags:
                expected_tag_hit_sources += 1

    metrics = {
        "total_queries": total,
        "failures": failures,
        "insufficient_count": insufficient,
        "insufficient_rate": (insufficient / total) if total else 0.0,
        "checked_with_expected": checked_with_expected,
        "expected_citation_hit_answer_count": expected_tag_hit_answer,
        "expected_citation_hit_answer_rate": (expected_tag_hit_answer / checked_with_expected) if checked_with_expected else 0.0,
        "expected_citation_hit_sources_count": expected_tag_hit_sources,
        "expected_citation_hit_sources_rate": (expected_tag_hit_sources / checked_with_expected) if checked_with_expected else 0.0,
        "categories": {},
        "failure_messages": failure_messages,
    }

    for cat, values in categories.items():
        cat_total = values["total"]
        metrics["categories"][cat] = {
            "total": cat_total,
            "failures": values["failures"],
            "insufficient_count": values["insufficient"],
            "insufficient_rate": (values["insufficient"] / cat_total) if cat_total else 0.0,
        }

    return metrics


def print_summary(metrics: dict[str, Any]) -> None:
    print("Evaluation Summary")
    print(f"- Total queries: {metrics['total_queries']}")
    print(f"- Failures: {metrics['failures']}")
    print(f"- Insufficient-basis responses: {metrics['insufficient_count']} ({metrics['insufficient_rate']:.1%})")
    print(
        f"- Expected citation hit in answer text: "
        f"{metrics['expected_citation_hit_answer_count']}/{metrics['checked_with_expected']} "
        f"({metrics['expected_citation_hit_answer_rate']:.1%})"
    )
    print(
        f"- Expected citation hit in source cards: "
        f"{metrics['expected_citation_hit_sources_count']}/{metrics['checked_with_expected']} "
        f"({metrics['expected_citation_hit_sources_rate']:.1%})"
    )

    print("\nBy Category")
    for cat in ["covered", "edge", "uncovered"]:
        c = metrics["categories"].get(cat, {})
        print(
            f"- {cat}: total={c.get('total', 0)} "
            f"failures={c.get('failures', 0)} "
            f"insufficient={c.get('insufficient_count', 0)} ({c.get('insufficient_rate', 0.0):.1%})"
        )

    if metrics["failure_messages"]:
        print("\nFailures:")
        for msg in metrics["failure_messages"][:30]:
            print(f"- {msg}")


def main() -> None:
    args = parse_args()
    dataset_rows = load_jsonl(Path(args.dataset))
    if args.limit > 0:
        dataset_rows = dataset_rows[: args.limit]

    if not dataset_rows:
        print("No evaluation rows found.")
        return

    metrics = evaluate(dataset_rows, args.api_base, args.top_k)
    print_summary(metrics)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved metrics JSON: {out_path}")


if __name__ == "__main__":
    main()
