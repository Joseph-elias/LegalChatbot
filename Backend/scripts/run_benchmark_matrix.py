from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark matrix across baseline/enhanced configs.")
    parser.add_argument("--dataset", default="Backend/data/eval/legal_eval_v1.jsonl")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", default="Backend/data/eval/results_v1")
    return parser.parse_args()


def run_eval(
    *,
    name: str,
    env_overrides: dict[str, str],
    dataset: str,
    api_base: str,
    top_k: int,
    limit: int,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{name}.json"
    cmd = [
        sys.executable,
        "Backend/scripts/eval_pipeline.py",
        "--dataset",
        dataset,
        "--api-base",
        api_base,
        "--top-k",
        str(top_k),
        "--out-json",
        str(out_json),
    ]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])

    env = os.environ.copy()
    env.update(env_overrides)

    print(f"\n=== Running: {name} ===")
    print(f"Overrides: {env_overrides}")
    subprocess.run(cmd, check=True, env=env)
    return json.loads(out_json.read_text(encoding="utf-8"))


def build_report(metrics_by_run: dict[str, dict[str, Any]], out_dir: Path) -> Path:
    lines = ["# Benchmark Report v1", ""]
    lines.append("| Run | Failures | Insufficient | Citation Hit (Answer) | Citation Hit (Sources) | Covered Insufficient | Uncovered Insufficient |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for name, m in metrics_by_run.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(m.get("failures", 0)),
                    f"{m.get('insufficient_rate', 0.0):.1%}",
                    f"{m.get('expected_citation_hit_answer_rate', 0.0):.1%}",
                    f"{m.get('expected_citation_hit_sources_rate', 0.0):.1%}",
                    f"{m.get('categories', {}).get('covered', {}).get('insufficient_rate', 0.0):.1%}",
                    f"{m.get('categories', {}).get('uncovered', {}).get('insufficient_rate', 0.0):.1%}",
                ]
            )
            + " |"
        )

    report_path = out_dir / "report_v1.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    matrix = {
        "baseline_dense": {
            "RETRIEVAL_MODE": "dense",
            "RERANKER_ENABLED": "0",
            "SELF_CHECK_ENABLED": "1",
            "MIN_ANSWER_CONFIDENCE": "0.35",
        },
        "enhanced_hybrid_conf_030": {
            "RETRIEVAL_MODE": "hybrid",
            "RERANKER_ENABLED": "1",
            "SELF_CHECK_ENABLED": "1",
            "MIN_ANSWER_CONFIDENCE": "0.30",
        },
        "enhanced_hybrid_conf_035": {
            "RETRIEVAL_MODE": "hybrid",
            "RERANKER_ENABLED": "1",
            "SELF_CHECK_ENABLED": "1",
            "MIN_ANSWER_CONFIDENCE": "0.35",
        },
        "enhanced_hybrid_conf_040": {
            "RETRIEVAL_MODE": "hybrid",
            "RERANKER_ENABLED": "1",
            "SELF_CHECK_ENABLED": "1",
            "MIN_ANSWER_CONFIDENCE": "0.40",
        },
    }

    metrics_by_run: dict[str, dict[str, Any]] = {}
    for name, overrides in matrix.items():
        metrics_by_run[name] = run_eval(
            name=name,
            env_overrides=overrides,
            dataset=args.dataset,
            api_base=args.api_base,
            top_k=args.top_k,
            limit=args.limit,
            out_dir=out_dir,
        )

    report = build_report(metrics_by_run, out_dir)
    print(f"\nSaved benchmark report: {report}")


if __name__ == "__main__":
    main()
