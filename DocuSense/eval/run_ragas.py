"""RAGAS evaluation harness. Usage: python -m eval.run_ragas"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from shared.types import Citation, QueryResponse

load_dotenv()

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
REPORT_PATH = Path(__file__).parent / "eval_report.md"

_EMPTY_RESPONSE = QueryResponse(answer="", citations=[], confidence=0.0, verified=False)


def _load_eval_set() -> list[dict[str, Any]]:
    with open(EVAL_SET_PATH) as f:
        data: dict[str, Any] = json.load(f)
    return list(data["examples"])


def _run_pipeline(question: str) -> QueryResponse:
    from agents.graph import answer

    return answer(question)


def _build_ragas_dataset(
    examples: list[dict[str, Any]],
    results: list[QueryResponse],
) -> Any:
    from datasets import Dataset

    rows: dict[str, list[Any]] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for ex, res in zip(examples, results):
        rows["question"].append(ex["question"])
        rows["answer"].append(res["answer"])
        rows["contexts"].append(
            [c["text_excerpt"] for c in res["citations"]] or [""]
        )
        rows["ground_truth"].append(ex.get("ground_truth", ""))

    return Dataset.from_dict(rows)


def _write_report(
    metrics: dict[str, float],
    examples: list[dict[str, Any]],
    results: list[QueryResponse],
) -> None:
    lines: list[str] = [
        "# DocuSense Evaluation Report\n",
        "## Aggregate Metrics\n",
        "| Metric | Score |",
        "|--------|-------|",
    ]
    for name, score in metrics.items():
        lines.append(f"| {name} | {score:.3f} |")

    lines += [
        "",
        "## Per-Question Results\n",
        "| ID | Doc Type | Question | Answer | GT | Verified |",
        "|----|----------|----------|--------|----|----------|",
    ]
    for ex, res in zip(examples, results):
        q = ex["question"][:60].replace("|", "/")
        a = res["answer"][:60].replace("|", "/")
        gt = (ex.get("ground_truth") or "")[:40].replace("|", "/")
        verified = "✓" if res["verified"] else "✗"
        lines.append(
            f"| {ex['id']} | {ex['doc_type']} | {q} | {a} | {gt} | {verified} |"
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Report written to {REPORT_PATH}")


def main() -> None:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    examples = _load_eval_set()
    print(f"Running pipeline on {len(examples)} examples…")

    results: list[QueryResponse] = []
    for ex in examples:
        print(f"  [{ex['id']}] {ex['question'][:60]}")
        try:
            res = _run_pipeline(ex["question"])
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            res = _EMPTY_RESPONSE
        results.append(res)

    dataset = _build_ragas_dataset(examples, results)

    score = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    metrics: dict[str, float] = {
        "faithfulness": float(score["faithfulness"]),
        "answer_relevancy": float(score["answer_relevancy"]),
        "context_precision": float(score["context_precision"]),
        "context_recall": float(score["context_recall"]),
    }

    print("\nResults:")
    for name, val in metrics.items():
        print(f"  {name}: {val:.3f}")

    _write_report(metrics, examples, results)


if __name__ == "__main__":
    main()
