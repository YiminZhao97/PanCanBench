#!/usr/bin/env python3
"""Summarize validated grades while excluding explicit missing responses."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


MISSING_RESPONSE_MARKERS = {"NO RESPONSE GENERATED"}


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path}: file is empty")
    data = (
        json.loads(text)
        if text.startswith("[")
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"{path}: expected a JSON array or JSONL objects")
    return data


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grades", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    response_rows = load_json_or_jsonl(args.responses)
    grade_rows = load_json_or_jsonl(args.grades)

    missing: set[tuple[str, str]] = set()
    response_keys: set[tuple[str, str]] = set()
    for row in response_rows:
        question_id = row.get("question_id")
        responses = row.get("responses")
        if not isinstance(question_id, str) or not isinstance(responses, dict):
            raise ValueError("Invalid family response row")
        for model, response in responses.items():
            key = (model, question_id)
            if key in response_keys:
                raise ValueError(f"Duplicate response: {model}/{question_id}")
            response_keys.add(key)
            if isinstance(response, str) and response.strip() in MISSING_RESPONSE_MARKERS:
                missing.add(key)

    grades_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grade_keys: set[tuple[str, str]] = set()
    for row in grade_rows:
        model = row.get("response_model")
        question_id = row.get("question_id")
        percentage = row.get("percentage")
        if not isinstance(model, str) or not isinstance(question_id, str):
            raise ValueError("Invalid normalized grade row")
        if not isinstance(percentage, (int, float)):
            raise ValueError(f"Missing percentage: {model}/{question_id}")
        key = (model, question_id)
        if key in grade_keys:
            raise ValueError(f"Duplicate grade: {model}/{question_id}")
        grade_keys.add(key)
        grades_by_model[model].append(row)

    if grade_keys != response_keys:
        raise ValueError(
            f"Grade/response coverage mismatch: missing grades={sorted(response_keys-grade_keys)}, "
            f"extra grades={sorted(grade_keys-response_keys)}"
        )

    summaries = []
    for model, rows in grades_by_model.items():
        included = [
            row for row in rows if (model, row["question_id"]) not in missing
        ]
        percentages = [float(row["percentage"]) for row in included]
        total_score = sum(row["total_score"] for row in included)
        maximum_score = sum(row["max_possible_score"] for row in included)
        summaries.append(
            {
                "response_model": model,
                "responses_total": len(rows),
                "responses_included_in_mean": len(included),
                "responses_excluded_from_mean": len(rows) - len(included),
                "mean_percentage": statistics.mean(percentages),
                "median_percentage": statistics.median(percentages),
                "minimum_percentage": min(percentages),
                "maximum_percentage": max(percentages),
                "pooled_points_percentage": 100.0 * total_score / maximum_score,
            }
        )

    result = {
        "scoring_summary": "Mean of per-question scores after each question is scaled to 0-100.",
        "missing_response_rule": (
            "Responses explicitly marked NO RESPONSE GENERATED are retained in the grade "
            "file but excluded from model-level averages."
        ),
        "excluded_model_question_pairs": [
            {"response_model": model, "question_id": question_id}
            for model, question_id in sorted(missing)
        ],
        "models": summaries,
    }
    atomic_json(args.output, result)
    print(f"Wrote summary for {len(summaries)} models to {args.output}")


if __name__ == "__main__":
    main()
