#!/usr/bin/env python3
"""Merge PanCanBench family response files and validate complete coverage."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path}: file is empty")
    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError(f"{path}: top-level data must be a JSON array or JSONL")
    return rows


def response_text(value: Any) -> str | None:
    """Accept either final text strings or raw claude_family_search.py objects."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict) and isinstance(value.get("text"), str):
        return value["text"].strip() or None
    if value is None:
        return None
    raise ValueError(f"Unsupported response value type: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-model", action="append", default=[])
    parser.add_argument("--expected-questions", type=int, default=282)
    args = parser.parse_args()

    merged: dict[str, dict[str, Any]] = {}
    model_order: list[str] = []

    for path in args.input:
        seen_in_file: set[str] = set()
        for row_number, row in enumerate(load_rows(path), start=1):
            if not isinstance(row, dict):
                raise ValueError(f"{path}: row {row_number} is not an object")
            question_id = row.get("question_id")
            question = row.get("question")
            responses = row.get("responses")
            if not isinstance(question_id, str) or not question_id.startswith("Q"):
                raise ValueError(f"{path}: row {row_number} has invalid question_id")
            if question_id in seen_in_file:
                raise ValueError(f"{path}: duplicate {question_id}")
            seen_in_file.add(question_id)
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{path}: {question_id} has no question text")
            if not isinstance(responses, dict):
                raise ValueError(f"{path}: {question_id} responses must be an object")

            target = merged.setdefault(
                question_id,
                {"question_id": question_id, "question": question.strip(), "responses": {}},
            )
            if target["question"] != question.strip():
                raise ValueError(f"Question text conflict for {question_id} in {path}")

            for model, value in responses.items():
                if not isinstance(model, str) or not model:
                    raise ValueError(f"{path}: {question_id} has an invalid model name")
                if model not in model_order:
                    model_order.append(model)
                text = response_text(value)
                if text is None:
                    continue
                previous = target["responses"].get(model)
                if previous is not None and previous != text:
                    raise ValueError(
                        f"Conflicting nonblank responses for {question_id} / {model}"
                    )
                target["responses"][model] = text

    expected_ids = {f"Q{i}" for i in range(1, args.expected_questions + 1)}
    observed_ids = set(merged)
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids, key=lambda value: int(value[1:]))
        extra = sorted(observed_ids - expected_ids)
        raise ValueError(f"Question coverage mismatch; missing={missing}, extra={extra}")

    expected_models = args.expected_model or model_order
    if len(expected_models) != len(set(expected_models)):
        raise ValueError("--expected-model contains duplicates")
    if set(model_order) != set(expected_models):
        raise ValueError(
            f"Model coverage mismatch; observed={model_order}, expected={expected_models}"
        )

    missing_responses: list[str] = []
    output_rows: list[dict[str, Any]] = []
    for question_id in sorted(merged, key=lambda value: int(value[1:])):
        row = merged[question_id]
        ordered_responses: dict[str, str] = {}
        for model in expected_models:
            value = row["responses"].get(model)
            if not isinstance(value, str) or not value.strip():
                missing_responses.append(f"{question_id}/{model}")
            else:
                ordered_responses[model] = value
        output_rows.append(
            {
                "question_id": question_id,
                "question": row["question"],
                "responses": ordered_responses,
            }
        )

    if missing_responses:
        preview = ", ".join(missing_responses[:20])
        raise ValueError(
            f"Found {len(missing_responses)} blank or missing responses: {preview}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(output_rows, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, args.output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    print(
        f"Validated and wrote {len(output_rows)} questions x "
        f"{len(expected_models)} models = {len(output_rows) * len(expected_models)} "
        f"responses to {args.output}"
    )


if __name__ == "__main__":
    main()
