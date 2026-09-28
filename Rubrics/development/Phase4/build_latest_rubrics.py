#!/usr/bin/env python3
"""Build the reviewed PanCanBench rubric JSON from the five Phase 4 folds.

Each Markdown ``Final Grades`` value becomes the rubric weight in the legacy
PanCanBench JSON schema. Positive criteria use ``min_points=0`` and
``max_points=grade``; negative criteria use ``min_points=grade`` and
``max_points=0``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


HEADING = re.compile(r"^###(?:\s+Question)?\s+(\d+)\s*$")
QUESTION_TEXT = re.compile(r"^\*\*Question:\*\*\s*(.*?)\s*$")
EXPECTED_QUESTION_NUMBERS = list(range(1, 283))
EXPECTED_TOTAL_ITEMS = 3639
EXPECTED_FOLD_ITEMS = {1: 558, 2: 809, 3: 816, 4: 674, 5: 782}


ROOT = Path(__file__).resolve().parents[3]


def default_source_dir() -> Path:
    return ROOT / "Data/Rubrics/reviewed_phase4"


def default_check_file() -> Path:
    return ROOT / "Data/human_ratings/final/rubrics_for_human_grading.md"


def default_output_dir() -> Path:
    return ROOT / "Outputs/rubrics"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_source_row(
    line: str, path: Path, line_number: int
) -> dict[str, Any] | None:
    if not line.startswith("|"):
        return None
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if not cells or not cells[0].isdigit():
        return None
    if len(cells) != 5:
        raise ValueError(
            f"{path.name}, line {line_number}: expected five Markdown columns"
        )
    description = cells[1]
    if not description:
        raise ValueError(f"{path.name}, line {line_number}: empty rubric description")
    if not cells[4]:
        raise ValueError(
            f"{path.name}, line {line_number}: Q/item row has no Final Grades value"
        )
    try:
        final_grade = int(cells[4])
    except ValueError as exc:
        raise ValueError(
            f"{path.name}, line {line_number}: Final Grades must be an integer"
        ) from exc
    if not -10 <= final_grade <= 10:
        raise ValueError(
            f"{path.name}, line {line_number}: Final Grades is outside -10 to 10"
        )
    return {
        "item_number": int(cells[0]),
        "description": description,
        "final_grade": final_grade,
    }


def parse_fold(path: Path, fold_number: int) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        heading = HEADING.match(line)
        if heading:
            if current is not None:
                questions.append(current)
            current = {
                "question_number": int(heading.group(1)),
                "question_text": "",
                "source_rows": [],
                "source_file": path.name,
            }
            continue
        if current is None:
            continue
        question = QUESTION_TEXT.match(line)
        if question:
            current["question_text"] = question.group(1).strip()
            continue
        row = parse_source_row(line, path, line_number)
        if row is not None:
            current["source_rows"].append(row)
    if current is not None:
        questions.append(current)

    for question in questions:
        number = int(question["question_number"])
        if not question["question_text"]:
            raise ValueError(f"{path.name}: Q{number} has no question text")
        actual = [row["item_number"] for row in question["source_rows"]]
        expected = list(range(1, len(actual) + 1))
        if actual != expected:
            raise ValueError(f"{path.name}: Q{number} numbering is not consecutive")

    actual_items = sum(len(question["source_rows"]) for question in questions)
    expected_items = EXPECTED_FOLD_ITEMS[fold_number]
    if actual_items != expected_items:
        raise ValueError(
            f"{path.name}: expected {expected_items} items, found {actual_items}"
        )
    return questions


def json_points(final_grade: int) -> tuple[int, int]:
    if final_grade < 0:
        return final_grade, 0
    return 0, final_grade


def parse_human_check(path: Path) -> dict[int, list[tuple[int, str]]]:
    row_pattern = re.compile(
        r"^\|\s*Q(\d+)\s*\|\s*(\d+)\s*\|\s*(.*?)\s*\|\s*$"
    )
    check: dict[int, list[tuple[int, str]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = row_pattern.match(line)
        if match:
            check.setdefault(int(match.group(1)), []).append(
                (int(match.group(2)), match.group(3).strip())
            )
    if not check:
        raise ValueError(f"{path}: no human-check rubric rows found")
    return check


def build(source_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_paths = [
        source_dir / f"fold{fold}_merged_rubrics_final_version.md"
        for fold in range(1, 6)
    ]
    missing = [str(path) for path in source_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Phase 4 folds: " + ", ".join(missing))

    source_questions = [
        question
        for fold, path in enumerate(source_paths, 1)
        for question in parse_fold(path, fold)
    ]
    source_questions.sort(key=lambda question: int(question["question_number"]))
    actual_numbers = [int(q["question_number"]) for q in source_questions]
    if actual_numbers != EXPECTED_QUESTION_NUMBERS:
        raise ValueError("Sources must contain each question from 1 through 282 once")

    questions = []
    grades: Counter[int] = Counter()
    for question in source_questions:
        rubric_items = []
        for row in question["source_rows"]:
            grade = int(row["final_grade"])
            minimum, maximum = json_points(grade)
            grades[grade] += 1
            rubric_items.append(
                {
                    "item_number": int(row["item_number"]),
                    "description": str(row["description"]),
                    "min_points": minimum,
                    "max_points": maximum,
                }
            )
        questions.append(
            {
                "question_number": int(question["question_number"]),
                "question_text": str(question["question_text"]),
                "rubric_items": rubric_items,
            }
        )

    item_count = sum(len(q["rubric_items"]) for q in questions)
    if item_count != EXPECTED_TOTAL_ITEMS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_ITEMS} final items, found {item_count}"
        )
    return {"questions": questions}, {
        "source_paths": source_paths,
        "item_count": item_count,
        "grade_counts": grades,
    }


def validate_human_subset(data: dict[str, Any], path: Path) -> tuple[int, int]:
    expected = parse_human_check(path)
    by_question = {int(q["question_number"]): q for q in data["questions"]}
    checked_items = 0
    for number, expected_rows in expected.items():
        actual_rows = [
            (int(row["item_number"]), str(row["description"]))
            for row in by_question[number]["rubric_items"]
        ]
        if actual_rows != expected_rows:
            raise ValueError(f"Generated Q{number} differs from the human rubric")
        checked_items += len(expected_rows)
    return len(expected), checked_items


def write_report(
    path: Path,
    output_path: Path,
    check_path: Path,
    metadata: dict[str, Any],
    checked_questions: int,
    checked_items: int,
) -> None:
    source_lines = "\n".join(
        f"- `{source.name}`: `{sha256(source)}`"
        for source in metadata["source_paths"]
    )
    grade_lines = ", ".join(
        f"{grade}: {count}"
        for grade, count in sorted(metadata["grade_counts"].items())
    )
    report = f"""# Final Phase 4 rubric JSON validation

- Output: `{output_path.name}`
- Questions: 282
- Rubric items: {metadata['item_count']}
- Point policy: reviewed weighted criteria from `Final Grades`
- Human subset: {checked_items}/{checked_items} exact item-number-and-text matches across {checked_questions} questions
- Human check file: `{check_path}`
- Output SHA-256: `{sha256(output_path)}`
- Grade distribution: {grade_lines}

Negative grades are encoded as `min_points=grade, max_points=0`; nonnegative
grades are encoded as `min_points=0, max_points=grade`.

## Phase 4 source SHA-256 hashes

{source_lines}
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=default_source_dir())
    parser.add_argument("--check-file", type=Path, default=default_check_file())
    parser.add_argument("--corrections", type=Path, default=ROOT / "Data/Rubrics/final_wording_corrections.json", help="Recorded wording updates after Phase 4 review")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_dir() / "rubrics_all_questions_final_version.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=default_output_dir() / "rubrics_validation.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, metadata = build(args.source_dir)
    checked_questions, checked_items = validate_human_subset(data, args.check_file)
    corrections = json.loads(args.corrections.read_text())["changes"]
    questions = {q["question_number"]: q for q in data["questions"]}
    for change in corrections:
        item = next(i for i in questions[change["question_number"]]["rubric_items"] if i["item_number"] == change["item_number"])
        if item[change["field"]] != change["before"]:
            raise ValueError(f"Correction source differs: Q{change['question_number']} item {change['item_number']}")
        item[change["field"]] = change["after"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(
        args.report,
        args.output,
        args.check_file,
        metadata,
        checked_questions,
        checked_items,
    )
    with args.report.open("a", encoding="utf-8") as report:
        report.write(f"\nApplied {len(corrections)} recorded wording updates after validating the archived human subset. Correction-file SHA-256: `{sha256(args.corrections)}`.\n")
    print(f"Wrote 282 questions and {metadata['item_count']} rubric items")
    print(
        f"Validated {checked_items} human-graded rubric items across "
        f"{checked_questions} questions"
    )


if __name__ == "__main__":
    main()
