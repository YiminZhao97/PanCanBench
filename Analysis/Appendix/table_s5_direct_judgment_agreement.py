#!/usr/bin/env python3
"""Recalculate Appendix Table S5 using final Claude Opus 5 rubric grades.

The original GPT-5 direct-judgment decisions are treated as fixed. For every
question and model pair, the rubric-based winner is determined from the final
response percentage. That percentage accounts for both positive and negative
rubric items:

    100 * total_score / max_possible_score

The script validates that formula against the stored ``percentage`` field
before calculating agreement.
"""

from __future__ import annotations

import argparse
import json
import csv
import hashlib
import math
import sys

sys.dont_write_bytecode = True
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

PAIR_SPECS = [
    {
        "model_a": "gpt-5",
        "model_b": "o3",
        "display": "GPT-5 vs o3",
        "judgments": "gpt-5_vs_o3_judgment_20251209_172054.json",
    },
    {
        "model_a": "gpt-5",
        "model_b": "grok-4-latest",
        "display": "GPT-5 vs Grok-4",
        "judgments": "gpt-5_vs_grok-4-latest_judgment_20251209_200938.json",
    },
    {
        "model_a": "gpt-5",
        "model_b": "gemini-2.5-flash",
        "display": "GPT-5 vs Gemini-2.5 Flash",
        "judgments": "gpt-5_vs_gemini-2.5-flash_judgment_20251209_194846.json",
    },
    {
        "model_a": "o3",
        "model_b": "gemini-2.5-flash",
        "display": "o3 vs Gemini-2.5 Flash",
        "judgments": "o3_vs_gemini-2.5-flash_judgment_20251209_223132.json",
    },
    {
        "model_a": "o3",
        "model_b": "grok-4-latest",
        "display": "o3 vs Grok-4",
        "judgments": "o3_vs_grok-4-latest_judgment_20251209_200025.json",
    },
    {
        "model_a": "grok-4-latest",
        "model_b": "gemini-2.5-flash",
        "display": "Grok-4 vs Gemini-2.5 Flash",
        "judgments": "grok-4-latest_vs_gemini-2.5-flash_judgment_20251209_225718.json",
    },
]

GRADE_FILES = [
    "gpt5_response_graded.jsonl",
    "grok_response_graded.jsonl",
    "openai_family_response_graded.jsonl",
    "gemini_family_response_graded.jsonl",
]

TARGET_MODELS = {spec[key] for spec in PAIR_SPECS for key in ("model_a", "model_b")}


def load_final_grades(grades_dir: Path) -> tuple[dict[str, dict[int, dict]], str]:
    grades: dict[str, dict[int, dict]] = {model: {} for model in TARGET_MODELS}
    rubric_hashes: set[str] = set()

    for filename in GRADE_FILES:
        path = grades_dir / filename
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                model = record.get("source")
                if model not in TARGET_MODELS:
                    continue

                if record.get("judge_model") != "claude-opus-5":
                    raise ValueError(f"Unexpected judge in {path}:{line_number}")

                question = int(record["question_number"])
                if question in grades[model]:
                    raise ValueError(f"Duplicate {model} Q{question}")

                maximum = float(record["max_possible_score"])
                total = float(record["total_score"])
                if not math.isfinite(maximum) or maximum <= 0 or not math.isfinite(total):
                    raise ValueError(f"Zero maximum score for {model} Q{question}")
                expected_percentage = 100.0 * total / maximum
                if not math.isfinite(float(record["percentage"])) or abs(expected_percentage - float(record["percentage"])) > 1e-9:
                    raise ValueError(
                        f"Stored percentage does not match the grading formula "
                        f"for {model} Q{question}"
                    )

                grades[model][question] = record
                rubric_hashes.add(record["rubrics_sha256"])

    expected_questions = set(range(1, 283))
    for model, records in grades.items():
        if set(records) != expected_questions:
            missing = sorted(expected_questions - set(records))
            extra = sorted(set(records) - expected_questions)
            raise ValueError(f"Question mismatch for {model}: missing={missing}, extra={extra}")

    if len(rubric_hashes) != 1:
        raise ValueError(f"Expected one final rubric hash, found {sorted(rubric_hashes)}")

    return grades, next(iter(rubric_hashes))


def rubric_winner(record_a: dict, record_b: dict) -> str:
    if (
        record_a["minimum_possible_score"] != record_b["minimum_possible_score"]
        or record_a["max_possible_score"] != record_b["max_possible_score"]
    ):
        raise ValueError(
            f"Rubric score bounds differ for Q{record_a['question_number']} "
            f"({record_a['source']} vs {record_b['source']})"
        )

    score_a = float(record_a["percentage"])
    score_b = float(record_b["percentage"])
    if abs(score_a - score_b) <= 1e-12:
        return "TIE"
    return "A" if score_a > score_b else "B"


def calculate_pair(spec: dict, grades: dict[str, dict[int, dict]], direct_dir: Path, question_rows: list[dict]) -> dict:
    path = direct_dir / spec["judgments"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload["judgment_metadata"]
    judgments = payload["judgment_results"]

    if metadata.get("judge_model") != "gpt-5":
        raise ValueError(f"Unexpected direct judge in {path}")
    if len(judgments) != 282:
        raise ValueError(f"Expected 282 direct judgments in {path}, found {len(judgments)}")

    by_question: dict[int, dict] = {}
    for judgment in judgments:
        question = int(judgment["question_number"])
        if question in by_question:
            raise ValueError(f"Duplicate direct judgment for Q{question} in {path}")
        if judgment["source_a"] != spec["model_a"] or judgment["source_b"] != spec["model_b"]:
            raise ValueError(f"Source orientation mismatch for Q{question} in {path}")
        if judgment["winner"] not in {"A", "B", "TIE"}:
            raise ValueError(f"Invalid direct winner for Q{question} in {path}")
        by_question[question] = judgment

    if set(by_question) != set(range(1, 283)):
        raise ValueError(f"Direct-judgment question set is incomplete in {path}")

    joint_outcomes: Counter[str] = Counter()
    direct_outcomes: Counter[str] = Counter()
    rubric_outcomes: Counter[str] = Counter()

    for question in range(1, 283):
        direct = by_question[question]["winner"]
        rubric = rubric_winner(
            grades[spec["model_a"]][question],
            grades[spec["model_b"]][question],
        )
        question_rows.append({
            "model_a": spec["model_a"], "model_b": spec["model_b"],
            "question_number": question,
            "score_a": grades[spec["model_a"]][question]["percentage"],
            "score_b": grades[spec["model_b"]][question]["percentage"],
            "direct_winner": direct, "rubric_winner": rubric,
            "agreement": direct == rubric,
        })
        direct_outcomes[direct] += 1
        rubric_outcomes[rubric] += 1
        joint_outcomes[f"{direct}_{rubric}"] += 1

    denominator = 282
    both_a = joint_outcomes["A_A"]
    both_b = joint_outcomes["B_B"]
    both_tie = joint_outcomes["TIE_TIE"]
    agreement = both_a + both_b + both_tie

    return {
        "model_pair": spec["display"],
        "model_a": spec["model_a"],
        "model_b": spec["model_b"],
        "n": denominator,
        "both_favor_a_count": both_a,
        "both_favor_a_percent": 100.0 * both_a / denominator,
        "both_favor_b_count": both_b,
        "both_favor_b_percent": 100.0 * both_b / denominator,
        "both_tie_count": both_tie,
        "both_tie_percent": 100.0 * both_tie / denominator,
        "overall_agreement_count": agreement,
        "overall_agreement_percent": 100.0 * agreement / denominator,
        "direct_winner_counts": dict(sorted(direct_outcomes.items())),
        "rubric_winner_counts": dict(sorted(rubric_outcomes.items())),
        "joint_outcome_counts": dict(sorted(joint_outcomes.items())),
        "direct_judgment_file": path.name,
    }


def render_markdown(results: list[dict], rubric_hash: str) -> str:
    lines = [
        "# Appendix Table S5 Recalculation",
        "",
        "The original GPT-5 direct-judgment decisions were retained. Rubric-based winners were recalculated from the final Claude Opus 5 grades using the stored response percentages (total awarded points divided by total positive points); negative rubric penalties reduce the numerator.",
        "",
        f"- Questions per pair: 282",
        f"- Original grading rubric SHA-256: `{rubric_hash}`",
        "- Validation: all four target models had one complete grade per question; all grade records used Claude Opus 5 and the same original grading rubrics; all stored percentages matched the grading formula. Q151 criterion 12 uses the corrected -5 weight and 135-point maximum, with the original judge decisions retained.",
        "",
        "| Model pair | Both favor A | Both favor B | Both tie | Overall agreement |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['model_pair']} | "
            f"{result['both_favor_a_percent']:.1f}% | "
            f"{result['both_favor_b_percent']:.1f}% | "
            f"{result['both_tie_percent']:.1f}% | "
            f"{result['overall_agreement_percent']:.1f}% |"
        )

    agreements = [result["overall_agreement_percent"] for result in results]
    lines.extend(
        [
            "",
            f"Overall agreement ranged from **{min(agreements):.1f}% to {max(agreements):.1f}%** across the six model pairs.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path,
                        help="Default: ROOT/Outputs/appendix")
    args = parser.parse_args()
    root = args.root.resolve()
    args.output_dir = args.output_dir or root / "Outputs/appendix"
    manifest = json.loads(Path(__file__).with_name("input_manifest.json").read_text())
    inputs = manifest["table_s5"]
    for relative, expected in inputs.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Input hash mismatch: {relative}")
    grades, rubric_hash = load_final_grades(root / "Data/grades")
    if rubric_hash != inputs["Data/grades/grading_rubrics_snapshot.json"]:
        raise ValueError("Grade rubric hash does not match the original grading snapshot")
    scoring_hash = inputs["Data/grades/scoring_rubrics.json"]
    for model, rows in grades.items():
        if rows[151].get("scoring_rubrics_sha256") != scoring_hash:
            raise ValueError(f"Missing corrected scoring provenance for {model} Q151")
    question_rows = []
    results = [calculate_pair(spec, grades, root / "Data/direct_judgments", question_rows)
               for spec in PAIR_SPECS]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "table_s5_questions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(question_rows[0]))
        writer.writeheader()
        writer.writerows(question_rows)
    summary_rows = [{k: v for k, v in row.items() if not isinstance(v, dict)} for row in results]
    with (args.output_dir / "table_s5.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    output = {
        "description": "Recalculation of Appendix Table S5",
        "direct_judge": "gpt-5 (original decisions retained)",
        "rubric_judge": "claude-opus-5",
        "rubric_sha256": rubric_hash,
        "scoring_rubrics_sha256": scoring_hash,
        "scoring_correction": "Q151.12 weight changed from +5 to -5; original decisions retained; maximum changed from 140 to 135.",
        "score_formula": "100 * total_score / max_possible_score",
        "pairs": results,
        "input_sha256": inputs,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "api_calls": 0,
    }
    (args.output_dir / "table_s5_results.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "table_s5_results.md").write_text(
        render_markdown(results, rubric_hash), encoding="utf-8"
    )

    print(render_markdown(results, rubric_hash))


if __name__ == "__main__":
    main()
