#!/usr/bin/env python3
"""Compare candidate LLM judges with oncology fellows (Table S2; offline)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
LABELS = {
    "claude-opus-5": "Claude Opus 5",
    "gpt-5.5-2026-04-23": "GPT-5.5 (2026-04-23)",
    "claude-sonnet-5": "Claude Sonnet 5",
    "gpt-5.6-sol": "GPT-5.6-sol",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Metric definitions retained from the reviewed human-agreement analysis.
QUESTION_COLUMN = "Question ID"

RUBRIC_COLUMN = "Rubric Item"

SCORE_COLUMN = (
    "Meets Criterion (1 = meets the criterion; 0 = does not meet the criterion)"
)

GRADERS = ("Sheela", "Karly", "Simone", "Manuel", "Jesse")

def load_ratings(path: Path) -> dict[tuple[str, int], int]:
    """Load and validate one grader's binary rubric-item ratings."""
    ratings: dict[tuple[str, int], int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {QUESTION_COLUMN, RUBRIC_COLUMN, SCORE_COLUMN}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path.name}: expected columns {sorted(required)}")

        for row_number, row in enumerate(reader, start=2):
            question = row[QUESTION_COLUMN].strip()
            try:
                rubric_item = int(row[RUBRIC_COLUMN])
                score = int(row[SCORE_COLUMN])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path.name}, row {row_number}: rubric item and score must be integers"
                ) from error
            if not question:
                raise ValueError(f"{path.name}, row {row_number}: blank Question ID")
            if rubric_item < 1:
                raise ValueError(f"{path.name}, row {row_number}: invalid rubric item")
            if score not in (0, 1):
                raise ValueError(f"{path.name}, row {row_number}: score must be 0 or 1")
            key = (question, rubric_item)
            if key in ratings:
                raise ValueError(f"{path.name}: duplicate rating for {question}.{rubric_item}")
            ratings[key] = score

    return ratings

def observed_agreement(left: list[int], right: list[int]) -> float:
    return sum(a == b for a, b in zip(left, right)) / len(left)

def cohen_kappa(left: list[int], right: list[int]) -> float:
    """Cohen's kappa for two complete binary rating vectors."""
    agreement = observed_agreement(left, right)
    p_left = mean(left)
    p_right = mean(right)
    expected = p_left * p_right + (1.0 - p_left) * (1.0 - p_right)
    return (agreement - expected) / (1.0 - expected)

def class_f1(left: list[int], right: list[int], label: int) -> float:
    """F1 for one class; the result is symmetric in the two graders."""
    shared_label = sum(a == label and b == label for a, b in zip(left, right))
    disagreements = sum(a != b for a, b in zip(left, right))
    denominator = 2 * shared_label + disagreements
    return 2 * shared_label / denominator if denominator else 0.0

def macro_f1(left: list[int], right: list[int]) -> float:
    """Unweighted mean of F1 for scores 0 and 1."""
    return mean(class_f1(left, right, label) for label in (0, 1))

def load_judge_ratings(path, keys, dataset_sha, model):
    ratings = {}
    seen_questions = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["dataset_sha256"] != dataset_sha or row["judge_model"] != model:
                raise ValueError(f"Historical dataset or judge identity mismatch: {path.name}")
            question = row["question_id"]
            if question in seen_questions:
                raise ValueError(f"Duplicate question in {path.name}: {question}")
            seen_questions.add(question)
            for grade in row["grades"]:
                key = (question, int(grade["rubric_item"]))
                score = grade["score"]
                if key in ratings or score not in (0, 1):
                    raise ValueError(f"Duplicate or invalid binary rating in {path.name}: {key}")
                ratings[key] = score
    if set(ratings) != set(keys):
        raise ValueError(f"Judge and human item keys do not match: {path.name}")
    return ratings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Root containing Data/judge_validation/ and Data/human_ratings/final/")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: ROOT/Outputs/appendix")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads(Path(__file__).with_name("input_manifest.json").read_text())
    inputs = manifest["table_s2"]
    for relative, expected in inputs.items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Input hash mismatch: {relative}")

    # The numerical table needs the saved binary ratings, not response text.
    # Validate the historical dataset identity recorded in every judge row;
    # the current editable judge_dataset.json is not that historical snapshot.
    dataset_sha = manifest["table_s2_recorded_dataset_sha256"]
    humans = {}
    keys = None
    for grader in GRADERS:
        path = root / "Data/human_ratings/final" / f"full_new_grading_results_{grader}.csv"
        ratings = load_ratings(path)
        if keys is None:
            keys = sorted(ratings, key=lambda key: (int(key[0][1:]), key[1]))
            if len(keys) != 487 or len({key[0] for key in keys}) != 40:
                raise ValueError("Table S2 requires the fixed 40-question, 487-item cohort")
        if set(ratings) != set(keys):
            raise ValueError(f"Human rating keys do not match: {grader}")
        humans[grader] = [ratings[key] for key in keys]

    rows, pairwise = [], []
    for model, label in LABELS.items():
        path = root / "Data/judge_validation" / f"{model}.jsonl"
        ratings = load_judge_ratings(path, keys, dataset_sha, model)
        judge = [ratings[key] for key in keys]
        comparisons = []
        for grader, human in humans.items():
            comparison = {
                "model": model, "human_grader": grader, "n_items": len(keys),
                "cohen_kappa": cohen_kappa(judge, human),
                "macro_f1": macro_f1(judge, human),
                "observed_agreement": observed_agreement(judge, human),
            }
            comparisons.append(comparison)
            pairwise.append(comparison)
        rows.append({
            "model": model, "candidate_judge": label, "n_questions": 40,
            "n_items": len(keys), "n_human_graders": len(humans),
            "mean_kappa": mean(x["cohen_kappa"] for x in comparisons),
            "macro_f1": mean(x["macro_f1"] for x in comparisons),
            "agreement_percent": 100 * mean(x["observed_agreement"] for x in comparisons),
        })
    rows.sort(key=lambda x: (-x["mean_kappa"], -x["macro_f1"], x["model"]))
    out = args.output_dir or root / "Outputs/appendix"
    out.mkdir(parents=True, exist_ok=True)
    for filename, values in [("table_s2.csv", rows), ("table_s2_pairwise.csv", pairwise)]:
        with (out / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    table = ["# Table S2. Agreement between candidate LLM judges and five oncology fellows",
             "", "| Candidate judge | Mean kappa | Macro-F1 | Agreement |",
             "|---|---:|---:|---:|"]
    table += [f"| {r['candidate_judge']} | {r['mean_kappa']:.3f} | {r['macro_f1']:.3f} | {r['agreement_percent']:.1f}% |" for r in rows]
    table += ["", "Each metric is computed separately against each fellow on all 487 items, then averaged across five fellows. Macro-F1 weights the two binary classes equally. Rank uses unrounded mean kappa, then macro-F1. No bootstrap interval is reported in Table S2.", ""]
    (out / "table_s2.md").write_text("\n".join(table))
    provenance = {"table": "S2", "rows": rows, "input_sha256": inputs,
                  "script_sha256": sha256(Path(__file__)), "python": sys.version,
                  "api_calls": 0, "recorded_judge_dataset_sha256": dataset_sha,
                  "dataset_provenance_note": "The current judge_dataset.json differs from the historical run hash and is not an input to this offline rating-based calculation."}
    (out / "table_s2_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("\n".join(table))
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
