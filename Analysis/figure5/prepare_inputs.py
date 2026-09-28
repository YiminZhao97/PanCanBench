#!/usr/bin/env python3
"""Validate saved human/synthetic grade JSONLs and freeze paired Figure 5 inputs.

No API calls. Recomputes every criterion score against the canonical rubrics,
checks source summaries, and requires matching response-input hashes.
"""
from __future__ import annotations
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from figure5_common import HERE, close, require, sha256, write_csv, write_json


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_rubrics(path, kind, config):
    expected_hash = config["scoring_rubrics_sha256" if kind == "human" else "synthetic_original_sha256"]
    require(sha256(path) == expected_hash, f"Canonical {kind} rubric hash mismatch")
    if kind == "human":
        result = {f"Q{q['question_number']}": q["rubric_items"]
                  for q in json.loads(path.read_text())["questions"]}
    else:
        result = {}
        for question in jsonl(path):
            qid = question["question_id"]
            require(qid not in result, f"Duplicate rubric {qid}")
            result[qid] = [{"item_number": i, "description": item["item_description"],
                            "min_points": min(0, item["points"]),
                            "max_points": max(0, item["points"])}
                           for i, item in enumerate(question["rubric_items"], 1)]
    require(set(result) == {f"Q{i}" for i in range(1, 283)}, "Unexpected rubric questions")
    count = config["expected_rubric_items" if kind == "human" else "expected_synthetic_rubric_items"]
    require(sum(map(len, result.values())) == count, f"Unexpected {kind} rubric item count")
    return result


def validate_row(row, items, kind, config):
    context = f"{kind}, {row['response_model']}, {row['question_id']}"
    for field in ("judge_model", "grader_model"):
        require(row[field] == config["judge_model"], f"Judge mismatch: {context}")
    for field in ("prompt_version", "rubrics_sha256"):
        expected = config[field if kind == "human" else f"synthetic_{field}"]
        require(row[field] == expected, f"{field} mismatch: {context}")
    require(row["question_number"] == int(row["question_id"][1:]), f"Question number: {context}")
    criteria = row["criterion_scores"]
    require(len(criteria) == len(items), f"Criterion count: {context}")
    for criterion, item in zip(criteria, items):
        for actual, canonical in (("criterion_number", "item_number"), ("description", "description"),
                                  ("min_points", "min_points"), ("max_points", "max_points")):
            require(criterion[actual] == item[canonical], f"Criterion {actual}: {context}")
        require(isinstance(criterion["meets_criterion"], bool), f"Invalid decision: {context}")
        expected = (item["min_points"] if item["min_points"] < 0 else item["max_points"]) if criterion["meets_criterion"] else 0
        close(criterion["score_given"], expected, f"Criterion score: {context}")
    total = sum(c["score_given"] for c in criteria)
    maximum = sum(c["max_points"] for c in items)
    minimum = sum(c["min_points"] for c in items)
    require(maximum > 0, f"Nonpositive denominator: {context}")
    for field, expected in (("total_score", total), ("max_possible_score", maximum),
                            ("minimum_possible_score", minimum), ("percentage", 100 * total / maximum)):
        close(row[field], expected, f"{field}: {context}")
    if kind == "human" and row["question_id"] == "Q151":
        require(row.get("scoring_rubrics_sha256") == config["scoring_rubrics_sha256"],
                f"Missing Q151 correction provenance: {context}")
    return {"total_score": total, "max_possible_score": maximum,
            "percentage": 100 * total / maximum, "criterion_count": len(criteria),
            "input_sha256": row["input_sha256"]}


def prepare(args):
    config = json.loads(args.config.read_text())
    models = {model["id"]: model for model in config["models"]}
    exclusions = {(e["response_model"], e["question_id"]) for e in config["excluded_responses"]}
    full = {(model, f"Q{q}") for model in models for q in range(1, 283)}
    paired, sources, references, validation = {}, [], {}, {}
    for kind in ("human", "synthetic"):
        rubric_path = getattr(args, f"{kind}_rubrics")
        rubrics = load_rubrics(rubric_path, kind, config)
        sources.append({"kind": kind, "role": "canonical_rubrics", "filename": rubric_path.name,
                        "sha256": sha256(rubric_path)})
        directory = getattr(args, f"{kind}_grades_dir")
        rows, refs, ignored = {}, {}, Counter()
        for source, relative in config["grade_files"].items():
            path = directory / Path(relative).name
            summary_path = path.with_suffix(".summary.json")
            summary = json.loads(summary_path.read_text())
            reference_rows = summary["models"] if kind == "human" else summary.get("model_summaries", [summary])
            expected_models = {model for model, definition in models.items() if definition["source"] == source}
            for reference in reference_rows:
                model = reference["response_model"]
                if model in expected_models:
                    require(model not in refs, f"Duplicate summary: {kind}, {model}")
                    refs[model] = {"n": reference["responses_included_in_mean" if kind == "human" else "responses"],
                                   "mean_percentage": reference["mean_percentage"]}
            for p, role in ((path, "grades"), (summary_path, "summary")):
                sources.append({"kind": kind, "role": role, "filename": p.name, "sha256": sha256(p)})
            for row in jsonl(path):
                model, qid = row["response_model"], row["question_id"]
                if model not in models:
                    require(model in config["excluded_models_in_grade_files"], f"Unlisted model: {model}")
                    ignored[model] += 1
                    continue
                require(model in expected_models, f"Model in wrong grade file: {model}")
                key = model, qid
                require(key in full and key not in rows, f"Unexpected/duplicate key: {kind}, {key}")
                rows[key] = validate_row(row, rubrics[qid], kind, config)
        require(set(rows) == (full if kind == "human" else full - exclusions), f"Coverage mismatch: {kind}")
        require(set(refs) == set(models), f"Missing reference summaries: {kind}")
        included = {key: value for key, value in rows.items() if key not in exclusions}
        for model, reference in refs.items():
            values = [row["percentage"] for key, row in included.items() if key[0] == model]
            require(len(values) == reference["n"], f"Summary sample count: {kind}, {model}")
            close(statistics.mean(values), reference["mean_percentage"], f"Summary mean: {kind}, {model}")
        paired[kind], references[kind] = included, refs
        validation[kind] = {"grade_records_validated": len(rows), "included": len(included),
                            "criterion_decisions_validated": sum(row["criterion_count"] for row in rows.values()),
                            "excluded_model_records": dict(ignored)}
    require(set(paired["human"]) == set(paired["synthetic"]), "Unpaired grades")
    output = []
    for model, qid in sorted(paired["human"], key=lambda key: (key[0], int(key[1][1:]))):
        human, synthetic = paired["human"][model, qid], paired["synthetic"][model, qid]
        require(human["input_sha256"] == synthetic["input_sha256"], f"Different response inputs: {model}, {qid}")
        row = {"response_model": model, "question_id": qid}
        for kind, values in (("human", human), ("synthetic", synthetic)):
            row.update({f"{kind}_{field}": value for field, value in values.items()})
        output.append(row)
    require(len(output) == config["expected_paired_responses"], "Incorrect total pair count")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "paired_question_scores.csv"
    write_csv(destination, output)
    write_json(args.output_dir / "input_provenance.json", {
        "config_sha256": sha256(args.config), "paired_question_scores_sha256": sha256(destination),
        "models": len(models), "paired_responses": len(output), "matching_response_input_hashes": len(output),
        "validation": validation, "reference_summaries": references, "sources": sources,
        "excluded_responses": config["excluded_responses"], "review_note": config["review_note"],
        "validation_note": "Mechanical and provenance checks; not clinical adjudication of judge decisions",
    })
    print(json.dumps({"paired_responses": len(output), "models": len(models), "validation": validation}, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("human-grades-dir", "synthetic-grades-dir", "human-rubrics", "synthetic-rubrics"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    cli.add_argument("--config", type=Path, default=HERE / "model_config.json")
    cli.add_argument("--output-dir", type=Path, default=HERE / "inputs")
    prepare(cli.parse_args())
