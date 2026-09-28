#!/usr/bin/env python3
"""Reproduce Figure 4a from saved, final Claude Opus 5 grades; no API calls.

Preserves the original standalone barplot_without_deduct.py plotting style.
Reuses grade_anthropic_batch.py's rubric loader and signed-score validator.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from importlib.metadata import version
import json
import math
import platform
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def load_grading_helpers(directory: Path):
    """Import existing pure validation helpers without executing the API CLI."""
    path = directory / "grade_anthropic_batch.py"
    if not path.is_file():
        raise ValueError(f"Missing {path}; set --grading-code-dir to 'Evaluation/rubric_scoring'.")
    spec = importlib.util.spec_from_file_location("pancanbench_grading", path)
    module = importlib.util.module_from_spec(spec)
    # The grader imports its adjacent grading_prompt.py module.
    sys.path.insert(0, str(directory.resolve()))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def load_scoring_rubrics(data_dir, config, helpers):
    """Verify original grading provenance and explicitly corrected scoring weights."""
    scoring_path = data_dir / config["rubrics_file"]
    expected_hash = config.get("scoring_rubrics_sha256", config["rubrics_sha256"])
    if helpers.sha256(scoring_path) != expected_hash:
        raise ValueError("Scoring rubric hash mismatch")
    rubrics = helpers.load_rubrics(scoring_path)
    paths = [scoring_path]
    if "grading_rubrics_file" in config:
        original_path = data_dir / config["grading_rubrics_file"]
        if helpers.sha256(original_path) != config["rubrics_sha256"]:
            raise ValueError("Original grading rubric hash mismatch")
        expected = helpers.load_rubrics(original_path)
        for correction in config.get("scoring_corrections", []):
            item = expected[correction["question_id"]]["rubric_items"][correction["criterion_number"] - 1]
            for key in ("min_points", "max_points"):
                if item[key] != correction["before"][key]:
                    raise ValueError("Unexpected original weight in scoring correction")
                item[key] = correction["after"][key]
        if expected != rubrics:
            raise ValueError("Scoring rubrics differ beyond the declared weight corrections")
        paths.append(original_path)
    return rubrics, paths


def require_close(actual, expected, context):
    if (isinstance(actual, bool) or not isinstance(actual, (int, float))
            or not math.isfinite(actual)
            or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8)):
        raise ValueError(f"{context}: expected {expected}, received {actual!r}")


def validate_score(row, rubrics, helpers, config):
    """Check criteria using existing grading code, then verify normalization."""
    question_id = helpers.validate_normalized_row(row, rubrics)
    if row.get("question_number") != int(question_id[1:]):
        raise ValueError(f"{question_id}: question_number mismatch")
    for key in ("judge_model", "rubrics_sha256", "prompt_version"):
        if row.get(key) != config[key]:
            raise ValueError(f"{question_id}: {key} mismatch")
    for correction in config.get("scoring_corrections", []):
        if question_id == correction["question_id"]:
            events = [e for e in row.get("manual_corrections", [])
                      if e.get("correction_id") == correction["correction_id"]]
            if len(events) != 1 or row.get("scoring_rubrics_sha256") != config["scoring_rubrics_sha256"]:
                raise ValueError(f"{question_id}: missing scoring-correction provenance")
    if row.get("grader_model") != config["judge_model"]:
        raise ValueError(f"{question_id}: grader_model mismatch")
    items = rubrics[question_id]["rubric_items"]
    maximum = sum(item["max_points"] for item in items)
    minimum = sum(item["min_points"] for item in items)
    if maximum <= 0:
        raise ValueError(f"{question_id}: no positive-point denominator")
    require_close(row.get("max_possible_score"), maximum, f"{question_id}: maximum")
    require_close(row.get("minimum_possible_score"), minimum, f"{question_id}: minimum")
    percentage = 100.0 * row["total_score"] / maximum
    require_close(row.get("percentage"), percentage, f"{question_id}: percentage")
    return percentage


def summarize(rows, model):
    included = [row for row in rows if row["included_in_mean"]]
    values = [row["percentage"] for row in included]
    if len(values) < 2:
        raise ValueError(f"{model['id']}: fewer than two included responses")
    sd = statistics.stdev(values)  # Sample SD, denominator n - 1.
    return {
        "response_model": model["id"], "display_name": model["label"],
        "provider": model["provider"], "n_total": len(rows),
        "n_included": len(values), "n_excluded": len(rows) - len(values),
        "mean_percentage": statistics.mean(values), "sd_percentage": sd,
        "se_percentage": sd / math.sqrt(len(values)),
        "median_percentage": statistics.median(values),
        "minimum_percentage": min(values), "maximum_percentage": max(values),
        "genuine_zero_scores_retained": sum(value == 0 for value in values),
        "negative_total_scores_retained": sum(value < 0 for value in values),
        "negative_penalties_applied": sum(row["negative_penalties_applied"] for row in included),
    }


def analyze(data_dir, config_path, helpers):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    models = {model["id"]: model for model in config["models"]}
    if len(models) != len(config["models"]) or len(models) != config["expected_models"]:
        raise ValueError("Incorrect or duplicate model manifest entries")
    if config["ordering"] != "provider_then_mean_percentage_descending":
        raise ValueError("Expected provider grouping with descending means within each family")
    provider_order = config["provider_order"]
    if (len(provider_order) != len(set(provider_order))
            or set(provider_order) != {m["provider"] for m in models.values()}):
        raise ValueError("Provider order must contain each model family exactly once")
    rubrics, rubric_paths = load_scoring_rubrics(data_dir, config, helpers)
    item_count = sum(len(q["rubric_items"]) for q in rubrics.values())
    if len(rubrics) != config["expected_questions"] or item_count != config["expected_rubric_items"]:
        raise ValueError("Unexpected benchmark size")
    exclusions = {(e["response_model"], e["question_id"]): e["reason"]
                  for e in config["excluded_responses"]}
    if len(exclusions) != len(config["excluded_responses"]):
        raise ValueError("Duplicate exclusion entries")
    if any(model not in models or qid not in rubrics or reason != "NO RESPONSE GENERATED"
           for (model, qid), reason in exclusions.items()):
        raise ValueError("Invalid missing-response exclusion")
    rows_by_model = defaultdict(list)
    references = {}
    input_paths = list(rubric_paths)
    seen = set()
    ignored_counts = Counter()
    for source, relative_path in config["grade_files"].items():
        path = data_dir / relative_path
        summary_path = path.with_suffix(".summary.json")
        reference = json.loads(summary_path.read_text(encoding="utf-8"))
        input_paths.extend([path, summary_path])
        source_models = {m for m, details in models.items() if details["source"] == source}
        reported_exclusions = {(e["response_model"], e["question_id"])
                               for e in reference["excluded_model_question_pairs"]
                               if e["response_model"] in source_models}
        if reported_exclusions != {key for key in exclusions if key[0] in source_models}:
            raise ValueError(f"{summary_path.name}: missing-response metadata mismatch")
        for result in reference["models"]:
            model_id = result["response_model"]
            if model_id in source_models:
                if model_id in references:
                    raise ValueError(f"Duplicate reference summary: {model_id}")
                references[model_id] = result
        for row in helpers.read_json_or_jsonl(path):
            model_id = row.get("response_model")
            if model_id not in models:
                if model_id not in config["excluded_models_in_grade_files"]:
                    raise ValueError(f"{path.name}: unlisted model {model_id!r}")
                ignored_counts[model_id] += 1
                continue
            if model_id not in source_models:
                raise ValueError(f"{model_id}: grade is in the wrong source file")
            key = (model_id, row.get("question_id"))
            if key in seen:
                raise ValueError(f"Duplicate grade: {key}")
            seen.add(key)
            try:
                percentage = validate_score(row, rubrics, helpers, config)
            except ValueError as error:
                raise ValueError(f"{model_id}: {error}") from error
            rows_by_model[model_id].append({
                "response_model": model_id, "display_name": models[model_id]["label"],
                "provider": models[model_id]["provider"], "question_id": row["question_id"],
                "question_number": row["question_number"], "grade_file": relative_path,
                "criterion_count": len(row["criterion_scores"]),
                "total_score": row["total_score"],
                "minimum_possible_score": row["minimum_possible_score"],
                "max_possible_score": row["max_possible_score"], "percentage": percentage,
                "included_in_mean": key not in exclusions,
                "exclusion_reason": exclusions.get(key, ""),
                "negative_penalties_applied": sum(c["score_given"] < 0 for c in row["criterion_scores"]),
            })
    expected_keys = {(model, question) for model in models for question in rubrics}
    if seen != expected_keys:
        raise ValueError(f"Coverage mismatch: missing {sorted(expected_keys-seen)}, extra {sorted(seen-expected_keys)}")
    if set(references) != set(models):
        raise ValueError("Reference summaries do not cover exactly the selected models")
    results = []
    for model_id, model in models.items():
        result = summarize(rows_by_model[model_id], model)
        ref = references[model_id]
        for key, ref_key in [("n_total", "responses_total"), ("n_included", "responses_included_in_mean"),
                             ("n_excluded", "responses_excluded_from_mean"),
                             ("mean_percentage", "mean_percentage")]:
            require_close(result[key], ref[ref_key], f"{model_id}: reference {key}")
        result["color"] = config["provider_colors"][model["provider"]]
        results.append(result)
    results.sort(key=lambda row: (provider_order.index(row["provider"]),
                                  -row["mean_percentage"], row["response_model"]))
    results = [dict(plot_order=position, **row) for position, row in enumerate(results, 1)]
    questions = [dict(plot_order=result["plot_order"], **row) for result in results
                 for row in sorted(rows_by_model[result["response_model"]], key=lambda r: r["question_number"])]
    provenance = {
        "figure": "4a", "judge_model": config["judge_model"],
        "prompt_version": config["prompt_version"], "rubrics_sha256": config["rubrics_sha256"],
        "scoring_rubrics_sha256": config.get("scoring_rubrics_sha256", config["rubrics_sha256"]),
        "scoring_corrections": config.get("scoring_corrections", []),
        "questions": len(rubrics), "rubric_items": item_count, "models": len(models),
        "grade_records_validated": len(questions),
        "criterion_ratings_validated": sum(row["criterion_count"] for row in questions),
        "records_included_in_means": sum(row["included_in_mean"] for row in questions),
        "excluded_responses": config["excluded_responses"],
        "ignored_models_in_source_files": dict(sorted(ignored_counts.items())),
        "genuine_zero_scores_retained": sum(row["genuine_zero_scores_retained"] for row in results),
        "negative_penalties_applied": sum(row["negative_penalties_applied"] for row in results),
        "negative_total_scores_retained": sum(row["negative_total_scores_retained"] for row in results),
        "score_definition": "100 * signed total_score / sum of positive rubric points; no clipping or extra factual-error deduction",
        "aggregation": "Unweighted arithmetic mean of included per-question percentages",
        "error_bars": "Plus/minus one SE; sample SD (n-1 denominator) / sqrt(n); not confidence intervals",
        "ordering": config["ordering"], "provider_order": provider_order,
        "model_order": [row["response_model"] for row in results],
        "checks": {"rubric_hash": "passed", "coverage_and_duplicates": "passed",
                   "criterion_descriptions_and_signed_weights": "passed", "normalization": "passed",
                   "explicit_missing_response_metadata": "passed", "reference_summary_means_and_counts": "passed"},
        "inputs": [{"path_relative_to_data": str(p.relative_to(data_dir)), "sha256": helpers.sha256(p)}
                   for p in input_paths],
    }
    return config, results, questions, provenance


def plot(results, config, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, findfont

    # Fail rather than silently substituting a different font from the original.
    font_paths = [findfont(FontProperties(family=config["font_family"], weight=weight),
                           fallback_to_default=False) for weight in ("normal", "bold")]
    plt.rcdefaults()
    plt.rcParams.update({"font.family": config["font_family"],
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(18, 8))
    x_pos = 0
    x_positions, x_labels, company_positions, company_labels = [], [], [], []
    for provider in config["provider_order"]:
        company_models = [row for row in results if row["provider"] == provider]
        company_start = x_pos
        for row in company_models:
            ax.bar(x_pos, row["mean_percentage"], color=row["color"], edgecolor="black",
                   linewidth=0.8, width=0.6, yerr=row["se_percentage"],
                   capsize=5, error_kw={"linewidth": 1.5, "ecolor": "black"})
            x_positions.append(x_pos)
            x_labels.append(row["response_model"])
            x_pos += 1
        company_positions.append((company_start + x_pos - 1) / 2)
        company_labels.append(config["provider_headings"][provider])
        x_pos += 0.5

    ax.set_ylabel("Average Score (%)", fontsize=14, fontweight="bold")
    ax.set_title("Average Score by Model (Grouped by Company)",
                 fontsize=20, fontweight="bold", pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=15)
    ax.set_ylim(0, max(row["mean_percentage"] for row in results) * 1.15)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(company_positions)
    ax2.set_xticklabels(company_labels, fontsize=15, fontweight="bold")
    ax2.tick_params(axis="x", length=0)
    # twiny() restores the lower-axis ticks, so hide them after creating it.
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(output_dir / "figure4a.pdf", dpi=300, bbox_inches="tight",
                metadata={"Title": "PanCanBench Figure 4a", "CreationDate": None})
    fig.savefig(output_dir / "figure4a.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    packages = ["matplotlib", "numpy", "contourpy", "cycler", "fonttools", "kiwisolver",
                "packaging", "pillow", "pyparsing", "python-dateutil", "six"]
    import hashlib
    return {"python": platform.python_version(),
            "font_files": [{"filename": Path(path).name,
                            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                           for path in dict.fromkeys(font_paths)],
            "packages": {package: version(package) for package in packages}}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    folder = Path(__file__).resolve().parent
    root = folder.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "Data")
    parser.add_argument("--grading-code-dir", type=Path, default=root / "Evaluation/rubric_scoring")
    parser.add_argument("--config", type=Path, default=folder / "model_config.json")
    parser.add_argument("--output-dir", type=Path, default=root / "Outputs/figure4")
    args = parser.parse_args()
    helpers = load_grading_helpers(args.grading_code_dir)
    config, results, questions, provenance = analyze(args.data_dir, args.config, helpers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance["software"] = plot(results, config, args.output_dir)
    provenance["plot_settings"] = {"template": "barplot_without_deduct.py (standalone Figure 4a)",
                                   "size_inches_before_tight_crop": [18, 8], "png_dpi": 300,
                                   "font": config["font_family"],
                                   "y_limits": [0, max(row["mean_percentage"] for row in results) * 1.15],
                                   "model_labels": "full response_model IDs", "model_label_rotation_degrees": 45,
                                   "x_tick_marks_visible": False,
                                   "bar_width": 0.6, "gap_between_families": 0.5,
                                   "numeric_bar_labels": False, "legend": False,
                                   "provider_headings": config["provider_headings"],
                                   "provider_colors": config["provider_colors"]}
    write_csv(args.output_dir / "figure4a_model_summary.csv", results)
    write_csv(args.output_dir / "figure4a_question_scores.csv", questions)
    provenance["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["code"] = [{"path": "Analysis/figure4/plot_figure4a.py", "sha256": helpers.sha256(Path(__file__))},
                          {"path": "Analysis/figure4/model_config.json", "sha256": helpers.sha256(args.config)},
                          {"path": "Evaluation/rubric_scoring/grade_anthropic_batch.py", "sha256": helpers.sha256(args.grading_code_dir / "grade_anthropic_batch.py")}]
    provenance["outputs"] = [{"file": name, "sha256": helpers.sha256(args.output_dir / name)}
                             for name in ["figure4a.pdf", "figure4a.png", "figure4a_model_summary.csv", "figure4a_question_scores.csv"]]
    (args.output_dir / "figure4a_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(results)} models, {len(questions)} records; {provenance['records_included_in_means']} included.")
    for row in results:
        print(f"{row['plot_order']:2d}. {row['display_name']:<25} {row['mean_percentage']:6.2f} ± {row['se_percentage']:.2f} SE (n={row['n_included']})")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
