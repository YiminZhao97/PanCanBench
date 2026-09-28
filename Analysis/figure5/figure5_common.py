"""Shared, offline Figure 5 data checks and descriptive statistics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, context):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8),
            f"{context}: {actual!r} != {expected!r}")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--inputs-dir", type=Path, default=HERE / "inputs")
    result.add_argument("--config", type=Path, default=HERE / "model_config.json")
    # Both the materials copy and the repository copy are self-contained.
    result.add_argument("--output-dir", type=Path, default=HERE)
    return result


def load_results(inputs_dir, config_path):
    config = json.loads(config_path.read_text())
    manifest = json.loads((inputs_dir / "input_provenance.json").read_text())
    require(sha256(config_path) == manifest["config_sha256"], "Figure 5 configuration changed")
    score_path = inputs_dir / "paired_question_scores.csv"
    require(sha256(score_path) == manifest["paired_question_scores_sha256"],
            "Frozen question-level scores changed; rebuild and verify the inputs")
    models = {row["id"]: row for row in config["models"]}
    exclusions = {(e["response_model"], e["question_id"]) for e in config["excluded_responses"]}
    expected = {(model, f"Q{q}") for model in models
                for q in range(1, config["expected_questions"] + 1)} - exclusions
    with score_path.open(newline="", encoding="utf-8") as stream:
        questions = list(csv.DictReader(stream))
    seen, by_model = set(), {model: [] for model in models}
    for row in questions:
        key = row["response_model"], row["question_id"]
        require(key in expected and key not in seen, f"Unexpected or duplicate score: {key}")
        seen.add(key)
        require(row["human_input_sha256"] == row["synthetic_input_sha256"],
                f"Response input differs between rubric sets: {key}")
        for kind in ("human", "synthetic"):
            for field in ("total_score", "max_possible_score", "percentage"):
                row[f"{kind}_{field}"] = float(row[f"{kind}_{field}"])
            maximum = row[f"{kind}_max_possible_score"]
            require(maximum > 0, f"Nonpositive denominator: {key}, {kind}")
            close(row[f"{kind}_percentage"], 100 * row[f"{kind}_total_score"] / maximum,
                  f"Percentage {key}, {kind}")
        by_model[key[0]].append(row)
    require(seen == expected, f"Question coverage mismatch: {len(expected - seen)} missing")
    require(len(seen) == config["expected_paired_responses"], "Wrong paired cohort size")
    results = []
    for model_id, model in models.items():
        rows = by_model[model_id]
        result = {"response_model": model_id, "display_name": model["label"],
                  "provider": model["provider"], "n": len(rows)}
        for kind in ("human", "synthetic"):
            values = [row[f"{kind}_percentage"] for row in rows]
            result[f"{kind}_mean"] = statistics.mean(values)
            result[f"{kind}_sd"] = statistics.stdev(values)
            result[f"{kind}_se"] = result[f"{kind}_sd"] / math.sqrt(len(values))
            result[f"{kind}_zero_scores_retained"] = sum(v == 0 for v in values)
            result[f"{kind}_negative_scores_retained"] = sum(v < 0 for v in values)
            reference = manifest["reference_summaries"][kind][model_id]
            close(result[f"{kind}_mean"], reference["mean_percentage"], f"Reference mean {model_id}, {kind}")
            require(result["n"] == reference["n"], f"Reference count mismatch: {model_id}, {kind}")
        result["synthetic_minus_human_mean"] = result["synthetic_mean"] - result["human_mean"]
        results.append(result)
    # Competition ranking (1, 2, 2, 4), matching the original method='min'.
    for row in results:
        for kind in ("human", "synthetic"):
            row[f"{kind}_rank"] = 1 + sum(other[f"{kind}_mean"] > row[f"{kind}_mean"]
                                           for other in results)
        row["rank_change"] = row["human_rank"] - row["synthetic_rank"]
    results.sort(key=lambda row: (config["provider_order"].index(row["provider"]),
                                  -row["human_mean"], row["response_model"]))
    for index, row in enumerate(results, 1):
        row["plot_order"] = index
    require(len(results) == config["expected_models"], "Wrong model count")
    return config, results, manifest


def setup_plotting(config):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    requested = config.get("font_family", "Helvetica")
    try:
        font_manager.findfont(requested, fallback_to_default=False)
        selected = requested
    except ValueError:
        selected = "DejaVu Sans"
    matplotlib.rcParams.update({"font.family": selected, "pdf.fonttype": 42,
                                "ps.fonttype": 42, "axes.unicode_minus": False})
    return selected


def save_outputs(fig, output_dir, stem, results, config, manifest, font):
    import matplotlib
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        options = {"metadata": {"CreationDate": None, "ModDate": None}} if extension == "pdf" else {}
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight", **options)
    plt.close(fig)
    write_csv(output_dir / "figure5_model_summary.csv", results)
    write_csv(output_dir / "rank_change_summary.csv", sorted(results, key=lambda row: row["human_rank"]))
    write_json(output_dir / f"{stem}_provenance.json", {
        "figure": stem, "judge_model": config["judge_model"],
        "paired_responses": sum(row["n"] for row in results), "models": len(results),
        "score_definition": "100 * signed total / sum of positive rubric weights; no clipping or additional factual-error deduction",
        "aggregation": "Unweighted mean across eligible questions; sample SD / sqrt(n) for SE",
        "rank_definition": "Descending mean; competition ranking (method=min); positive rank_change means improvement with synthetic rubrics",
        "excluded_responses": config["excluded_responses"],
        "excluded_models": config["excluded_models_in_grade_files"],
        "human_scoring_corrections": config["scoring_corrections"],
        "review_note": config["review_note"],
        "comparison_note": "Same response inputs and judge; rubric sets and grading prompts differ",
        "paired_question_scores_sha256": manifest["paired_question_scores_sha256"],
        "config_sha256": manifest["config_sha256"],
        "font": font, "matplotlib": matplotlib.__version__,
        "source_code_sha256": {path.name: sha256(path) for path in sorted(HERE.glob("*.py"))},
        "outputs_sha256": {f"{stem}.{ext}": sha256(output_dir / f"{stem}.{ext}") for ext in ("pdf", "png")},
    })
    print(f"{stem}: {len(results)} models, {sum(row['n'] for row in results):,} paired responses; saved PDF and PNG to {output_dir}")
