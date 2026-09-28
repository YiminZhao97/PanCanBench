#!/usr/bin/env python3
"""Reproduce Figure 4c from final saved grades with two-sided paired t-tests.

Adapts the original compare_websearch_scores.py layout. Reuses Figure 4a's
criterion/signed-score validation; no response generation or grading API calls.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import statistics

from plot_figure4a import load_grading_helpers, load_scoring_rubrics, require_close, validate_score


# Fixed citation-required subset used in the original Figure 4c, NOT the separate
# 40-question human/judge validation set. Verified against the original inputs.
QUESTION_NUMBERS = [
    58, 102, 113, 116, 122, 171, 173, 178, 179, 180, 181, 184, 185, 187,
    193, 197, 198, 199, 202, 203, 206, 207, 212, 219, 223, 226, 227, 229,
    237, 239, 241, 242, 244, 252, 253, 255, 259, 264, 276, 279,
]
MODELS = [
    {
        "id": "claude-sonnet-4-5-20250929", "label": "Claude-Sonnet-4.5",
        "baseline_source": "anthropic", "web_model": "claude-sonnet-4-5-20250929",
        "web_file": "claude-sonnet-4-5_websearch_response_graded.jsonl",
        "baseline_input_sha256": "62d9f1985e6f142079fe77f7a39c8a4d18e6e0deb18a53ff782528cf72baae08",
        "web_input_sha256": "f2f1f1d5c43226d29083eea620f035e12566ef026b58d3d4540aa8f9219cb4ba",
    },
    {
        "id": "gemini-2.5-pro", "label": "Gemini-2.5 Pro",
        "baseline_source": "gemini", "web_model": "gemini-2.5-pro",
        "web_file": "gemini-2.5-pro_websearch_response_graded.jsonl",
        "baseline_input_sha256": "d9744157386f2941be30cd62cb4d9c6f59eb41b0f6706608b5ac3702f7847fa4",
        "web_input_sha256": "8714fbc18633e5eb610ae35b6d28ca2489d4b5a5456dc879194fd071b73347b4",
    },
    {
        "id": "gpt-5", "label": "GPT-5",
        "baseline_source": "gpt5", "web_model": "gpt5",
        "web_file": "gpt5_websearch_response_graded.jsonl",
        "baseline_input_sha256": "bfffa95f4be0fd2a8f50166c7769bb2260d2e7c9b1e0a5ffc50eeb75feba0107",
        "web_input_sha256": "f78841c6a87fa7ed8eb2c225c67e72d697d69c45f41d888d8afff067e4383592",
    },
]
CONDITION_COLORS = {"Baseline": "#4472C4", "Web Search": "#ED7D31"}


def load_model_rows(path, model_id):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("response_model") != model_id:
                continue
            question_id = row["question_id"]
            if question_id in rows:
                raise ValueError(f"Duplicate {model_id}/{question_id} in {path.name}")
            rows[question_id] = row
    return rows


def compare_pairs(pairs):
    from scipy.stats import ttest_rel

    before = [pair["baseline_percentage"] for pair in pairs]
    after = [pair["web_search_percentage"] for pair in pairs]
    differences = [b - a for a, b in zip(before, after)]
    if len(pairs) != 40 or not all(math.isfinite(x) for x in before + after):
        raise ValueError("Expected 40 complete, finite question pairs")
    result = ttest_rel(after, before, alternative="two-sided", nan_policy="raise")
    mean_difference = statistics.mean(differences)
    difference_sd = statistics.stdev(differences)
    if difference_sd == 0:
        raise ValueError("Paired differences have zero variance; t-test is undefined")
    # Independent check of the t statistic: a one-sample test of the differences.
    t_manual = mean_difference / (difference_sd / math.sqrt(len(pairs)))
    require_close(float(result.statistic), t_manual, "paired t statistic")
    require_close(float(result.df), len(pairs) - 1, "paired t-test degrees of freedom")
    if not math.isfinite(float(result.pvalue)) or not 0 <= result.pvalue <= 1:
        raise ValueError("Invalid paired-test p-value")
    return {
        "n_pairs": len(pairs), "mean_baseline_percentage": statistics.mean(before),
        "mean_web_search_percentage": statistics.mean(after),
        "mean_difference_percentage_points": mean_difference,
        "t_statistic": float(result.statistic), "degrees_of_freedom": len(pairs) - 1,
        "p_value_two_sided_unadjusted": float(result.pvalue),
        "significant_at_0_05_unadjusted": bool(result.pvalue < 0.05),
    }


def analyze(data_dir, config, helpers):
    rubrics, rubric_paths = load_scoring_rubrics(data_dir, config, helpers)
    question_ids = [f"Q{q}" for q in QUESTION_NUMBERS]
    subset = set(question_ids)
    if len(subset) != 40 or sum(len(rubrics[q]["rubric_items"]) for q in subset) != 509:
        raise ValueError("Citation-required subset must contain 40 questions and 509 rubric items")
    inputs = [{"path": str(p.relative_to(data_dir)), "sha256": helpers.sha256(p)} for p in rubric_paths]
    results = []
    for model in MODELS:
        baseline_relative = config["grade_files"][model["baseline_source"]]
        web_relative = "grades/web_search/" + model["web_file"]
        baseline = load_model_rows(data_dir / baseline_relative, model["id"])
        web = load_model_rows(data_dir / web_relative, model["web_model"])
        if set(baseline) != {f"Q{q}" for q in range(1, 283)} or set(web) != subset:
            raise ValueError(f"Missing or unexpected question IDs for {model['id']}")
        pairs = []
        for q in question_ids:
            a, b = baseline[q], web[q]
            if a.get("input_sha256") != model["baseline_input_sha256"]:
                raise ValueError(f"Baseline response input changed for {model['id']}/{q}")
            if b.get("input_sha256") != model["web_input_sha256"]:
                raise ValueError(f"Web-search response input changed for {model['id']}/{q}")
            before = validate_score(a, rubrics, helpers, config)
            after = validate_score(b, rubrics, helpers, config)
            pairs.append({
                "question_id": q, "baseline_total_score": a["total_score"],
                "web_search_total_score": b["total_score"],
                "max_possible_score": a["max_possible_score"],
                "rubric_item_count": len(rubrics[q]["rubric_items"]),
                "baseline_percentage": before, "web_search_percentage": after,
                "difference_percentage_points": after - before,
            })
        results.append({"response_model": model["id"], "display_name": model["label"],
                        **compare_pairs(pairs), "pairs": pairs})
        for relative, expected_hash in [(baseline_relative, model["baseline_input_sha256"]),
                                        (web_relative, model["web_input_sha256"])]:
            inputs.append({"path": relative, "sha256": helpers.sha256(data_dir / relative),
                           "recorded_response_input_sha256": expected_hash})
    return results, inputs


def plot(results, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np

    font_paths = [font_manager.findfont(font_manager.FontProperties(
        family=config["font_family"], weight=weight), fallback_to_default=False)
        for weight in ("normal", "bold")]
    matplotlib.rcParams.update({"font.family": config["font_family"],
                               "pdf.fonttype": 42, "ps.fonttype": 42})
    x = np.arange(len(results))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))
    baseline = [r["mean_baseline_percentage"] for r in results]
    web = [r["mean_web_search_percentage"] for r in results]
    groups = [
        ax.bar(x - width / 2, baseline, width, label="Baseline", alpha=0.8,
               color=CONDITION_COLORS["Baseline"]),
        ax.bar(x + width / 2, web, width, label="Web Search", alpha=0.8,
               color=CONDITION_COLORS["Web Search"]),
    ]
    ax.set_xlabel("Model", fontsize=15, fontweight="bold")
    ax.set_ylabel("Average Score (%)", fontsize=15, fontweight="bold")
    ax.set_title("Model Performance: Baseline vs Web Search", fontsize=14, fontweight="bold")
    ax.set_xticks(x, [r["display_name"] for r in results], fontsize=15)
    ax.tick_params(axis="x", length=0)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, 100)
    for group in groups:
        for bar in group:
            ax.annotate(f"{bar.get_height():.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")
    for i, result in enumerate(results):
        p = result["p_value_two_sided_unadjusted"]
        marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        y = max(baseline[i], web[i]) + 5
        ax.plot([x[i] - width / 2, x[i] + width / 2], [y, y], color="black", linewidth=1.5)
        p_text = f"p={p:.4f}" if p >= 0.001 else "p<0.001"
        ax.text(x[i], y + 2, f"{p_text}\n{marker}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig, list(dict.fromkeys(font_paths))


def main():
    folder = Path(__file__).resolve().parent
    root = folder.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "Data")
    parser.add_argument("--grading-code-dir", type=Path, default=root / "Evaluation/rubric_scoring")
    parser.add_argument("--config", type=Path, default=folder / "model_config.json")
    parser.add_argument("--output-dir", type=Path, default=root / "Outputs/figure4")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    helpers = load_grading_helpers(args.grading_code_dir)
    results, inputs = analyze(args.data_dir, config, helpers)
    fig, fonts = plot(results, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "figure4c.pdf", bbox_inches="tight",
                metadata={"Title": "PanCanBench Figure 4c", "CreationDate": None})
    fig.savefig(args.output_dir / "figure4c.png", dpi=300, bbox_inches="tight")
    report = {
        "figure": "4c", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "judge_model": config["judge_model"], "rubrics_sha256": config["rubrics_sha256"],
        "scoring_rubrics_sha256": config.get("scoring_rubrics_sha256", config["rubrics_sha256"]),
        "scoring_corrections": config.get("scoring_corrections", []),
        "questions_per_model": 40, "rubric_items_per_condition_per_model": 509,
        "question_ids": [f"Q{q}" for q in QUESTION_NUMBERS],
        "score_definition": "100 * signed rubric points / positive rubric-point maximum",
        "aggregation": "arithmetic mean of question-level percentages; no factual-error deduction",
        "test": {"name": "paired t-test", "alternative": "two-sided", "pairing_key": "question_id",
                 "difference": "web search minus baseline", "alpha": 0.05,
                 "multiple_comparison_adjustment": "none, matching the original Figure 4c analysis",
                 "implementation": "scipy.stats.ttest_rel(after, before, alternative='two-sided', nan_policy='raise')"},
        "results": results, "inputs": inputs,
        "checks": ["same fixed 40 question IDs in all three comparisons", "all 240 score records criterion-validated",
                   "509 rubric items per model per condition", "recorded input hashes match audited original response inputs",
                   "signed scores, totals, and normalization verified", "t statistic independently checked from paired differences"],
        "plot_settings": {"template": "compare_websearch_scores.py", "size_inches": [12, 7],
                          "y_limits": [0, 100], "bar_width": 0.35, "condition_colors": CONDITION_COLORS,
                          "color_alpha": 0.8, "legend": "upper left", "error_bars": "none, as in original",
                          "font": config["font_family"], "model_order": [m["id"] for m in MODELS]},
        "software": {"python": platform.python_version(),
                     "packages": {p: version(p) for p in ("matplotlib", "numpy", "scipy", "fonttools", "pillow")},
                     "font_files": [{"filename": Path(p).name, "sha256": helpers.sha256(Path(p))} for p in fonts]},
        "code": [{"file": p.name, "sha256": helpers.sha256(p)} for p in
                 (Path(__file__), folder / "plot_figure4a.py", args.config,
                  args.grading_code_dir / "grade_anthropic_batch.py")],
        "outputs": [{"file": name, "sha256": helpers.sha256(args.output_dir / name)}
                    for name in ("figure4c.pdf", "figure4c.png")],
    }
    (args.output_dir / "figure4c_results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for r in results:
        print(f"{r['display_name']}: {r['mean_baseline_percentage']:.4f} -> "
              f"{r['mean_web_search_percentage']:.4f}; paired p={r['p_value_two_sided_unadjusted']:.8f}; n={r['n_pairs']}")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
