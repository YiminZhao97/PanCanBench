#!/usr/bin/env python3
"""Reproduce Figure 4b from saved factual-error counts, without grading/API calls.

Adapts create_wide_upset.py's category matrix and the factual-error panel in
combined_stacked_barplot.py. Uses Figure 4a's saved order and shared palette.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform


CATEGORY_TOTALS = {
    "Diagnosis & Screening": 20,
    "Genetics & Risk": 11,
    "Mechanistic / Research Topics": 67,
    "Side Effects & Symptom Management": 53,
    "Supportive & Palliative Care": 79,
    "Treatment & Clinical Trials": 52,
}
CATEGORY_LABELS = [
    "Diagnosis &\nScreening", "Genetics & Risk", "Mechanistic /\nResearch Topics",
    "Side Effects &\nSymptom Management", "Supportive &\nPalliative Care",
    "Treatment &\nClinical Trials",
]
# Only names that differ between the saved factual-error data and final grades.
ALIASES = {
    "GPT_o3": "o3", "GPT_o4-mini": "o4-mini", "gpt-4_1": "gpt-4.1",
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
}
# Preserve the screenshot's compact labels, rather than Figure 4a's full IDs.
LABELS = {
    "o3": "o3", "gpt-5": "GPT-5", "o4-mini": "o4-mini",
    "gpt-4.1": "GPT-4.1", "gpt-4o": "GPT-4o",
    "gemini-2.5-flash": "Gemini-2.5 Flash", "gemini-2.5-pro": "Gemini-2.5 Pro",
    "google_gemma-3-27b-it": "Gemma-3-27b-it",
    "google_gemma-3-12b-it": "Gemma-3-12b-it", "grok-4-latest": "Grok-4-latest",
    "claude-opus-4-1-20250805": "Claude-Opus-4.1",
    "claude-sonnet-4-5-20250929": "Claude-Sonnet-4.5",
    "claude-opus-4-20250514": "Claude-Opus-4",
    "claude-sonnet-4-20250514": "Claude-Sonnet-4",
    "claude-haiku-4-5-20251001": "Claude-Haiku-4.5",
    "meta-llama_Llama-3.1-70B-Instruct": "Llama-3.1-70B-Instruct",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "allenai_olmo-3-32b-think": "OLMo-3-32b-think",
    "allenai_olmo-3.1-32b-instruct": "OLMo-3.1-32b-instruct",
    "Qwen_Qwen3-32B": "Qwen3-32B", "Qwen_Qwen3-8B": "Qwen3-8B",
    "Qwen_Qwen3-14B": "Qwen3-14B",
}
INPUT_HASHES = {
    "unique_question_counts.csv": "a40a46d82dff7d677467e66f8571bb158a6d705d2b30dcc8ede0656829798fcd",
    "factual_errors_by_category.csv": "e85c9d36feba8101cf7a5f9cc7bcf728c30aa9c2576dab9f2d9bdc14538f555a",
}
TOTAL_QUESTIONS = 282  # Retain the baseline definition, including for Opus 4.1.
THRESHOLD_PERCENT = 10


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_rate(value, expected, context):
    require(math.isfinite(value) and math.isclose(value, expected, abs_tol=1e-10),
            f"Inconsistent rate: {context}")


def above_threshold(count, denominator):
    # Integer comparison ensures that exactly 10% is NOT marked active.
    return 100 * count > THRESHOLD_PERCENT * denominator


def load_data(input_dir, config, order_provenance):
    for name, digest in INPUT_HASHES.items():
        require(sha256(input_dir / name) == digest, f"Changed factual-error input: {name}")
    models = {model["id"]: model for model in config["models"]}
    order = order_provenance["model_order"]
    require(len(order) == len(set(order)) == len(models) == 22,
            "Figure 4a must supply exactly 22 unique model IDs")
    require(set(order) == set(models) == set(LABELS), "Figure 4a/4b cohorts differ")
    providers = [models[model]["provider"] for model in order]
    require(list(dict.fromkeys(providers)) == config["provider_order"], "Family order mismatch")
    require(providers == sorted(providers, key=config["provider_order"].index),
            "Models must be contiguous within families")
    require(order_provenance["rubrics_sha256"] == config["rubrics_sha256"],
            "Figure 4a order uses a different rubric version")

    overall = {}
    categories = {}
    ignored = set()
    allowed_ignored = set(config["excluded_models_in_grade_files"]) | {"OpenEvidence"}
    for row in read_csv(input_dir / "unique_question_counts.csv"):
        model = ALIASES.get(row["Model"], row["Model"])
        if model not in models:
            require(model in allowed_ignored, f"Unknown model: {model}")
            ignored.add(model)
            continue
        require(model not in overall, f"Duplicate overall row: {model}")
        count = int(row["Unique_Question_Count"])
        require(0 <= count <= TOTAL_QUESTIONS, f"Invalid overall count: {model}")
        check_rate(float(row["Percentage"]), count / TOTAL_QUESTIONS, model)
        overall[model] = {"source_model": row["Model"], "count": count}
    require(set(overall) == set(models), "Missing overall model rows")
    for row in read_csv(input_dir / "factual_errors_by_category.csv"):
        model = ALIASES.get(row["Model"], row["Model"])
        if model not in models:
            require(model in allowed_ignored, f"Unknown model: {model}")
            continue
        category = row["Category"]
        require(category in CATEGORY_TOTALS, f"Unknown category: {category}")
        key = (model, category)
        require(key not in categories, f"Duplicate category row: {key}")
        count, total = int(row["Count"]), int(row["Total_Questions_in_Category"])
        require(total == CATEGORY_TOTALS[category] and 0 <= count <= total,
                f"Invalid category counts: {key}")
        check_rate(float(row["Percentage"]), 100 * count / total, str(key))
        categories[key] = count

    summary, matrix = [], []
    x = 0
    for position, model in enumerate(order, 1):
        provider = models[model]["provider"]
        if position > 1 and provider != providers[position - 2]:
            x += 0.5
        count = overall[model]["count"]
        # The original category exporter omits zero-count categories. Completing
        # them with zeros is checked against the model's independent overall total.
        require(sum(categories.get((model, cat), 0) for cat in CATEGORY_TOTALS) == count,
                f"Category counts do not reconcile with overall total: {model}")
        common = {"plot_order": position, "response_model": model,
                  "display_name": LABELS[model], "provider": provider,
                  "color": config["provider_colors"][provider], "x_position": x}
        summary.append({**common, "source_model": overall[model]["source_model"],
                        "questions_with_factual_errors": count,
                        "denominator_questions": TOTAL_QUESTIONS,
                        "percentage": 100 * count / TOTAL_QUESTIONS})
        for category, total in CATEGORY_TOTALS.items():
            c = categories.get((model, category), 0)
            matrix.append({**common, "category": category, "questions_with_factual_errors": c,
                           "denominator_questions": total, "percentage": 100 * c / total,
                           "above_10_percent": above_threshold(c, total)})
        x += 1
    return summary, matrix, sorted(ignored)


def audit_legacy(overlap_dir, summary, matrix):
    """Optional read-only reconciliation against the original question-level flags."""
    classification = overlap_dir / "upset/question_classification_gemini.csv"
    rows = read_csv(classification)
    qcat = {r["id"]: r["category"] for r in rows}
    require(len(rows) == len(qcat) == TOTAL_QUESTIONS, "Invalid classification coverage")
    require(set(qcat) == {f"Q{i}" for i in range(1, TOTAL_QUESTIONS + 1)}, "Question ID gap")
    require(Counter(qcat.values()) == CATEGORY_TOTALS, "Category denominators changed")
    hashes = [{"file": "upset/question_classification_gemini.csv", "sha256": sha256(classification)}]
    for row in summary:
        path = overlap_dir / f"overlap_{row['source_model']}.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        ids = {record["question_id"] for record in records if "question_id" in record}
        require(ids <= set(qcat), f"Unclassified factual-error question: {path.name}")
        require(len(ids) == row["questions_with_factual_errors"], f"Count changed: {path.name}")
        counts = Counter(qcat[q] for q in ids)
        for cell in matrix:
            if cell["response_model"] == row["response_model"]:
                require(counts[cell["category"]] == cell["questions_with_factual_errors"],
                        f"Category count changed: {path.name}, {cell['category']}")
        hashes.append({"file": path.name, "sha256": sha256(path)})
    return {"status": "all 22 models match the original question-level flags", "files": hashes}


def plot(summary, matrix, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_paths = [font_manager.findfont(font_manager.FontProperties(
        family=config["font_family"], weight=weight), fallback_to_default=False)
        for weight in ("normal", "bold")]
    matplotlib.rcParams.update({"font.family": config["font_family"],
                               "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (bars, dots) = plt.subplots(2, 1, figsize=(18, 9.7), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 0.035})
    fig.subplots_adjust(left=0.15, right=0.99, top=0.93, bottom=0.26)
    xs = [row["x_position"] for row in summary]
    bars.bar(xs, [row["percentage"] for row in summary], width=0.6,
             color=[row["color"] for row in summary], edgecolor="black", linewidth=0.8)
    bars.set_ylim(0, max(row["percentage"] for row in summary) * 1.15)
    bars.set_yticks(range(0, 61, 10))
    bars.set_ylabel("Percentage (%)", fontsize=20)
    bars.set_title("Percentage of Responses with Factual Errors", fontsize=22, pad=14)
    bars.tick_params(axis="y", labelsize=18)
    bars.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    bars.grid(axis="y", alpha=0.3, linestyle="--")
    bars.set_axisbelow(True)

    for cell in matrix:
        y = list(CATEGORY_TOTALS).index(cell["category"])
        active = cell["above_10_percent"]
        dots.scatter(cell["x_position"], y, s=70 if active else 30,
                     c=cell["color"] if active else "lightgray",
                     edgecolors="black" if active else "gray",
                     linewidths=0.9 if active else 0.4,
                     alpha=1 if active else 0.13, zorder=3 if active else 2)
    dots.set_ylim(5.5, -0.5)  # Preserve screenshot's top-to-bottom category order.
    dots.set_yticks(range(6), CATEGORY_LABELS, fontsize=16)
    dots.set_xticks(xs, [row["display_name"] for row in summary],
                   rotation=45, ha="right", fontsize=18)
    dots.set_xlabel("Model", fontsize=16, labelpad=12)
    dots.tick_params(axis="both", length=0)
    dots.grid(axis="x", color="gray", alpha=0.15, linewidth=0.5)
    for spine in dots.spines.values():
        spine.set_visible(False)
    software = {"python": platform.python_version(),
                "packages": {p: version(p) for p in ("matplotlib", "numpy", "fonttools", "pillow")},
                "font_files": [{"filename": Path(p).name, "sha256": sha256(Path(p))}
                               for p in dict.fromkeys(font_paths)]}
    return fig, software


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    folder = Path(__file__).resolve().parent
    root = folder.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=folder / "inputs/figure4b")
    parser.add_argument("--config", type=Path, default=folder / "model_config.json")
    parser.add_argument("--figure4a-provenance", type=Path, default=root / "Outputs/figure4/figure4a_provenance.json")
    parser.add_argument("--output-dir", type=Path, default=root / "Outputs/figure4")
    parser.add_argument("--legacy-overlap-dir", type=Path, help="Optional independent source audit; never changes inputs")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    order_source = json.loads(args.figure4a_provenance.read_text())
    summary, matrix, ignored = load_data(args.input_dir, config, order_source)
    audit = audit_legacy(args.legacy_overlap_dir, summary, matrix) if args.legacy_overlap_dir else None
    fig, software = plot(summary, matrix, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "figure4b.pdf", bbox_inches="tight",
                metadata={"Title": "PanCanBench Figure 4b", "CreationDate": None})
    fig.savefig(args.output_dir / "figure4b.png", dpi=300, bbox_inches="tight")
    write_csv(args.output_dir / "figure4b_model_summary.csv", summary)
    write_csv(args.output_dir / "figure4b_category_matrix.csv", matrix)
    provenance = {
        "figure": "4b", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_order": order_source["model_order"], "provider_colors": config["provider_colors"],
        "category_denominators": CATEGORY_TOTALS, "models": len(summary),
        "category_cells": len(matrix), "active_cells": sum(c["above_10_percent"] for c in matrix),
        "overall_rate": "100 * unique questions flagged for factual errors / 282",
        "category_rate": "100 * unique flagged questions in category / all questions in category",
        "dot_rule": "strictly greater than 10%; exactly 10% is inactive",
        "denominator_note": "Baseline denominator retained: 282 for every model, including Opus 4.1. This is not Figure 4a's available-response mean denominator.",
        "factual_error_deduction_from_rubric_scores": False,
        "grading_rerun": False, "ignored_source_models": ignored,
        "plot_settings": {"size_inches": [18, 9.7], "bar_width": 0.6,
                          "family_gap": 0.5, "shared_x_positions": True,
                          "font": config["font_family"], "x_tick_marks_visible": False},
        "inputs": [{"file": name, "sha256": sha256(args.input_dir / name)} for name in INPUT_HASHES],
        "order_source": {"file": args.figure4a_provenance.name, "sha256": sha256(args.figure4a_provenance)},
        "code": [{"file": Path(__file__).name, "sha256": sha256(Path(__file__))},
                 {"file": args.config.name, "sha256": sha256(args.config)}],
        "checks": ["22 models and 132 cells", "input snapshots match pinned hashes",
                   "rates reconcile with numerators and denominators",
                   "category counts reconcile with each overall count", "order matches Figure 4a"],
        "legacy_source_audit": audit, "software": software,
        "outputs": [{"file": name, "sha256": sha256(args.output_dir / name)} for name in
                    ("figure4b.pdf", "figure4b.png", "figure4b_model_summary.csv", "figure4b_category_matrix.csv")],
    }
    (args.output_dir / "figure4b_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Validated {len(summary)} models and {len(matrix)} category cells; "
          f"{provenance['active_cells']} active dots. Factual-error results unchanged.")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
