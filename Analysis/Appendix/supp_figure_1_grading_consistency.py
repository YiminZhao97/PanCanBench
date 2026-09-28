#!/usr/bin/env python3
"""Reproduce Supplementary Figure 1 from saved original and polished grades."""

import csv
import math
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import arguments, keyed_records, provenance, read_json, score, validate_inputs, write_csv

MODELS = ("gpt-4o", "grok-4-latest", "meta-llama_Llama-3.1-70B-Instruct")
MAPPINGS = {
    "Expert-A": ((1, 2), (3, 1)),
    "Expert-B": ((2, 1), (4, 2)),
    "Expert-C": ((2, 2), (3, 2)),
    "Expert-D": ((1, 1), (5, 1)),
    "Expert-E": ((4, 1), (5, 2)),
}
POLISHED = {
    1: "20250918_165429", 2: "20250918_164312", 3: "20250916_175733",
    4: "20250918_163539", 5: "20250918_112946",
}
STEM = "supp_figure_1_grading_consistency"


def calculate(directory):
    rows = []
    for expert, folds in MAPPINGS.items():
        for fold, slot in folds:
            folder = directory / f"fold{fold}"
            scores_dir = folder / "gpt_claude_scores" if fold == 3 else folder
            gpt = keyed_records(read_json(scores_dir / f"fold{fold}_gpt_scores_expert{slot}.json"))
            claude = keyed_records(read_json(scores_dir / f"fold{fold}_claude_scores_expert{slot}.json"))
            polished = read_json(folder / "polished_grading_comparison" /
                                 f"polished_rubrics_grading_analysis_{POLISHED[fold]}.json")
            improvements = polished[f"expert{slot}_improvements"]
            keys = {key for key in gpt if key[1] in MODELS}
            if keys != {key for key in claude if key[1] in MODELS}:
                raise ValueError(f"Unpaired original grades: {expert}, fold{fold}")
            for q, model in sorted(keys):
                original = score(gpt[q, model]) - score(claude[q, model])
                updated = original
                result = improvements.get(f"Q{q}", {}).get(model, {}).get("grading_result", {})
                used = "gpt_grading" in result and "claude_grading" in result
                if used:
                    updated = score(result["gpt_grading"]) - score(result["claude_grading"])
                rows.append({"Expert": expert, "Fold": f"fold{fold}", "Question_ID": f"Q{q}",
                             "Model": model, "Difference_Original": original,
                             "Difference_Polished": updated, "Has_Polished_Scores": used})
    # Independently compare all paired differences with the historical export.
    with (directory / "detailed_question_data.csv").open(newline="") as handle:
        expected_rows = list(csv.DictReader(handle))
    key = lambda r: (r["Expert"], r["Fold"], r["Question_ID"], r["Model"])
    expected = {key(r): r for r in expected_rows}
    if len(rows) != 1692 or len(expected) != len(expected_rows) or {key(r) for r in rows} != set(expected):
        raise ValueError("Figure 1 must match the historical 1,692 comparisons")
    for row in rows:
        reference = expected[key(row)]
        for field in ("Difference_Original", "Difference_Polished"):
            if not math.isclose(row[field], float(reference[field]), rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"Historical difference mismatch: {key(row)}, {field}")
        if row["Has_Polished_Scores"] != (reference["Has_Polished_Scores"] == "True"):
            raise ValueError(f"Polished-score availability mismatch: {key(row)}")
    summary = []
    for expert in MAPPINGS:
        for model in MODELS:
            selected = [r for r in rows if r["Expert"] == expert and r["Model"] == model]
            expected_n = 112 if expert == "Expert-E" else 113
            if len(selected) != expected_n or len({r["Question_ID"] for r in selected}) != expected_n:
                raise ValueError(f"Unexpected question cohort: {expert} {model}")
            for label, field in (("Original", "Difference_Original"), ("Polished", "Difference_Polished")):
                summary.append({"Expert": expert, "Model": model, "Rubric_Type": label,
                                "Abs_Average_Difference": mean(abs(r[field]) for r in selected),
                                "N_Questions": len(selected)})
    return summary, rows


def plot(summary, out):
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    maximum = max(r["Abs_Average_Difference"] for r in summary) * 1.12
    for ax, expert in zip(axes, MAPPINGS):
        for label, offset, color in (("Original", -0.175, "#ff7f7f"), ("Polished", 0.175, "#7f7fff")):
            values = [next(r["Abs_Average_Difference"] for r in summary
                           if r["Expert"] == expert and r["Model"] == model and r["Rubric_Type"] == label)
                      for model in MODELS]
            bars = ax.bar([i + offset for i in range(3)], values, 0.35,
                          label=f"{label} Rubrics", color=color, edgecolor="black", alpha=0.8)
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
        ax.set_title(expert, fontsize=12)
        ax.set_xticks(range(3), ["GPT-4o", "Grok-4", "Llama-3.1\n-70B"], fontsize=10)
        ax.set_ylim(0, maximum)
        ax.grid(axis="y", alpha=0.5, linestyle="--")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean absolute score difference\n(percentage points)", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Grading Consistency Improvement Across All Experts", fontsize=16)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(out / f"{STEM}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = arguments(__doc__)
    inputs = validate_inputs(args.root, "supp_figure_1")
    summary, rows = calculate(args.root / "Data/appendix/supp_figure_1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"{STEM}.csv", summary)
    write_csv(args.output_dir / f"{STEM}_questions.csv", rows)
    plot(summary, args.output_dir)
    provenance(args.output_dir, STEM, inputs, __file__, {
        "figure": "Supplementary Figure 1", "n_comparisons": len(rows),
        "definition": "Mean absolute difference in GPT-4.1 and Claude Sonnet 4 percentage scores across questions; not kappa or binary agreement.",
        "unchanged_questions": "Use original difference when no paired polished grades exist.",
        "historical_pairwise_export_checked": True, "expert_fold_slots": MAPPINGS,
        "question_output": f"{STEM}_questions.csv"})
    print(f"Supplementary Figure 1: verified {len(rows)} comparisons; {args.output_dir}")


if __name__ == "__main__":
    main()
