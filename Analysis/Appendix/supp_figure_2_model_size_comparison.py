#!/usr/bin/env python3
"""Reproduce Supplementary Figure 2 from saved Phase 2 model grades."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import arguments, keyed_records, provenance, read_json, score, validate_inputs, write_csv

EXPERTS = {
    "Expert-A": ("Sheela", (1, 3)), "Expert-B": ("Manuel", (1, 4)),
    "Expert-C": ("Simone", (2, 3)), "Expert-D": ("Jesse", (2, 5)),
    "Expert-E": ("Karly", (4, 5)),
}
PAIRS = {
    "Llama-3.1-8B vs Llama-3.1-70B": ("meta-llama_Llama-3.1-8B-Instruct", "meta-llama_Llama-3.1-70B-Instruct"),
    "Qwen3-8B vs Qwen3-32B": ("Qwen_Qwen3-8B", "Qwen_Qwen3-32B"),
}
STEM = "supp_figure_2_model_size_comparison"


def calculate(directory):
    summary, comparisons = [], []
    for expert, (name, folds) in EXPERTS.items():
        raw = []
        for fold in folds:
            raw.extend(read_json(directory / f"opensource_2pairs_scores_fold{fold}_{name}.json"))
        records = keyed_records(raw)
        expected_n = 112 if expert == "Expert-E" else 113
        for pair, (small, large) in PAIRS.items():
            small_keys = {q for q, model in records if model == small}
            large_keys = {q for q, model in records if model == large}
            if small_keys != large_keys or len(small_keys) != expected_n:
                raise ValueError(f"Unpaired or unexpected question cohort: {expert}, {pair}")
            win, tie, loss = 0, 0, 0
            for q in sorted(small_keys):
                small_score, large_score = score(records[q, small]), score(records[q, large])
                # Keep the original comparison exactly: ties count as success.
                win += large_score > small_score
                tie += large_score == small_score
                loss += large_score < small_score
                comparisons.append({"expert": expert, "pair": pair, "question_id": f"Q{q}",
                                    "small_score": small_score, "large_score": large_score,
                                    "large_at_least_small": large_score >= small_score})
            summary.append({"expert": expert, "pair": pair, "n_questions": expected_n,
                            "large_wins": win, "ties": tie, "small_wins": loss,
                            "large_at_least_small": win + tie,
                            "percentage": 100 * (win + tie) / expected_n})
    return summary, comparisons


def plot(summary, out):
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for pair, offset, color in zip(PAIRS, (-0.175, 0.175), ("#4C72B0", "#DD8452")):
        values = [next(r["percentage"] for r in summary if r["expert"] == expert and r["pair"] == pair)
                  for expert in EXPERTS]
        bars = ax.bar([i + offset for i in range(5)], values, 0.35, label=pair, color=color)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=11)
    ax.axhline(50, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="50% reference")
    ax.set_xticks(range(5), list(EXPERTS), fontsize=12)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Expert", fontsize=12)
    ax.set_ylabel("Questions where large model scores ≥ small model (%)", fontsize=12)
    ax.set_title("Comparison of Large vs Small Model Performance Across Experts", fontsize=14, pad=12)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(out / f"{STEM}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = arguments(__doc__)
    inputs = validate_inputs(args.root, "supp_figure_2")
    summary, comparisons = calculate(args.root / "Data/appendix/supp_figure_2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"{STEM}.csv", summary)
    write_csv(args.output_dir / f"{STEM}_questions.csv", comparisons)
    plot(summary, args.output_dir)
    provenance(args.output_dir, STEM, inputs, __file__, {
        "figure": "Supplementary Figure 2", "n_comparisons": len(comparisons),
        "definition": "100 * count(large score >= small score) / paired questions; includes ties.",
        "score_formula": "100 * sum(score_given) / sum(max_points) in historical Phase 2 grades",
        "expert_mapping": EXPERTS, "model_pairs": PAIRS,
        "label_correction": "Legacy script said Qwen2.5; actual inputs and current appendix use Qwen3."})
    print(f"Supplementary Figure 2: {len(comparisons)} paired comparisons; {args.output_dir}")


if __name__ == "__main__":
    main()
