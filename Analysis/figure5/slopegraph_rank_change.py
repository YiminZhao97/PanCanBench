#!/usr/bin/env python3
"""Figure 5b: rank changes from human to synthetic rubrics, colored by provider.

Adapted from the original Figure 5 slopegraph_rank_change.py. Positive rank_change
means the model ranks higher under synthetic rubrics. Ranks use unrounded means.
"""
from figure5_common import load_results, parser, save_outputs, setup_plotting


def main():
    args = parser(__doc__).parse_args()
    config, results, manifest = load_results(args.inputs_dir, args.config)
    font = setup_plotting(config)
    import matplotlib.pyplot as plt

    # Helvetica on macOS lacks arrow glyphs; use DejaVu Sans for those symbols.
    arrow_fonts = list(dict.fromkeys([font, "DejaVu Sans"]))
    fig, ax = plt.subplots(figsize=(15, 12))
    for row in sorted(results, key=lambda row: row["human_rank"]):
        color = config["provider_colors"][row["provider"]]
        human, synthetic = row["human_rank"], row["synthetic_rank"]
        ax.plot([0, 1], [human, synthetic], color=color, alpha=0.75, linewidth=2, zorder=1)
        ax.scatter([0, 1], [human, synthetic], color=color, s=80,
                   edgecolor="black", linewidth=0.5, zorder=2)
        ax.text(-0.035, human, f"{row['display_name']} (#{human})",
                ha="right", va="center", fontsize=13.5)
        change = row["rank_change"]
        indicator = f" ↑{change}" if change > 0 else f" ↓{abs(change)}" if change < 0 else " →"
        ax.text(1.035, synthetic, f"{row['display_name']} (#{synthetic}){indicator}",
                ha="left", va="center", fontsize=13.5, fontfamily=arrow_fonts)
    ax.set_xlim(-0.9, 2.05)
    ax.set_ylim(len(results) + 0.9, -0.1)
    ax.text(0, 0, "Human rubrics", ha="center", va="bottom", fontsize=17, fontweight="bold")
    ax.text(1, 0, "Synthetic rubrics", ha="center", va="bottom", fontsize=17, fontweight="bold")
    ax.set_title("Model Ranking: Human vs Synthetic Rubrics", fontsize=21, fontweight="bold", pad=26)
    ax.axis("off")
    fig.text(0.5, 0.024, "Change in rank with synthetic rubrics: ↑ higher; ↓ lower; → unchanged.",
             ha="center", fontsize=11, color="#444444", fontfamily=arrow_fonts)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_outputs(fig, args.output_dir, "figure5b", results, config, manifest, font)


if __name__ == "__main__":
    main()
