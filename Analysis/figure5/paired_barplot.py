#!/usr/bin/env python3
"""Figure 5a: paired human/synthetic rubric mean scores, with one standard error.

Adapted from the original Figure 5 paired_barplot.py. Preserves provider colors,
solid human bars, hatched synthetic bars, and ordering by human mean within family.
"""
from figure5_common import load_results, parser, save_outputs, setup_plotting


def main():
    args = parser(__doc__).parse_args()
    config, results, manifest = load_results(args.inputs_dir, args.config)
    font = setup_plotting(config)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(24, 8))
    x, positions, labels, centers, headings = 0.0, [], [], [], []
    width = 0.35
    for provider in config["provider_order"]:
        family = [row for row in results if row["provider"] == provider]
        start = x
        color = config["provider_colors"][provider]
        for row in family:
            for offset, kind, face, edge, hatch in (
                    (-width / 2, "human", color, "black", None),
                    (width / 2, "synthetic", "white", color, "///")):
                ax.bar(x + offset, row[f"{kind}_mean"], width=width,
                       color=face, edgecolor=edge, linewidth=0.8, hatch=hatch,
                       yerr=row[f"{kind}_se"], capsize=4,
                       error_kw={"linewidth": 1.3, "ecolor": "black"})
            positions.append(x)
            labels.append(row["display_name"])
            x += 1
        centers.append((start + x - 1) / 2)
        headings.append(config["provider_headings"][provider])
        x += 0.5

    ax.set_ylabel("Average score (%)", fontsize=15, fontweight="bold")
    ax.set_title("Average Score by Model: Human vs Synthetic Rubrics",
                 fontsize=21, fontweight="bold", pad=46)
    ax.set_xticks(positions, labels, rotation=45, ha="right", fontsize=13)
    # Include any genuinely negative mean instead of clipping it at zero.
    low = min(row[f"{kind}_mean"] - row[f"{kind}_se"] for row in results for kind in ("human", "synthetic"))
    high = max(row[f"{kind}_mean"] + row[f"{kind}_se"] for row in results for kind in ("human", "synthetic"))
    ax.set_ylim(min(0, low - 5), max(112, high + 14))
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    upper = ax.twiny()
    upper.set_xlim(ax.get_xlim())
    upper.set_xticks(centers, headings, fontsize=13, fontweight="bold")
    upper.tick_params(axis="x", length=0, pad=8)
    ax.legend(handles=[Patch(facecolor="gray", edgecolor="black", label="Human rubrics"),
                       Patch(facecolor="white", edgecolor="gray", hatch="///", label="Synthetic rubrics")],
              loc="upper right", fontsize=12, framealpha=0.95)
    fig.tight_layout()
    save_outputs(fig, args.output_dir, "figure5a", results, config, manifest, font)


if __name__ == "__main__":
    main()
