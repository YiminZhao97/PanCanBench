#!/usr/bin/env python3
"""Figure 5b: rank changes from human to synthetic rubrics, colored by provider.

Adapted from the original Figure 5 slopegraph_rank_change.py. Positive rank_change
means the model ranks higher under synthetic rubrics. Ranks use unrounded means.
Arrows are vector line paths, so Illustrator need not resolve an arrow font.
"""
from figure5_common import load_results, parser, save_outputs, setup_plotting


def vector_arrow(direction, fontsize, color="black"):
    """Create a fixed-size open arrow from two polylines, never a text glyph."""
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import DrawingArea

    scale = fontsize / 13.5
    area = DrawingArea(10 * scale, 12 * scale, 0, 0)
    points = {
        "up": ([(5, 1.5), (5, 10.5)], [(2.5, 8), (5, 10.5), (7.5, 8)]),
        "down": ([(5, 10.5), (5, 1.5)], [(2.5, 4), (5, 1.5), (7.5, 4)]),
        "right": ([(1, 6), (9, 6)], [(6.5, 8.5), (9, 6), (6.5, 3.5)]),
    }
    for segment in points[direction]:
        x, y = zip(*segment)
        area.add_artist(Line2D([v * scale for v in x], [v * scale for v in y],
                               color=color, linewidth=0.9 * scale,
                               solid_capstyle="round", solid_joinstyle="miter"))
    return area


def rank_label(row, fontsize):
    from matplotlib.offsetbox import HPacker, TextArea

    change = row["rank_change"]
    direction = "up" if change > 0 else "down" if change < 0 else "right"
    children = [TextArea(f"{row['display_name']} (#{row['synthetic_rank']})",
                         textprops={"fontsize": fontsize}),
                vector_arrow(direction, fontsize)]
    if change:
        children.append(TextArea(str(abs(change)), textprops={"fontsize": fontsize}))
    return HPacker(children=children, align="center", pad=0, sep=2.5)


def main():
    args = parser(__doc__).parse_args()
    config, results, manifest = load_results(args.inputs_dir, args.config)
    font = setup_plotting(config)
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

    fig, ax = plt.subplots(figsize=(15, 12))
    for row in sorted(results, key=lambda row: row["human_rank"]):
        color = config["provider_colors"][row["provider"]]
        human, synthetic = row["human_rank"], row["synthetic_rank"]
        ax.plot([0, 1], [human, synthetic], color=color, alpha=0.75, linewidth=2, zorder=1)
        ax.scatter([0, 1], [human, synthetic], color=color, s=80,
                   edgecolor="black", linewidth=0.5, zorder=2)
        ax.text(-0.035, human, f"{row['display_name']} (#{human})",
                ha="right", va="center", fontsize=13.5)
        ax.add_artist(AnnotationBbox(rank_label(row, 13.5), (1.035, synthetic),
                                    xycoords="data", box_alignment=(0, 0.5),
                                    frameon=False, pad=0, annotation_clip=False))
    ax.set_xlim(-0.9, 2.05)
    ax.set_ylim(len(results) + 0.9, -0.1)
    ax.text(0, 0, "Human rubrics", ha="center", va="bottom", fontsize=17, fontweight="bold")
    ax.text(1, 0, "Synthetic rubrics", ha="center", va="bottom", fontsize=17, fontweight="bold")
    ax.set_title("Model Ranking: Human vs Synthetic Rubrics", fontsize=21, fontweight="bold", pad=26)
    ax.axis("off")
    legend_props = {"fontsize": 11, "color": "#444444"}
    legend_items = [TextArea("Change in rank with synthetic rubrics:", textprops=legend_props)]
    for direction, label in (("up", "higher;"), ("down", "lower;"), ("right", "unchanged.")):
        legend_items.extend([vector_arrow(direction, 11, "#444444"),
                             TextArea(label, textprops=legend_props)])
    legend = HPacker(children=legend_items, align="center", pad=0, sep=3)
    fig.add_artist(AnnotationBbox(legend, (0.5, 0.024), xycoords=fig.transFigure,
                                 box_alignment=(0.5, 0.5), frameon=False, pad=0,
                                 annotation_clip=False))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_outputs(fig, args.output_dir, "figure5b", results, config, manifest, font)


if __name__ == "__main__":
    main()
