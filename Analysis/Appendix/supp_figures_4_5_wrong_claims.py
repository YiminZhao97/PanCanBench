#!/usr/bin/env python3
"""Plot wrong-claim counts and percentages by model (Supplementary Figures 4–5).

Uses frozen claim-level count tables, not rubric scores or counts of responses
with errors. Optional --factuality-dir audits these tables against source JSONs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {
    "GPT Family (OpenAI)": ("#0072B2", ["gpt-5", "gpt-4o", "gpt-4_1", "o4-mini", "o3"]),
    "Gemini/Gemma (Google)": ("#E69F00", ["gemini-2.5-flash", "gemini-2.5-pro", "google_gemma-3-27b-it", "google_gemma-3-12b-it"]),
    "Grok (xAI)": ("#009E73", ["grok-4-latest"]),
    "Claude (Anthropic)": ("#D55E00", ["claude-opus-4-1-20250805", "claude-opus-4", "claude-sonnet-4-5", "claude-sonnet-4", "claude-haiku-4-5"]),
    "Llama (Meta)": ("#CC79A7", ["meta-llama_Llama-3.1-70B-Instruct", "meta-llama_Llama-3.1-8B-Instruct"]),
    "OLMo (Ai2)": ("#56B4E9", ["allenai_olmo-3-32b-think", "allenai_olmo-3.1-32b-instruct"]),
    "Qwen (Alibaba)": ("#F0E442", ["Qwen_Qwen3-32B", "Qwen_Qwen3-14B", "Qwen_Qwen3-8B"]),
}
ALIASES = {"GPT_o3": "o3", "GPT_o4-mini": "o4-mini"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_counts(path, model_column, value_column):
    values = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            model = ALIASES.get(row[model_column], row[model_column])
            if model in values:
                raise ValueError(f"Duplicate model in {path.name}: {model}")
            count = int(row[value_column])
            if count < 0:
                raise ValueError(f"Negative count: {model}")
            values[model] = count
    return values


def audit_sources(directory, errors, claims):
    """Recount the legacy numerator and denominator, retaining their definitions."""
    hashes = {}
    reverse = {v: k for k, v in ALIASES.items()}
    for model in errors:
        source = reverse.get(model, model)
        overlap = directory / "overlap" / f"overlap_{source}.json"
        gemini = directory / "gemini_final_res" / f"factuality_judgment_results_{source}_claimid.json"
        raw_errors = json.loads(overlap.read_text())
        raw_claims = json.loads(gemini.read_text())
        if not isinstance(raw_errors, list) or not isinstance(raw_claims, list):
            raise ValueError(f"Expected JSON arrays for {model}")
        count = sum(len(row["atomic_claims_evaluation"]) for row in raw_claims)
        if len(raw_errors) != errors[model] or count != claims[model]:
            raise ValueError(f"Source counts disagree with saved figure inputs: {model}")
        for path in (overlap, gemini):
            hashes[str(path.relative_to(directory))] = sha256(path)
    return hashes


def plot(rows, metric, output):
    plt.rcdefaults()
    plt.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(16, 6))
    ordered, boundaries = [], []
    position = 0
    for family, (color, _) in GROUPS.items():
        # Stable ties preserve the legacy count table's row order (e.g. Gemini).
        group = sorted((r for r in rows if r["family"] == family), key=lambda r: r[metric])
        start = position
        for row in group:
            ordered.append({**row, "plot_position": position})
            position += 1
        boundaries.append((start, position - 0.5, family))
        position += 0.5
    heights = [r[metric] for r in ordered]
    ax.bar([r["plot_position"] for r in ordered], heights,
           color=[r["color"] for r in ordered], edgecolor="black", linewidth=0.5)
    ax.set_xticks([r["plot_position"] for r in ordered])
    ax.set_xticklabels([r["model"] for r in ordered], rotation=45, ha="right", fontsize=9)
    maximum = max(heights)
    for start, end, family in boundaries:
        ax.text((start + end) / 2, maximum * 1.02, family,
                ha="center", fontsize=10, fontweight="bold")
    count_plot = metric == "wrong_claims"
    ax.set_ylabel("Error Count" if count_plot else "Percentage of Wrong Claims (%)",
                  fontsize=12, fontweight="bold")
    ax.set_title("Factual Error Count by Model (Grouped by Company)" if count_plot else
                 "Percentage of Factually Wrong Claims by Model (Grouped by Company)",
                 fontsize=14, fontweight="bold", pad=40)
    ax.set_ylim(0, maximum * 1.08)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Root containing Data/appendix/")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--factuality-dir", type=Path,
                        help="Optional raw-data audit: directory with overlap/ and gemini_final_res/")
    args = parser.parse_args()
    inputs = json.loads(Path(__file__).with_name("input_manifest.json").read_text())["supp_figures_4_5"]
    for relative, expected in inputs.items():
        if sha256(args.root / relative) != expected:
            raise ValueError(f"Input hash mismatch: {relative}")
    errors = load_counts(args.root / "Data/appendix/error_summary.csv", "Model", "Error Count")
    claims = load_counts(args.root / "Data/appendix/model_claims_summary.csv", "Model Name", "Total Number of Claims")
    selected = {model for _, models in GROUPS.values() for model in models}
    if set(errors) != selected or set(claims) != selected:
        raise ValueError("Both count tables must contain exactly the specified 22 models")
    families = {model: (family, color) for family, (color, models) in GROUPS.items() for model in models}
    rows = []
    for model, count in errors.items():
        if claims[model] <= 0 or count > claims[model]:
            raise ValueError(f"Invalid wrong-claim numerator or denominator: {model}")
        family, color = families[model]
        rows.append({"model": model, "family": family, "color": color,
                     "wrong_claims": count, "total_claims": claims[model],
                     "wrong_claims_percent": 100 * count / claims[model]})
    source_hashes = audit_sources(args.factuality_dir, errors, claims) if args.factuality_dir else None
    out = args.output_dir or args.root / "Outputs/appendix"
    out.mkdir(parents=True, exist_ok=True)
    plot(rows, "wrong_claims", out / "supp_figure_4_wrong_claim_counts")
    plot(rows, "wrong_claims_percent", out / "supp_figure_5_wrong_claim_rates")
    provenance = {"figures": ["Supplementary Figure 4", "Supplementary Figure 5"],
                  "n_models": len(rows), "input_sha256": inputs, "source_audit_sha256": source_hashes,
                  "definition": "S4: number of saved overlapping wrong-claim records. S5: 100 times this count divided by the total extracted claims in the Gemini factuality files.",
                  "script_sha256": sha256(Path(__file__)), "python": sys.version,
                  "matplotlib": matplotlib.__version__, "api_calls": 0}
    provenance["output_sha256"] = {
        f"{stem}.{suffix}": sha256(out / f"{stem}.{suffix}")
        for stem in ("supp_figure_4_wrong_claim_counts", "supp_figure_5_wrong_claim_rates")
        for suffix in ("png", "pdf", "csv")
    }
    (out / "wrong_claim_counts_and_rates_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Generated Supplementary Figures 4–5 for {len(rows)} models: {out}")


if __name__ == "__main__":
    main()
