#!/usr/bin/env python3
"""Plot response-length distributions by model (Supplementary Figure 6).

Default: reproduce the original plot from saved per-response token counts.
--responses-dir additionally recounts tokens from the exact nine response files
using cl100k_base and requires agreement with every saved count before plotting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recount(directory, manifest, frame):
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    from count_tokens import count_responses
    files = manifest["supp_figure_6_response_files"]
    for filename, expected_hash in files.items():
        if sha256(directory / filename) != expected_hash:
            raise ValueError(f"Historical response hash mismatch: {filename}")
    rows = count_responses(directory, list(files))
    counts = {(row["file"], row["question_id"], row["model"]): row["token_count"] for row in rows}
    expected = {(r.file, r.question_id, r.model): int(r.token_count)
                for r in frame.itertuples(index=False)}
    if set(counts) != set(expected):
        raise ValueError("Recounted response keys do not match the original 7,048 rows")
    mismatches = [key for key in counts if counts[key] != expected[key]]
    if mismatches:
        raise ValueError(f"Token counts changed for {len(mismatches)} responses; examples: {mismatches[:3]}")
    return {"encoding": "cl100k_base", "tiktoken": tiktoken.__version__,
            "matched_response_counts": len(counts),
            "response_sha256": manifest["supp_figure_6_response_files"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Root containing Data/appendix/")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--responses-dir", type=Path, help="Optional exact response files for a token recount")
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("input_manifest.json").read_text())
    inputs = manifest["supp_figure_6"]
    for relative, expected in inputs.items():
        if sha256(args.root / relative) != expected:
            raise ValueError(f"Input hash mismatch: {relative}")
    frame = pd.read_csv(args.root / "Data/appendix/token_counts_detailed.csv")
    if frame[["file", "question_id", "model", "token_count"]].isna().any().any():
        raise ValueError("Missing response identity or token count")
    if frame.duplicated(["model", "question_id"]).any():
        raise ValueError("Duplicate model/question pair")
    tokens = frame["token_count"]
    if not np.isfinite(tokens).all() or (tokens <= 0).any() or (tokens != np.floor(tokens)).any():
        raise ValueError("Token counts must be positive finite integers")
    expected_models = manifest["supp_figure_6_expected_questions"]
    if len(frame) != 7048 or set(frame["model"]) != set(expected_models):
        raise ValueError("Figure 6 requires the original 25-model, 7,048-response cohort")
    for model, missing in expected_models.items():
        expected = {f"Q{i}" for i in range(1, 283)} - set(missing)
        if set(frame.loc[frame["model"] == model, "question_id"]) != expected:
            raise ValueError(f"Question coverage changed for {model}")
    recount_provenance = recount(args.responses_dir, manifest, frame) if args.responses_dir else None

    # Match the original median ordering and seaborn boxplot settings.
    order = frame.groupby("model")["token_count"].median().sort_values(ascending=False).index
    stats = []
    for model in order:
        values = frame.loc[frame["model"] == model, "token_count"].to_numpy()
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        iqr = q3 - q1
        lower = max(q1 - 1.5 * iqr, float(values.min()))
        upper = min(q3 + 1.5 * iqr, float(values.max()))
        whisker_low = min(float(values[values >= lower].min()), q1)
        whisker_high = max(float(values[values <= upper].max()), q3)
        stats.append({"model": model, "n_responses": len(values), "mean_tokens": float(values.mean()),
                      "q1_tokens": q1, "median_tokens": median, "q3_tokens": q3,
                      "lower_whisker_tokens": whisker_low, "upper_whisker_tokens": whisker_high,
                      "outlier_count": int(((values < whisker_low) | (values > whisker_high)).sum()),
                      "total_tokens": int(values.sum())})
    out = args.output_dir or args.root / "Outputs/appendix"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()
    plt.rcParams["pdf.fonttype"] = 42
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=frame, y="model", x="token_count", order=order, ax=ax)
    ax.set_title("Token Count Distribution by Model", fontsize=14, fontweight="bold")
    ax.set_xlabel("Token Count", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    stem = "supp_figure_6_response_token_distributions"
    for extension in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(stats).to_csv(out / f"{stem}.csv", index=False)
    provenance = {"figure": "Supplementary Figure 6", "n_models": 25, "n_responses": len(frame),
                  "encoding": "cl100k_base", "input_sha256": inputs, "token_recount": recount_provenance,
                  "missing_questions_by_model": {m: q for m, q in expected_models.items() if q},
                  "ordering": "Descending median token count; includes original OLMo-2 models",
                  "boxplot": "Quartiles with 1.5-IQR whiskers; individual outliers shown",
                  "script_sha256": sha256(Path(__file__)), "python": sys.version,
                  "versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__,
                               "pandas": pd.__version__, "seaborn": sns.__version__}, "api_calls": 0}
    provenance["output_sha256"] = {f"{stem}.{ext}": sha256(out / f"{stem}.{ext}") for ext in ("png", "pdf", "csv")}
    (out / "response_token_distributions_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Generated Supplementary Figure 6: {len(frame)} responses, 25 models: {out}")


if __name__ == "__main__":
    main()
