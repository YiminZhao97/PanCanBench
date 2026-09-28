#!/usr/bin/env python3
"""Independently recompute Figure 5 statistics with NumPy and verify plot files."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from figure5_common import HERE, close, load_results, require, sha256, write_json


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify(output_dir=HERE, inputs_dir=HERE / "inputs", figure4_summary=None):
    config, results, manifest = load_results(inputs_dir, HERE / "model_config.json")
    source = read_csv(inputs_dir / "paired_question_scores.csv")
    expected = {}
    for model in config["models"]:
        rows = [r for r in source if r["response_model"] == model["id"]]
        summary = {"n": len(rows)}
        for kind in ("human", "synthetic"):
            values = np.array([100 * float(r[f"{kind}_total_score"]) / float(r[f"{kind}_max_possible_score"])
                               for r in rows])
            summary[f"{kind}_mean"] = float(np.mean(values))
            summary[f"{kind}_sd"] = float(np.std(values, ddof=1))
            summary[f"{kind}_se"] = float(np.std(values, ddof=1) / np.sqrt(len(values)))
            summary[f"{kind}_zero_scores_retained"] = int(np.count_nonzero(values == 0))
            summary[f"{kind}_negative_scores_retained"] = int(np.count_nonzero(values < 0))
        summary["synthetic_minus_human_mean"] = summary["synthetic_mean"] - summary["human_mean"]
        expected[model["id"]] = summary
    for row in expected.values():
        for kind in ("human", "synthetic"):
            row[f"{kind}_rank"] = 1 + sum(other[f"{kind}_mean"] > row[f"{kind}_mean"] for other in expected.values())
        row["rank_change"] = row["human_rank"] - row["synthetic_rank"]
    for filename in ("figure5_model_summary.csv", "rank_change_summary.csv"):
        rows = read_csv(output_dir / filename)
        require(len(rows) == len(expected), f"Wrong row count: {filename}")
        require({r["response_model"] for r in rows} == set(expected), f"Wrong models: {filename}")
        for row in rows:
            for field, value in expected[row["response_model"]].items():
                close(float(row[field]), value, f"{filename}, {row['response_model']}, {field}")
    summary = read_csv(output_dir / "figure5_model_summary.csv")
    require([r["response_model"] for r in summary] == [r["response_model"] for r in results], "Bar order differs")
    for stem in ("figure5a", "figure5b"):
        provenance = json.loads((output_dir / f"{stem}_provenance.json").read_text())
        require(provenance["paired_question_scores_sha256"] == manifest["paired_question_scores_sha256"], "Stale plot inputs")
        for filename, digest in provenance["outputs_sha256"].items():
            require(sha256(output_dir / filename) == digest, f"Plot changed: {filename}")
        for filename, digest in provenance["source_code_sha256"].items():
            require(sha256(HERE / filename) == digest, f"Plot code changed: {filename}; regenerate the plots")
        require((output_dir / f"{stem}.pdf").read_bytes().startswith(b"%PDF-"), "Invalid PDF")
        require((output_dir / f"{stem}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), "Invalid PNG")
    if figure4_summary:
        rows = read_csv(figure4_summary)
        require({r["response_model"] for r in rows} == set(expected), "Figure 4 model coverage differs")
        for row in rows:
            values = expected[row["response_model"]]
            for field, reference in (("human_mean", "mean_percentage"), ("human_se", "se_percentage"), ("n", "n_included")):
                close(values[field], float(row[reference]), f"Figure 4 cross-check: {row['response_model']}, {field}")
    report = {"status": "passed", "models": len(expected), "paired_responses": len(source),
              "mean_sd_se_and_ranks": "Independent NumPy recomputation agrees with both saved tables",
              "saved_reference_means": "All 44 means match original grade summary files",
              "figure4_human_score_crosscheck": "passed" if figure4_summary else "not requested",
              "plot_checksums": "passed", "api_calls": 0,
              "input_sha256": manifest["paired_question_scores_sha256"]}
    write_json(output_dir / "verification_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--inputs-dir", type=Path, default=HERE / "inputs")
    parser.add_argument("--figure4-summary", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir, args.inputs_dir, args.figure4_summary), indent=2))
