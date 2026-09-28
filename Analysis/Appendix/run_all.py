#!/usr/bin/env python3
"""Reproduce the seven selected appendix results offline and verify their values."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from _common import CODE, ROOT, sha256, validate_inputs
from verify_results import verify

ANALYSIS = CODE.parent

TASKS = {
    "figure1": ("supp_figure_1_grading_consistency.py", "supp_figure_1", ["supp_figure_1_grading_consistency.csv"]),
    "figure2": ("supp_figure_2_model_size_comparison.py", "supp_figure_2", ["supp_figure_2_model_size_comparison.csv"]),
    "table-s2": ("table_s2_judge_agreement.py", "table_s2", ["table_s2.csv"]),
    "table-s5": ("table_s5_direct_judgment_agreement.py", "table_s5", ["table_s5.csv"]),
    "figure4": ("supp_figures_4_5_wrong_claims.py", "supp_figures_4_5", ["supp_figure_4_wrong_claim_counts.csv"]),
    "figure5": ("supp_figures_4_5_wrong_claims.py", "supp_figures_4_5", ["supp_figure_5_wrong_claim_rates.csv"]),
    "figure6": ("supp_figure_6_token_distribution.py", "supp_figure_6", ["supp_figure_6_response_token_distributions.csv"]),
}


def export_bundle(root, destination, inputs):
    """Export only the required code and hash-verified data; never scan/copy a whole source tree."""
    if destination.exists():
        raise ValueError(f"Export destination already exists: {destination}. Choose a new directory.")
    destination.mkdir(parents=True)
    code_out = destination / "Analysis/Appendix"
    code_out.mkdir(parents=True)
    names = {task[0] for task in TASKS.values()} | {
        "input_manifest.json",
        "count_tokens.py", "expected_results.json", "run_all.py", "verify_results.py", "_common.py", ".gitignore",
    }
    for name in sorted(names):
        shutil.copy2(CODE / name, code_out / name)
    shutil.copy2(ANALYSIS / "requirements.txt", destination / "Analysis/requirements.txt")
    shutil.copy2(ANALYSIS.parent / "README.md", destination / "README.md")
    for relative, expected in inputs.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
        if sha256(target) != expected:
            raise ValueError(f"Export copy hash mismatch: {relative}")
    print(f"Exported {len(inputs)} verified input files and the analysis code to {destination}")
    print("Guide: README.md; install: python -m pip install -r Analysis/requirements.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Root containing the manifest-listed input tree")
    parser.add_argument("--output-dir", type=Path, help="Default: ROOT/Outputs/appendix")
    parser.add_argument("--only", nargs="+", choices=TASKS, help="Run a subset; figure4/5 share one plotting script")
    parser.add_argument("--check-inputs", action="store_true", help="Validate required files and hashes without plotting")
    parser.add_argument("--export-bundle", type=Path, help="Copy all required code and inputs into a new portable folder; do not run")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.export_bundle and args.only:
        parser.error("--export-bundle includes all seven results; omit --only")
    selected = args.only or list(TASKS)
    inputs = {}
    for group in dict.fromkeys(TASKS[name][1] for name in selected):
        inputs.update(validate_inputs(root, group))
    print(f"Verified {len(inputs)} required input files.", flush=True)
    if args.export_bundle:
        export_bundle(root, args.export_bundle.resolve(), inputs)
        return
    if args.check_inputs:
        return
    out = (args.output_dir or root / "Outputs/appendix").resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    steps = []
    for script in dict.fromkeys(TASKS[name][0] for name in selected):
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, "-B", str(CODE / script), "--root", str(root),
                        "--output-dir", str(out)], check=True, env=environment)
        steps.append({"script": script, "sha256": sha256(CODE / script)})
    filenames = [filename for name in selected for filename in TASKS[name][2]]
    checks = verify(out, filenames)
    report = {"selected_results": selected, "numeric_checks": checks, "steps": steps,
              "input_sha256": inputs, "python": sys.version, "api_calls": 0,
              "elapsed_seconds": round(time.monotonic() - started, 3),
              "expected_results_sha256": sha256(CODE / "expected_results.json"),
              "manifest_sha256": sha256(CODE / "input_manifest.json"),
              "requirements_sha256": sha256(ANALYSIS / "requirements.txt"),
              "code_sha256": {p.name: sha256(p) for p in sorted(CODE.glob("*.py"))}}
    report["installed_versions"] = {}
    for line in (ANALYSIS / "requirements.txt").read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            package = line.split("==")[0]
            try:
                report["installed_versions"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report["installed_versions"][package] = "not installed (not required for selected results)"
    (out / "reproduction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(checks)} results match frozen reference values. Outputs: {out}")


if __name__ == "__main__":
    main()
