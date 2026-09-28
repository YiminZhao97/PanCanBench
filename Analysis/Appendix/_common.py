"""Small offline helpers shared by the appendix reproduction commands."""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", type=Path, default=ROOT, help="Root of the saved input tree")
    parser.add_argument("--output-dir", type=Path, help="Default: ROOT/Outputs/appendix")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.output_dir = args.output_dir or args.root / "Outputs/appendix"
    return args


def validate_inputs(root, group):
    inputs = read_json(CODE / "input_manifest.json")[group]
    for relative, expected in inputs.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Missing required input: {relative}. See README.md for the data layout.")
        if sha256(path) != expected:
            raise ValueError(f"Input hash mismatch: {relative}")
    return inputs


def score(record):
    """Historical Phase 2 scoring; do not substitute the final human rubrics."""
    if "criterion_scores" in record:
        criteria = record["criterion_scores"]
        total = sum(c["score_given"] for c in criteria)
        maximum = sum(c["max_points"] for c in criteria)
        if not math.isfinite(total) or not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("Invalid historical score or denominator")
        value = total / maximum * 100
    else:
        value = float(record["percentage"])
    if not math.isfinite(value):
        raise ValueError("Nonfinite score")
    return value


def keyed_records(records):
    result = {}
    for row in records:
        key = (int(row["question_number"]), row["source"])
        if key in result:
            raise ValueError(f"Duplicate question/model: {key}")
        result[key] = row
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def provenance(out, stem, inputs, script, details, extensions=("png", "pdf", "csv")):
    import matplotlib
    payload = {**details, "input_sha256": inputs, "script_sha256": sha256(script),
               "shared_code_sha256": sha256(__file__), "python": sys.version,
               "matplotlib": matplotlib.__version__, "font": "DejaVu Sans",
               "api_calls": 0,
               "output_sha256": {f"{stem}.{ext}": sha256(out / f"{stem}.{ext}") for ext in extensions}}
    questions = out / f"{stem}_questions.csv"
    if questions.is_file():
        payload["output_sha256"][questions.name] = sha256(questions)
    (out / f"{stem}_provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
