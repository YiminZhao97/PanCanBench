#!/usr/bin/env python3
"""Compare reproduced numeric results with frozen historical reference values."""

import argparse
import csv
import json
import math
from pathlib import Path

CODE = Path(__file__).resolve().parent


def equivalent(actual, expected):
    try:
        a, b = float(actual), float(expected)
    except (TypeError, ValueError):
        return actual == expected
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)


def verify(output, filenames=None):
    references = json.loads((CODE / "expected_results.json").read_text())
    filenames = filenames or list(references)
    checked = {}
    for name in filenames:
        expected = references[name]
        with (output / name).open(newline="", encoding="utf-8") as handle:
            actual = list(csv.DictReader(handle))
        if len(actual) != len(expected):
            raise ValueError(f"{name}: expected {len(expected)} rows, found {len(actual)}")
        for index, (got, wanted) in enumerate(zip(actual, expected), start=2):
            for field, value in wanted.items():
                if field not in got or not equivalent(got[field], value):
                    raise ValueError(f"{name}:{index}: {field}: expected {value!r}, found {got.get(field)!r}")
        if name.startswith("supp_figure_"):
            for suffix in (".png", ".pdf"):
                image = (output / name).with_suffix(suffix)
                if not image.is_file() or image.stat().st_size == 0:
                    raise ValueError(f"Missing or empty plot: {image.name}")
        checked[name] = {"rows": len(actual), "matches_reference": True}
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=CODE.parents[1] / "Outputs/appendix")
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
