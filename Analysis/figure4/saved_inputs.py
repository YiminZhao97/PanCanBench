"""Locate immutable saved inputs after moving the repository to another machine."""
import hashlib
import json
from pathlib import Path


def resolve_saved_file(recorded_path, expected_sha256, root):
    root = Path(root).resolve()
    manifest = root / 'Data/input_manifest.json'
    if manifest.is_file():
        files = json.loads(manifest.read_text())['files']
        for relative, digest in files.items():
            if digest == expected_sha256:
                candidate = root / relative
                if candidate.is_file() and hashlib.sha256(candidate.read_bytes()).hexdigest() == expected_sha256:
                    return candidate
    candidate = Path(recorded_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    # Newly generated run files may not yet be in the frozen data manifest.
    if candidate.resolve().is_relative_to(root) and candidate.is_file():
        if hashlib.sha256(candidate.read_bytes()).hexdigest() == expected_sha256:
            return candidate
    raise ValueError(f'Missing or changed saved input: {Path(recorded_path).name}; run Data/prepare_data.py --check')
