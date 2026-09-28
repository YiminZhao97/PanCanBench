#!/usr/bin/env python3
"""Check, export, or install the exact saved inputs for this code release."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check(root, files):
    failures = [name for name, digest in files.items()
                if not (root/name).is_file() or sha256(root/name) != digest]
    if failures:
        raise ValueError('Missing or changed data:\n'+'\n'.join(failures))
    return len(files)


def install(archive, root, files):
    # Validate the whole archive before changing any input file.
    with tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary)
        with tarfile.open(archive, 'r:*') as bundle:
            seen = set()
            for member in bundle:
                if not member.isfile() or member.name not in files or member.name in seen:
                    raise ValueError(f'Unexpected archive member: {member.name}')
                seen.add(member.name)
                target = stage/member.name
                if not target.resolve().is_relative_to(stage.resolve()):
                    raise ValueError('Unsafe archive path')
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
        if seen != set(files):
            raise ValueError('Archive does not contain the complete input manifest')
        check(stage, files)
        for name, digest in files.items():
            destination = root/name
            if destination.exists() and sha256(destination) != digest:
                raise ValueError(f'Existing input differs: {name}; move it aside before installing')
        for name in files:
            destination = root/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stage/name,destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check', action='store_true')
    group.add_argument('--archive', type=Path, help='Install a downloaded data bundle')
    group.add_argument('--download', metavar='URL', help='Download and install a data bundle, verifying every input hash')
    group.add_argument('--export', type=Path, help='Create a gzip tar bundle from the verified local inputs')
    args = parser.parse_args()
    files = json.loads((ROOT/'Data/input_manifest.json').read_text())['files']
    if args.archive:
        install(args.archive, ROOT, files)
    if args.download:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'data.tar.gz'
            with urllib.request.urlopen(args.download, timeout=60) as response, path.open('wb') as output:
                shutil.copyfileobj(response,output)
            install(path,ROOT,files)
    total = check(ROOT,files)
    if args.export:
        if args.export.exists():
            parser.error('Export already exists; select a new path')
        args.export.parent.mkdir(parents=True,exist_ok=True)
        with tarfile.open(args.export,'w:gz') as bundle:
            for name in sorted(files):
                info = bundle.gettarinfo(str(ROOT/name),arcname=name)
                info.uid=info.gid=0
                info.uname=info.gname=''
                info.mtime=0
                with (ROOT/name).open('rb') as stream:
                    bundle.addfile(info,stream)
        print(f'Exported {args.export}; SHA-256 {sha256(args.export)}')
    print(f'PASS: {total} saved input files match the release manifest')


if __name__ == '__main__':
    main()
