#!/usr/bin/env python3
"""Install exact fixed reference inputs from the original authorized archives."""

import argparse
from pathlib import Path, PurePosixPath
import hashlib
import json
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent


def restore(archive_dir, workflow, spec):
    archive = archive_dir / spec["source_archive"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != spec["archive_sha256"]:
        raise ValueError(f"Archive SHA-256 mismatch: {archive.name}")

    with ZipFile(archive) as package:
        archive_folder = PurePosixPath(package.namelist()[0]).parts[0]
        for name in spec["reference_inputs"]:
            relative = PurePosixPath(name)
            if relative.parts[0] != "reference" or ".." in relative.parts:
                raise ValueError(f"Unexpected reference path: {name}")
            destination = HERE / workflow / relative
            content = package.read(str(PurePosixPath(archive_folder) / relative))
            if destination.exists():
                if destination.read_bytes() != content:
                    raise FileExistsError(f"Reference input differs from the checked archive: {destination}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)


def main():
    manifest = json.loads((HERE / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_dir", type=Path, help="Directory containing the original source ZIP files")
    parser.add_argument("--workflow", choices=manifest, help="Restore a single extract or training workflow")
    args = parser.parse_args()

    for workflow, spec in manifest.items():
        if args.workflow is None or args.workflow == workflow:
            restore(args.archive_dir, workflow, spec)
            print(f"Reference inputs ready for {workflow}")


if __name__ == "__main__":
    main()
