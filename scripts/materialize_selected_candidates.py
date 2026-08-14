#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path


def parse_candidate_roots(value):
    roots = {}
    for item in value.split(","):
        schedule, separator, path = item.partition("=")
        if not separator or not schedule.strip() or not path.strip():
            raise argparse.ArgumentTypeError(
                "candidate roots must use schedule=/path entries separated by commas"
            )
        roots[schedule.strip()] = Path(path.strip())
    return roots


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path in selection record: {value}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Materialize selector outputs without re-encoding candidate images."
    )
    parser.add_argument("--selection-records", type=Path, required=True)
    parser.add_argument("--candidate-image-roots", type=parse_candidate_roots, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    records = json.loads(args.selection_records.read_text(encoding="utf-8"))
    output_root = args.output_root.resolve()
    source_roots = {
        schedule: root.resolve() for schedule, root in args.candidate_image_roots.items()
    }
    if output_root in source_roots.values():
        raise ValueError("output root must differ from every candidate image root")

    counts = Counter()
    copied = 0
    skipped = 0
    for record in records:
        schedule = record["selected_schedule_mode"]
        if schedule not in source_roots:
            raise KeyError(f"missing candidate root for selected schedule: {schedule}")
        relative_path = safe_relative_path(record["file"])
        source = source_roots[schedule] / relative_path
        destination = output_root / relative_path
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.is_file() and not args.overwrite:
            if file_digest(source) != file_digest(destination):
                raise FileExistsError(
                    f"destination differs from selected candidate: {destination}"
                )
            skipped += 1
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
            if file_digest(source) != file_digest(destination):
                raise OSError(f"copy verification failed: {destination}")
            copied += 1
        counts[schedule] += 1

    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.selection_records, output_root / "selection_records.json")
    summary = {
        "selection_records": str(args.selection_records.resolve()),
        "candidate_image_roots": {
            schedule: str(root) for schedule, root in source_roots.items()
        },
        "output_root": str(output_root),
        "records": len(records),
        "copied": copied,
        "skipped_identical": skipped,
        "selected_schedule_counts": dict(sorted(counts.items())),
        "copy_mode": "byte_preserving",
        "hash": "sha256",
    }
    (output_root / "materialization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
