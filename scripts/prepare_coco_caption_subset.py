#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions_json", required=True, type=Path)
    parser.add_argument("--image_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max_samples", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    data = json.loads(args.captions_json.read_text(encoding="utf-8"))

    captions_by_id = {}
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        captions_by_id.setdefault(image_id, []).append(ann["caption"])

    records = []
    for image_path in sorted(args.image_dir.glob("*.jpg")):
        image_id = int(image_path.stem)
        captions = captions_by_id.get(image_id)
        if not captions:
            continue
        records.append(
            {
                "id": image_path.stem,
                "image_path": str(image_path.resolve()),
                "prompt": captions[0],
            }
        )
        if len(records) >= args.max_samples:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
