"""Prepare a small VL-RewardBench JSONL pilot for run_judgment.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Image, load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/VL-RewardBench"))
    parser.add_argument("--output", type=Path, default=Path("data/VL-RewardBench/pilot_50.jsonl"))
    parser.add_argument(
        "--image-dir", type=Path, default=Path("data/VL-RewardBench/images")
    )
    parser.add_argument("--limit", type=int, default=50)
    return parser.parse_args()


def as_list(value: Any) -> list[Any]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"Expected a list of two values, got: {value!r}")
    return value


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    dataset = load_from_disk(str(args.dataset_dir))["test"]
    dataset = dataset.cast_column("image", Image(decode=False))
    if len(dataset) < args.limit:
        raise ValueError(f"Dataset has {len(dataset)} rows, fewer than {args.limit}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for index, row in enumerate(dataset.select(range(args.limit))):
        responses = as_list(row["response"])
        rankings = [int(value) for value in as_list(row["human_ranking"])]
        if rankings[0] == rankings[1]:
            raise ValueError(f"Tied ranking at row {index}: {rankings}")

        image = row["image"]
        image_bytes = image.get("bytes") if isinstance(image, dict) else None
        if not image_bytes:
            raise ValueError(f"Missing image bytes at row {index}")

        instance_id = str(row["id"])
        safe_id = "".join(char if char.isalnum() or char in "-_." else "_" for char in instance_id)
        image_name = f"{index:04d}_{safe_id}.jpg"
        image_path = args.image_dir / image_name
        image_path.write_bytes(image_bytes)

        # VL-RewardBench uses rank 0 for the preferred response.
        gold = "A" if rankings[0] < rankings[1] else "B"
        records.append(
            {
                "benchmark": "VL-RewardBench",
                "instance_id": instance_id,
                "image": str(image_path).replace("\\", "/"),
                "question": str(row["query"]),
                "candidate_a": str(responses[0]),
                "candidate_b": str(responses[1]),
                "gold": gold,
                "source_index": index,
                "source_models": row.get("models"),
                "source_query_source": row.get("query_source"),
            }
        )

    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} samples to {args.output}")
    print(f"wrote {len(records)} images to {args.image_dir}")


if __name__ == "__main__":
    main()
