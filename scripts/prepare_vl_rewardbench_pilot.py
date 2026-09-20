"""Create a reproducible, stratified, label-balanced VL-RewardBench pilot."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Image, load_from_disk


DEFAULT_SEED = 20260920


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/VL-RewardBench"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits/vl_rewardbench_pilot_100_seed20260920.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/vl_rewardbench_pilot_100_seed20260920.manifest.json"),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/VL-RewardBench/pilot_100_seed20260920_images"),
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--stratum-field", default="query_source")
    return parser.parse_args()


def as_pair(value: Any, field: str, index: int) -> list[Any]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"row {index}: {field} must contain exactly two values")
    return value


def source_gold(row: dict[str, Any], index: int) -> str:
    rankings = [int(value) for value in as_pair(row["human_ranking"], "human_ranking", index)]
    if rankings[0] == rankings[1]:
        raise ValueError(f"row {index}: tied rankings are not supported")
    return "A" if rankings[0] < rankings[1] else "B"


def allocate_strata(counts: Counter[str], limit: int) -> dict[str, int]:
    """Largest-remainder proportional allocation with capacity constraints."""
    total = sum(counts.values())
    if limit > total:
        raise ValueError(f"requested {limit} samples from only {total} eligible rows")
    exact = {key: limit * count / total for key, count in counts.items()}
    quotas = {key: min(counts[key], int(value)) for key, value in exact.items()}
    remaining = limit - sum(quotas.values())
    order = sorted(counts, key=lambda key: (exact[key] - int(exact[key]), counts[key], key), reverse=True)
    while remaining:
        progressed = False
        for key in order:
            if quotas[key] < counts[key]:
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("unable to allocate all requested samples")
    return {key: value for key, value in quotas.items() if value}


def choose_indices(
    dataset: Any, limit: int, seed: int, stratum_field: str,
) -> tuple[list[int], dict[str, int]]:
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset):
        try:
            source_gold(row, index)
        except ValueError:
            continue
        stratum = str(row.get(stratum_field) or "unknown")
        by_stratum[stratum].append(index)

    counts = Counter({key: len(value) for key, value in by_stratum.items()})
    quotas = allocate_strata(counts, limit)
    rng = random.Random(seed)
    selected: list[int] = []
    for stratum in sorted(quotas):
        candidates = list(by_stratum[stratum])
        rng.shuffle(candidates)
        selected.extend(candidates[:quotas[stratum]])
    rng.shuffle(selected)
    return selected, quotas


def balanced_swap_flags(
    source_golds: list[str], strata: list[str], seed: int,
) -> list[bool]:
    limit = len(source_golds)
    if limit % 2:
        raise ValueError("--limit must be even to guarantee exact A/B balance")
    rng = random.Random(seed ^ 0xA5A5A5A5)
    positions: dict[str, list[int]] = defaultdict(list)
    for position, stratum in enumerate(strata):
        positions[stratum].append(position)

    target_golds = [""] * limit
    base_a = sum(len(items) // 2 for items in positions.values())
    odd_strata = [key for key, items in positions.items() if len(items) % 2]
    rng.shuffle(odd_strata)
    extra_a = limit // 2 - base_a
    a_heavy = set(odd_strata[:extra_a])
    for stratum, items in positions.items():
        rng.shuffle(items)
        count_a = len(items) // 2 + int(stratum in a_heavy)
        labels = ["A"] * count_a + ["B"] * (len(items) - count_a)
        rng.shuffle(labels)
        for position, label in zip(items, labels):
            target_golds[position] = label
    return [source_gold != target_gold for source_gold, target_gold in zip(source_golds, target_golds)]


def write_image(image: Any, path: Path, index: int) -> None:
    image_bytes = image.get("bytes") if isinstance(image, dict) else None
    source_path = image.get("path") if isinstance(image, dict) else None
    if image_bytes:
        path.write_bytes(image_bytes)
        return
    if source_path and Path(source_path).exists():
        path.write_bytes(Path(source_path).read_bytes())
        return
    raise ValueError(f"row {index}: missing image bytes")


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    dataset = load_from_disk(str(args.dataset_dir))["test"]
    dataset = dataset.cast_column("image", Image(decode=False))
    indices, quotas = choose_indices(dataset, args.limit, args.seed, args.stratum_field)
    source_golds = [source_gold(dataset[index], index) for index in indices]
    strata = [str(dataset[index].get(args.stratum_field) or "unknown") for index in indices]
    swap_flags = balanced_swap_flags(source_golds, strata, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    manifest_samples: list[dict[str, Any]] = []
    for pilot_index, (source_index, swap) in enumerate(zip(indices, swap_flags)):
        row = dataset[source_index]
        responses = [str(value) for value in as_pair(row["response"], "response", source_index)]
        original_gold = source_gold(row, source_index)
        if swap:
            responses.reverse()
            gold = "B" if original_gold == "A" else "A"
        else:
            gold = original_gold

        instance_id = str(row["id"])
        safe_id = "".join(char if char.isalnum() or char in "-_." else "_" for char in instance_id)
        image_path = args.image_dir / f"{pilot_index:04d}_{safe_id}.jpg"
        write_image(row["image"], image_path, source_index)
        stratum = str(row.get(args.stratum_field) or "unknown")
        record = {
            "benchmark": "VL-RewardBench",
            "instance_id": instance_id,
            "image": image_path.as_posix(),
            "question": str(row["query"]),
            "candidate_a": responses[0],
            "candidate_b": responses[1],
            "gold": gold,
            "source_index": source_index,
            "source_gold": original_gold,
            "candidate_order_swapped": swap,
            "stratum": stratum,
            "source_models": row.get("models"),
            "source_query_source": row.get("query_source"),
        }
        records.append(record)
        manifest_samples.append({
            "pilot_index": pilot_index,
            "instance_id": instance_id,
            "source_index": source_index,
            "stratum": stratum,
            "source_gold": original_gold,
            "candidate_order_swapped": swap,
            "gold": gold,
        })

    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "benchmark": "VL-RewardBench",
        "dataset_path": str(args.dataset_dir),
        "dataset_size": len(dataset),
        "seed": args.seed,
        "limit": args.limit,
        "stratum_field": args.stratum_field,
        "stratum_quotas": quotas,
        "label_counts": dict(sorted(Counter(record["gold"] for record in records).items())),
        "label_counts_by_stratum": {
            stratum: dict(sorted(Counter(
                record["gold"] for record in records if record["stratum"] == stratum
            ).items()))
            for stratum in sorted(quotas)
        },
        "output": args.output.as_posix(),
        "image_dir": args.image_dir.as_posix(),
        "samples": manifest_samples,
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "samples": len(records),
        "label_counts": manifest["label_counts"],
        "stratum_quotas": quotas,
        "output": str(args.output),
        "manifest": str(args.manifest),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
