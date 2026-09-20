"""Summarize VL-RewardBench judge runs.

The script consumes the JSON records produced by ``run_judgment.py``.  It
reports the basic task metrics plus selective-prediction metrics for several
risk scores.  Lower risk means a sample should be accepted first.

Examples:
    python scripts/vl_summarize.py
    python scripts/vl_summarize.py --results-dir results/vl_rewardbench_pilot \
        --output results/vl_rewardbench_pilot/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_RESULTS_DIR = Path("results/vl_rewardbench_pilot")
DEFAULT_OUTPUT = Path("results/vl_rewardbench_pilot/summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize VL-RewardBench results")
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="Directory containing per-sample JSON result files",
    )
    parser.add_argument(
        "--ledger", type=Path, default=None,
        help="Optional JSONL ledger; per-sample JSON files are preferred",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Path for the JSON summary; use '-' to print only",
    )
    parser.add_argument(
        "--target-risks", type=float, nargs="+", default=[0.05, 0.10, 0.20],
        help="Target accepted risk levels for coverage reporting",
    )
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_json_records(results_dir: Path) -> list[dict[str, Any]]:
    if not results_dir.exists():
        raise FileNotFoundError(f"找不到结果目录: {results_dir}")
    records: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"结果文件必须是 JSON 对象: {path}")
        records.append(value)
    return records


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 ledger: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"ledger 第 {line_number} 行必须是对象")
            records.append(value)
    return records


def deduplicate_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        key = str(record.get("run_id") or record.get("instance_id") or index)
        by_id[key] = record
    return list(by_id.values())


def mean(values: Iterable[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def token_entropy(record: dict[str, Any]) -> float | None:
    """Return length-normalized entropy when a trajectory is available."""
    trajectory = record.get("entropy_trajectory")
    if isinstance(trajectory, list):
        values = [finite_float(item) for item in trajectory]
        return mean(values)
    return finite_float(record.get("entropy_mean"))


def normalized_token_entropy(
    record: dict[str, Any], results_dir: Path | None,
) -> float | None:
    """Normalize mean token entropy by ``log(vocab_size)`` when possible."""
    raw = token_entropy(record)
    if raw is None:
        return None
    logits_path = resolve_logits_path(record, results_dir)
    if logits_path is None:
        return None
    try:
        logits = np.load(logits_path, mmap_mode="r")
        vocab_size = int(logits.shape[-1])
    except (OSError, ValueError, IndexError):
        return None
    denominator = math.log(vocab_size) if vocab_size > 1 else 0.0
    return raw / denominator if denominator else None


def resolve_logits_path(
    record: dict[str, Any], results_dir: Path | None,
) -> Path | None:
    value = record.get("logits_path")
    if not value:
        return None
    original = Path(str(value))
    candidates = [original]
    if results_dir is not None:
        candidates.append(results_dir / original.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def top2_mass(record: dict[str, Any], results_dir: Path | None) -> float | None:
    """Return top-2 probability mass at the first generated token."""
    direct = finite_float(record.get("lambda_2"))
    if direct is not None:
        return direct
    logits_path = resolve_logits_path(record, results_dir)
    if logits_path is None:
        return None
    try:
        logits = np.asarray(np.load(logits_path, mmap_mode="r")[0], dtype=np.float64)
    except (OSError, ValueError, IndexError):
        return None
    if logits.size == 0:
        return None
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    denominator = probabilities.sum()
    if not math.isfinite(float(denominator)) or denominator <= 0:
        return None
    probabilities /= denominator
    if probabilities.size == 1:
        return float(probabilities[0])
    top_indices = np.argpartition(probabilities, -2)[-2:]
    return float(probabilities[top_indices].sum())


def bpe_inputs(record: dict[str, Any]) -> tuple[float, float] | None:
    """Read aligned A-probabilities from AB and BA runs when available."""
    probability_ab = record.get("p_ab", record.get("bpe_p_ab"))
    probability_ba = record.get("p_ba", record.get("bpe_p_ba"))
    if isinstance(record.get("bpe"), dict):
        probability_ab = record["bpe"].get("p_ab", probability_ab)
        probability_ba = record["bpe"].get("p_ba", probability_ba)
    probability_ab = finite_float(probability_ab)
    probability_ba = finite_float(probability_ba)
    if probability_ab is None or probability_ba is None:
        return None
    if not (0.0 <= probability_ab <= 1.0 and 0.0 <= probability_ba <= 1.0):
        return None
    return probability_ab, probability_ba


def binary_entropy(probability: float) -> float:
    terms = []
    if probability > 0:
        terms.append(-probability * math.log(probability))
    if probability < 1:
        terms.append(-(1 - probability) * math.log(1 - probability))
    return sum(terms)


def bpe_entropy(record: dict[str, Any]) -> float | None:
    direct = finite_float(record.get("bpe_entropy"))
    if direct is not None:
        return direct
    values = bpe_inputs(record)
    if values is None:
        return None
    probability_ab, probability_ba = values
    return binary_entropy((probability_ab + probability_ba) / 2)


def order_probability_gap(record: dict[str, Any]) -> float | None:
    direct = finite_float(record.get("D_order", record.get("d_order")))
    if direct is not None:
        return direct
    values = bpe_inputs(record)
    if values is None:
        return None
    return abs(values[0] - values[1])


def sequence_entropy(record: dict[str, Any]) -> float | None:
    """Return unnormalized sequence entropy (sum over generated tokens)."""
    trajectory = record.get("entropy_trajectory")
    if not isinstance(trajectory, list):
        return None
    values = [finite_float(item) for item in trajectory]
    usable = [value for value in values if value is not None]
    return sum(usable) if usable else None


def length_risk(record: dict[str, Any]) -> float | None:
    return finite_float(record.get("num_generated_tokens"))


def first_token_risk(record: dict[str, Any]) -> float | None:
    probability_a = finite_float(record.get("first_token_p_a"))
    probability_b = finite_float(record.get("first_token_p_b"))
    if probability_a is None or probability_b is None:
        return None
    # Low margin means the judge is uncertain between A and B.
    return 1.0 - abs(probability_a - probability_b)


def error_rate(records: list[dict[str, Any]]) -> float | None:
    return safe_rate(sum(not bool(record.get("correct")) for record in records), len(records))


def oracle_aurc(records: list[dict[str, Any]]) -> float:
    """AURC of an oracle ordering with all correct examples first."""
    errors = sorted(1 if not bool(record.get("correct")) else 0 for record in records)
    if not errors:
        return 0.0
    cumulative_errors = 0
    total = 0.0
    for index, error in enumerate(errors, 1):
        cumulative_errors += error
        total += cumulative_errors / index
    return total / len(errors)


def aurc(records: list[dict[str, Any]], score_name: str) -> float | None:
    """Compute AURC after accepting lowest-risk samples first."""
    scored = []
    for record in records:
        score = record.get("_risk_scores", {}).get(score_name)
        correct = record.get("_risk_correct", {}).get(score_name)
        if score is not None and correct is not None:
            scored.append((float(score), 1 if not bool(correct) else 0))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    cumulative_errors = 0
    total = 0.0
    for index, (_, error) in enumerate(scored, 1):
        cumulative_errors += error
        total += cumulative_errors / index
    return total / len(scored)


def random_aurc(records: list[dict[str, Any]], seed: int = 0) -> float:
    """A deterministic random-order baseline for the same error labels."""
    errors = [1 if not bool(record.get("correct")) else 0 for record in records]
    random.Random(seed).shuffle(errors)
    cumulative_errors = 0
    total = 0.0
    for index, error in enumerate(errors, 1):
        cumulative_errors += error
        total += cumulative_errors / index
    return total / len(errors) if errors else 0.0


def risk_coverage(records: list[dict[str, Any]], score_name: str) -> dict[str, Any]:
    scored = []
    for record in records:
        score = record.get("_risk_scores", {}).get(score_name)
        correct = record.get("_risk_correct", {}).get(score_name)
        if score is not None and correct is not None:
            scored.append((float(score), not bool(correct)))
    scored.sort(key=lambda item: item[0])
    points: list[dict[str, float | int]] = []
    errors = 0
    for accepted, (_, wrong) in enumerate(scored, 1):
        errors += int(wrong)
        points.append({
            "coverage": accepted / len(scored),
            "risk": errors / accepted,
            "accepted": accepted,
            "errors": errors,
        })
    return {"n_scored": len(scored), "points": points}


def coverage_at_risk(
    curve: dict[str, Any], target_risk: float,
) -> float | None:
    points = curve.get("points", [])
    eligible = [point["coverage"] for point in points if point["risk"] <= target_risk]
    return max(eligible) if eligible else 0.0 if points else None


def summarize(
    records: list[dict[str, Any]],
    target_risks: list[float],
    results_dir: Path | None = None,
) -> dict[str, Any]:
    records = deduplicate_records(records)
    for record in records:
        record["_risk_scores"] = {
            "entropy_mean": token_entropy(record),
            "entropy_mean_normalized": normalized_token_entropy(record, results_dir),
            "sequence_entropy": sequence_entropy(record),
            "length": length_risk(record),
            "first_token_uncertainty": first_token_risk(record),
            "bpe_entropy": bpe_entropy(record),
            "top2_mass_risk": (
                1.0 - top2_mass(record, results_dir)
                if top2_mass(record, results_dir) is not None else None
            ),
        }
        record["_risk_correct"] = {
            name: (
                record.get("forced_correct")
                if name == "bpe_entropy" else record.get("correct")
            )
            for name in record["_risk_scores"]
        }

    n = len(records)
    parsed = [record for record in records if bool(record.get("parse_ok"))]
    correct = [record for record in records if bool(record.get("correct"))]
    top2_values = [top2_mass(record, results_dir) for record in records]
    bpe_values = [bpe_entropy(record) for record in records]
    order_gaps = [order_probability_gap(record) for record in records]
    risk_scores = [
        "entropy_mean",
        "entropy_mean_normalized",
        "sequence_entropy",
        "length",
        "first_token_uncertainty",
        "bpe_entropy",
        "top2_mass_risk",
    ]
    curves = {name: risk_coverage(records, name) for name in risk_scores}
    oracle = oracle_aurc(records)
    selective: dict[str, Any] = {}
    for name in risk_scores:
        value = aurc(records, name)
        scored_records = [
            {"correct": record["_risk_correct"][name]}
            for record in records
            if record["_risk_scores"][name] is not None
            and record["_risk_correct"][name] is not None
        ]
        score_oracle = oracle_aurc(scored_records)
        selective[name] = {
            "n_scored": curves[name]["n_scored"],
            "aurc": value,
            "oracle_aurc": score_oracle,
            "e_aurc": value - score_oracle if value is not None else None,
            "coverage_at_target_risk": {
                str(target): coverage_at_risk(curves[name], target)
                for target in target_risks
            },
            "risk_coverage": curves[name],
        }

    flip_rate = safe_rate(
        sum(bool(record.get("order_flip")) for record in records if record.get("order_flip") is not None),
        sum(record.get("order_flip") is not None for record in records),
    )
    summary = {
        "benchmark": records[0].get("benchmark") if records else None,
        "model": records[0].get("model") if records else None,
        "prompt_version": records[0].get("prompt_version") if records else None,
        "n": n,
        "n_parsed": len(parsed),
        "n_correct": len(correct),
        "parse_rate": safe_rate(len(parsed), n),
        "accuracy": safe_rate(len(correct), n),
        "mean_latency_sec": mean(finite_float(record.get("latency_sec")) for record in records),
        "mean_forced_latency_sec": mean(
            finite_float(record.get("forced_latency_sec")) for record in records
        ),
        "mean_total_latency_sec": mean(
            finite_float(record.get("total_latency_sec")) for record in records
        ),
        "mean_generated_tokens": mean(
            finite_float(record.get("num_generated_tokens")) for record in records
        ),
        "mean_entropy_correct": mean(token_entropy(record) for record in correct),
        "mean_entropy_wrong": mean(
            token_entropy(record) for record in records if not bool(record.get("correct"))
        ),
        "mean_entropy_normalized": mean(
            normalized_token_entropy(record, results_dir) for record in records
        ),
        "n_bpe": sum(value is not None for value in bpe_values),
        "mean_bpe_entropy": mean(bpe_values),
        "mean_order_probability_gap": mean(order_gaps),
        "n_order_pairs": sum(value is not None for value in order_gaps),
        "flip_rate": flip_rate,
        "order_flip_rate": flip_rate,
        "forced_accuracy": safe_rate(
            sum(bool(record.get("forced_correct")) for record in records if record.get("forced_correct") is not None),
            sum(record.get("forced_correct") is not None for record in records),
        ),
        "free_forced_agreement": safe_rate(
            sum(bool(record.get("free_forced_agree")) for record in records if record.get("free_forced_agree") is not None),
            sum(record.get("free_forced_agree") is not None for record in records),
        ),
        "n_top2_mass": sum(value is not None for value in top2_values),
        "mean_top2_mass": mean(top2_values),
        "oracle_aurc": oracle,
        "random_aurc": random_aurc(records),
        "target_risks": target_risks,
        "selective_metrics": selective,
    }
    for record in records:
        record.pop("_risk_scores", None)
        record.pop("_risk_correct", None)
    return summary


def main() -> None:
    args = parse_args()
    records = load_json_records(args.results_dir)
    if not records and args.ledger is not None:
        records = load_ledger(args.ledger)
    if not records:
        raise ValueError(f"结果目录中没有 per-sample JSON: {args.results_dir}")
    if any(target < 0 or target > 1 for target in args.target_risks):
        raise ValueError("--target-risks 必须位于 [0, 1]")

    summary = summarize(records, args.target_risks, args.results_dir)
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if str(args.output) == "-":
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote summary to {args.output}")
    print(json.dumps({
        "n": summary["n"],
        "parse_rate": summary["parse_rate"],
        "accuracy": summary["accuracy"],
        "forced_accuracy": summary["forced_accuracy"],
        "free_forced_agreement": summary["free_forced_agreement"],
        "order_flip_rate": summary["order_flip_rate"],
        "mean_bpe_entropy": summary["mean_bpe_entropy"],
        "mean_order_probability_gap": summary["mean_order_probability_gap"],
        "oracle_aurc": summary["oracle_aurc"],
        "e_aurc": {
            name: values["e_aurc"]
            for name, values in summary["selective_metrics"].items()
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
