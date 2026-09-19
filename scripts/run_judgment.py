"""批量运行多模态 judge，并保存可复现实验记录。"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from transformers import AutoModelForImageTextToText as ModelClass
except ImportError:
    from transformers import AutoModelForMultimodalLM as ModelClass
from transformers import AutoProcessor


PROMPT_VERSION = "pairwise-v1"
REQUIRED_FIELDS = ("instance_id", "image", "question", "candidate_a", "candidate_b", "gold")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"配置文件顶层必须是对象: {path}")
    return value


def load_samples(path: Path) -> list[dict[str, Any]]:
    """支持 JSON 数组、单个 JSON 对象和 JSONL，统一返回样本列表。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到样本文件: {path}")
    if path.suffix.lower() == ".jsonl":
        samples: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL 第 {line_number} 行必须是对象: {path}")
                samples.append(value)
        return samples

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    raise ValueError(f"样本文件必须是对象、对象数组或 JSONL: {path}")


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def validate_sample(sample: dict[str, Any], index: int) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in sample]
    if missing:
        raise ValueError(f"第 {index} 条样本缺少字段: {', '.join(missing)}")
    if str(sample["gold"]).upper() not in {"A", "B"}:
        raise ValueError(f"第 {index} 条样本的 gold 必须为 A 或 B")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt(sample: dict[str, Any]) -> str:
    return f"""You are a multimodal judge.

Question:
{sample['question']}

Candidate A:
{sample['candidate_a']}

Candidate B:
{sample['candidate_b']}

Judge the two candidate answers using the image.
First output exactly A or B.
Then give a short reason."""


def parse_prediction(text: str) -> tuple[str | None, bool, str | None]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        match = re.match(r"^([AB])(?:\s|$|[.):-])", lines[0], re.IGNORECASE)
        if match:
            return match.group(1).upper(), True, "first_line"
    match = re.search(
        r"\b(?:answer|choice|candidate)(?:\s+is)?\s*[:\-]?\s*([AB])\b",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper(), True, "labelled_text"
    return None, False, None


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def model_revision(model: Any, configured: str | None) -> str | None:
    if configured:
        return configured
    config = getattr(model, "config", None)
    return getattr(config, "_commit_hash", None) or getattr(config, "revision", None)


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    existing_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    existing_ids.add(str(json.loads(line)["run_id"]))
                except (KeyError, json.JSONDecodeError):
                    continue
    return existing_ids


def append_ledger(path: Path, record: dict[str, Any], existing_ids: set[str]) -> None:
    if record["run_id"] in existing_ids:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = {
        key: record.get(key)
        for key in (
            "run_id", "benchmark", "instance_id", "model", "model_revision",
            "prompt_version", "prediction", "gold", "correct", "parse_ok",
            "entropy_mean", "entropy_max", "num_generated_tokens", "latency_sec",
        )
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(fields, ensure_ascii=False) + "\n")
    existing_ids.add(record["run_id"])


def load_model(config: dict[str, Any]) -> tuple[Any, Any]:
    model_id = str(config["model_id"])
    model_revision_value = config.get("model_revision")
    dtype_name = str(config.get("dtype", "bfloat16"))
    dtype = getattr(torch, dtype_name) if torch.cuda.is_available() else torch.float32
    print(f"Loading model: {model_id}", flush=True)
    processor = AutoProcessor.from_pretrained(
        model_id, revision=model_revision_value, trust_remote_code=True
    )
    model_kwargs = {
        "revision": model_revision_value,
        "device_map": config.get("device_map", "auto"),
        "trust_remote_code": True,
    }
    try:
        model = ModelClass.from_pretrained(model_id, dtype=dtype, **model_kwargs)
    except TypeError:
        model = ModelClass.from_pretrained(model_id, torch_dtype=dtype, **model_kwargs)
    model.eval()
    return processor, model


def run_sample(
    sample: dict[str, Any],
    config: dict[str, Any],
    project_root: Path,
    processor: Any,
    model: Any,
    output_dir: Path,
    ledger_path: Path,
    existing_ids: set[str],
    sample_index: int,
) -> dict[str, Any]:
    validate_sample(sample, sample_index)
    instance_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample["instance_id"]))
    benchmark = str(sample.get("benchmark", config.get("benchmark", "unknown")))
    run_prefix = str(config.get("run_prefix", "judgment"))
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{run_prefix}_{benchmark}_{instance_id}")

    image_path = resolve_path(str(sample["image"]), project_root)
    if not image_path.exists():
        raise FileNotFoundError(f"第 {sample_index} 条样本找不到图片: {image_path}")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": build_prompt(sample)},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    max_new_tokens = int(config.get("max_new_tokens", 120))
    do_sample = bool(config.get("do_sample", False))
    generation_kwargs = {
        "max_new_tokens": max_new_tokens, "do_sample": do_sample,
        "return_dict_in_generate": True, "output_scores": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = float(config.get("temperature", 0.7))
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)
    latency_sec = time.perf_counter() - start

    input_length = inputs["input_ids"].shape[-1]
    generated_ids = outputs.sequences[0, input_length:]
    prediction_text = processor.decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    prediction, parse_ok, parse_method = parse_prediction(prediction_text)
    score_tensors = [score[0].detach().cpu() for score in outputs.scores]
    if score_tensors:
        score_matrix = torch.stack(score_tensors)
        entropy_trajectory = entropy_from_logits(score_matrix).tolist()
        first_probs = torch.softmax(score_matrix[0].float(), dim=-1)
        token_a = processor.tokenizer.encode("A", add_special_tokens=False)
        token_b = processor.tokenizer.encode("B", add_special_tokens=False)
        p_a = float(first_probs[token_a[-1]]) if token_a else None
        p_b = float(first_probs[token_b[-1]]) if token_b else None
    else:
        score_matrix = torch.empty((0, 0), dtype=torch.float32)
        entropy_trajectory, p_a, p_b = [], None, None

    output_dir.mkdir(parents=True, exist_ok=True)
    save_logits = bool(config.get("save_logits", True))
    logits_path = output_dir / f"{run_id}.logits.npy"
    if save_logits:
        np.save(logits_path, score_matrix.numpy().astype(np.float32))
    gold = str(sample["gold"]).upper()
    record = {
        "run_id": run_id, "benchmark": benchmark, "instance_id": sample["instance_id"],
        "model": str(config["model_id"]),
        "model_revision": model_revision(model, config.get("model_revision")),
        "prompt_version": config.get("prompt_version", PROMPT_VERSION),
        "seed": int(config.get("seed", 0)), "image": str(image_path),
        "question": sample["question"], "candidate_a": sample["candidate_a"],
        "candidate_b": sample["candidate_b"], "prediction": prediction,
        "prediction_text": prediction_text, "parse_ok": parse_ok,
        "parse_method": parse_method, "gold": gold,
        "correct": prediction == gold if parse_ok else False,
        "entropy_trajectory": entropy_trajectory,
        "entropy_mean": float(np.mean(entropy_trajectory)) if entropy_trajectory else None,
        "entropy_max": float(np.max(entropy_trajectory)) if entropy_trajectory else None,
        "first_token_p_a": p_a, "first_token_p_b": p_b,
        "num_input_tokens": int(input_length), "num_generated_tokens": len(entropy_trajectory),
        "latency_sec": latency_sec, "temperature": float(config.get("temperature", 0.7)) if do_sample else None,
        "do_sample": do_sample, "max_new_tokens": max_new_tokens,
        "logits_path": str(logits_path) if save_logits else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result_path = output_dir / f"{run_id}.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    append_ledger(ledger_path, record, existing_ids)
    print(json.dumps({"instance_id": sample["instance_id"], "result": str(result_path)}, ensure_ascii=False))
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量运行多模态 judge")
    parser.add_argument("--config", required=True, type=Path, help="JSON 配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    config_dir = config_path.parent
    project_root = resolve_path(config.get("project_root", config_dir.parent), config_dir)
    samples_path = resolve_path(config["samples_path"], project_root)
    output_dir = resolve_path(config.get("output_dir", "results"), project_root)
    ledger_path = resolve_path(config.get("ledger_path", "ledger/judgments.jsonl"), project_root)
    samples = load_samples(samples_path)
    if not samples:
        raise ValueError(f"样本文件为空: {samples_path}")
    set_seed(int(config.get("seed", 0)))
    processor, model = load_model(config)
    existing_ids = load_existing_ids(ledger_path)
    records = []
    for index, sample in enumerate(samples, 1):
        records.append(run_sample(
            sample, config, project_root, processor, model, output_dir,
            ledger_path, existing_ids, index,
        ))
    correct = sum(bool(record["correct"]) for record in records)
    parsed = sum(bool(record["parse_ok"]) for record in records)
    print(json.dumps({
        "num_samples": len(records), "parse_rate": parsed / len(records),
        "accuracy": correct / len(records), "output_dir": str(output_dir),
        "ledger": str(ledger_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
