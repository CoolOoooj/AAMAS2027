"""Run the configured AB/BA forced-choice experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/vl_rewardbench_pilot_100_order_swap.json"),
    )
    args = parser.parse_args()
    runner = Path(__file__).with_name("run_judgment.py")
    summarizer = Path(__file__).with_name("vl_summarize.py")
    status = subprocess.call([sys.executable, str(runner), "--config", str(args.config)])
    if status:
        raise SystemExit(status)

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root_value = Path(config.get("project_root", config_path.parent.parent))
    project_root = (
        project_root_value if project_root_value.is_absolute()
        else (config_path.parent / project_root_value).resolve()
    )
    results_dir_value = Path(config["output_dir"])
    results_dir = (
        results_dir_value if results_dir_value.is_absolute()
        else (project_root / results_dir_value).resolve()
    )
    summary_value = Path(config.get("summary_path", results_dir / "summary.json"))
    summary_path = (
        summary_value if summary_value.is_absolute()
        else (project_root / summary_value).resolve()
    )
    raise SystemExit(subprocess.call([
        sys.executable, str(summarizer),
        "--results-dir", str(results_dir),
        "--output", str(summary_path),
    ]))


if __name__ == "__main__":
    main()
