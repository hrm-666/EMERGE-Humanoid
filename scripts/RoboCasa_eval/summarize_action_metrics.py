#!/usr/bin/env python3
"""Aggregate per-action accuracy diagnostics from RoboCasa episodes.

The RoboCasa driver appends one JSON object per executed action to
``action_metrics.jsonl``. This script answers the baseline questions:

  - How far from the target does a geometric action actually finish?
  - How often does it stop early as ``approached`` / ``not_converged``?
  - Which action types dominate, and what are their reasons?

Stdlib only; scans any directory tree containing action_metrics.jsonl.

  python scripts/RoboCasa_eval/summarize_action_metrics.py \
      tests/robocasa_validation/runs --csv /tmp/actions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


def _fmt(value: float | None, scale: float = 1.0, digits: int = 2) -> str:
    return "-" if value is None else f"{value * scale:.{digits}f}"


def iter_records(roots: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for root in roots:
        for path in sorted(Path(root).rglob("action_metrics.jsonl")):
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    record["_metrics_path"] = str(path)
                    yield record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "completed": 0,
            "failed": 0,
            "errors": 0,
            "position_errors": [],
            "rotation_errors": [],
            "reasons": Counter(),
        }
    )
    approached_residuals: list[float] = []
    reason_counts: Counter[str] = Counter()
    for record in records:
        action_type = str(record.get("action_type", "unknown"))
        bucket = by_type[action_type]
        bucket["count"] += 1
        reason = record.get("reason")
        if reason is not None:
            bucket["reasons"][str(reason)] += 1
            reason_counts[str(reason)] += 1
        if record.get("error") is not None:
            bucket["errors"] += 1
        elif record.get("completed") is True:
            bucket["completed"] += 1
        else:
            bucket["failed"] += 1
        position_error = record.get("position_error_m")
        if isinstance(position_error, (int, float)):
            bucket["position_errors"].append(float(position_error))
            if reason == "approached":
                approached_residuals.append(float(position_error))
        rotation_error = record.get("rotation_error_rad")
        if isinstance(rotation_error, (int, float)):
            bucket["rotation_errors"].append(float(rotation_error))
    return {
        "total": len(records),
        "by_type": by_type,
        "reason_counts": reason_counts,
        "approached_residuals": approached_residuals,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"records: {summary['total']}")
    print()
    header = (
        f"{'action_type':22s} {'n':>5s} {'ok':>5s} {'fail':>5s} {'err':>4s} "
        f"{'pos_med_cm':>10s} {'pos_p90_cm':>10s} {'pos_max_cm':>10s} "
        f"{'rot_p90_rad':>11s}"
    )
    print(header)
    print("-" * len(header))
    for action_type, bucket in sorted(
        summary["by_type"].items(), key=lambda item: (-item[1]["count"], item[0])
    ):
        positions = bucket["position_errors"]
        rotations = bucket["rotation_errors"]
        print(
            f"{action_type:22s} {bucket['count']:5d} {bucket['completed']:5d} "
            f"{bucket['failed']:5d} {bucket['errors']:4d} "
            f"{_fmt(percentile(positions, 50), 100):>10s} "
            f"{_fmt(percentile(positions, 90), 100):>10s} "
            f"{_fmt(percentile(positions, 100), 100):>10s} "
            f"{_fmt(percentile(rotations, 90)):>11s}"
        )
    print()
    print("reasons:")
    for reason, count in summary["reason_counts"].most_common():
        print(f"  {reason:24s} {count}")
    residuals = summary["approached_residuals"]
    if residuals:
        print()
        print(
            "approached residuals (cm): "
            f"n={len(residuals)} "
            f"median={percentile(residuals, 50) * 100:.2f} "
            f"p90={percentile(residuals, 90) * 100:.2f} "
            f"max={max(residuals) * 100:.2f}"
        )


def write_csv(records: list[dict[str, Any]], out: Path) -> None:
    fields = [
        "timestamp",
        "action_type",
        "reason",
        "completed",
        "task_success",
        "total_steps",
        "action_steps",
        "position_error_m",
        "rotation_error_rad",
        "environment_success",
        "error",
        "_metrics_path",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path, help="write one row per action")
    args = parser.parse_args()

    records = list(iter_records(args.results_root))
    if not records:
        print("no action_metrics.jsonl records found")
        return 1
    print_summary(summarize(records))
    if args.csv:
        write_csv(records, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
