#!/usr/bin/env python3
"""Analyze a placement force log to find the load-bearing joints.

Reads the JSONL produced by the orchestrator's ``record_force`` step flag and,
per joint, reports the torque drop over the placement window. The load-bearing
joints are the ones whose torque falls the most when the box weight transfers
to the surface (arm goes from "self + box" to "self only").

A single log ranks joints by raw drop, but that drop is confounded by the arm's
own pose change while lowering. Pass ``--compare <empty_run.jsonl>`` to subtract
the empty (no-box) run's drop and isolate the box-induced unloading.
"""

from __future__ import annotations

import argparse
import json
import sys


def load(path: str) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _drop(samples: list[dict], head_frac: float = 0.2) -> dict[str, float]:
    """Per-joint (mean head - mean tail): positive = torque fell (unloaded)."""
    joints = sorted({j for s in samples for j in s["forces"]})
    n = len(samples)
    head = max(1, int(n * head_frac))
    drops: dict[str, float] = {}
    for j in joints:
        vals = [s["forces"].get(j, float("nan")) for s in samples]
        loaded = [v for v in vals[:head] if v == v]
        unloaded = [v for v in vals[-head:] if v == v]
        if loaded and unloaded:
            drops[j] = sum(loaded) / len(loaded) - sum(unloaded) / len(unloaded)
    return drops


def _rank(drops: dict[str, float], top: int) -> list[tuple[str, float]]:
    return sorted(drops.items(), key=lambda kv: -abs(kv[1]))[:top]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", help="force JSONL from record_force (with-box run)")
    p.add_argument("--compare", help="empty/no-box run JSONL, for box-induced contrast")
    p.add_argument("--plot", help="optional PNG path (requires matplotlib)")
    p.add_argument("--top", type=int, default=10, help="show top N joints (default 10)")
    args = p.parse_args(argv)

    samples = load(args.log)
    if not samples:
        sys.exit("no samples in log")

    drops = _drop(samples)

    print(f"{args.log}: {len(samples)} samples over "
          f"{samples[-1]['t'] - samples[0]['t']:.2f}s")
    print(f"\n{'joint':<28} {'drop (Nm)':>12}")
    print("-" * 42)
    for j, d in _rank(drops, args.top):
        print(f"{j:<28} {d:>+12.2f}")

    if args.compare:
        empty = load(args.compare)
        if not empty:
            sys.exit("no samples in compare log")
        empty_drops = _drop(empty)
        print(f"\n== box-induced drop (with-box minus empty) ==")
        print(f"{'joint':<28} {'box_drop (Nm)':>14}")
        print("-" * 44)
        box_drops = {
            j: drops[j] - empty_drops.get(j, 0.0)
            for j in drops
            if j in empty_drops
        }
        for j, d in _rank(box_drops, args.top):
            print(f"{j:<28} {d:>+14.2f}")
        print("\nLoad-bearing joints (largest box-induced drop):")
        print("  " + ", ".join(j for j, _ in _rank(box_drops, args.top)))
    else:
        print("\nLoad-bearing joints (largest raw drop; consider --compare "
              "to remove pose-change noise):")
        print("  " + ", ".join(j for j, d in _rank(drops, args.top) if d > 0))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        joints = sorted({j for s in samples for j in s["forces"]})
        fig, ax = plt.subplots(figsize=(11, 6))
        for j in joints:
            ax.plot([s["t"] for s in samples],
                    [s["forces"].get(j, float("nan")) for s in samples],
                    label=j, alpha=0.7)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("torque (Nm)")
        ax.set_title(f"Placement force profile — {args.log}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"\nPlot saved: {args.plot}")


if __name__ == "__main__":
    main()
