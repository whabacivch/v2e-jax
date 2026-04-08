#!/usr/bin/env python3
"""Deterministic numerical checks for outside-to-inside adaptation recovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT / "src", _ROOT):
    _s = str(_d)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import numpy as np

try:
    from .adaptation_common import (
        build_outside_to_inside_scenario,
        make_timestamps,
        make_uniform_frames,
        phase_last_nonzero,
        phase_total,
        run_scan_counts,
        run_stream_trace,
    )
except ImportError:
    from adaptation_common import (
        build_outside_to_inside_scenario,
        make_timestamps,
        make_uniform_frames,
        phase_last_nonzero,
        phase_total,
        run_scan_counts,
        run_stream_trace,
    )
from v2e_jax import DVSParams


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--height", type=int, default=32)
    p.add_argument("--adaptation_rate_hz", type=float, default=4.0)
    p.add_argument("--out", type=Path, default=None, help="optional path to write metrics JSON")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    scenario = build_outside_to_inside_scenario(
        fps=args.fps,
        baseline_frames=8,
        bright_hold_frames=12,
        recovery_frames=10,
        reentry_frames=3,
        settle_frames=4,
        indoor_level=32.0,
        outdoor_level=255.0,
        reentry_level=50.0,
    )
    frames_np = make_uniform_frames(scenario.levels, args.height, args.width)
    timestamps_np = make_timestamps(frames_np.shape[0], scenario.fps)

    common_params = dict(
        pos_thres=0.25,
        neg_thres=0.25,
        sigma_thres=0.0,
        cutoff_hz=80.0,
        refractory_s=0.0,
        shot_noise_std=0.0,
        max_events_per_step=1,
    )
    params_no = DVSParams(adaptation_rate_hz=0.0, **common_params)
    params_yes = DVSParams(adaptation_rate_hz=args.adaptation_rate_hz, **common_params)

    no_adapt = run_stream_trace(frames_np, timestamps_np, params_no, seed=args.seed, track_xy=(0, 0))
    yes_adapt = run_stream_trace(frames_np, timestamps_np, params_yes, seed=args.seed, track_xy=(0, 0))
    no_scan = run_scan_counts(frames_np, timestamps_np, params_no, seed=args.seed)
    yes_scan = run_scan_counts(frames_np, timestamps_np, params_yes, seed=args.seed)

    bright_phase = scenario.phase_slices["outdoor_hold"]
    recovery_phase = scenario.phase_slices["indoor_recovery"]
    gap_tail = slice(max(bright_phase.stop - 3, bright_phase.start), bright_phase.stop)

    metrics = {
        "stream_scan_match_no_adaptation": bool(np.array_equal(no_adapt.on_counts, no_scan[0]) and np.array_equal(no_adapt.off_counts, no_scan[1])),
        "stream_scan_match_with_adaptation": bool(np.array_equal(yes_adapt.on_counts, yes_scan[0]) and np.array_equal(yes_adapt.off_counts, yes_scan[1])),
        "bright_hold_on_total_no_adaptation": phase_total(no_adapt.on_totals, bright_phase),
        "bright_hold_on_total_with_adaptation": phase_total(yes_adapt.on_totals, bright_phase),
        "bright_hold_gap_tail_mean_no_adaptation": float(np.mean(no_adapt.ref_gap_trace[gap_tail])),
        "bright_hold_gap_tail_mean_with_adaptation": float(np.mean(yes_adapt.ref_gap_trace[gap_tail])),
        "indoor_recovery_last_off_frame_no_adaptation": phase_last_nonzero(no_adapt.off_totals, recovery_phase),
        "indoor_recovery_last_off_frame_with_adaptation": phase_last_nonzero(yes_adapt.off_totals, recovery_phase),
    }

    checks = {
        "stream_scan_match_no_adaptation": metrics["stream_scan_match_no_adaptation"],
        "stream_scan_match_with_adaptation": metrics["stream_scan_match_with_adaptation"],
        "adaptation_reduces_bright_hold_on_total": (
            metrics["bright_hold_on_total_with_adaptation"] < metrics["bright_hold_on_total_no_adaptation"]
        ),
        "adaptation_reduces_reference_gap_at_bright_tail": (
            metrics["bright_hold_gap_tail_mean_with_adaptation"] < metrics["bright_hold_gap_tail_mean_no_adaptation"]
        ),
        "adaptation_quiets_dark_recovery_earlier": (
            metrics["indoor_recovery_last_off_frame_with_adaptation"]
            < metrics["indoor_recovery_last_off_frame_no_adaptation"]
        ),
    }

    report = {"metrics": metrics, "checks": checks}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    if all(checks.values()):
        return 0
    failed = [name for name, ok in checks.items() if not ok]
    print(f"Failed checks: {', '.join(failed)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
