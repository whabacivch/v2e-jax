#!/usr/bin/env python3
"""Visualize and benchmark adaptation on a moving-box clip with a whiteout interval."""

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
        benchmark_scan,
        benchmark_streaming,
        build_moving_box_whiteout_scenario,
        phase_last_nonzero,
        phase_total,
        run_scan_counts,
        run_stream_trace,
    )
except ImportError:
    from adaptation_common import (
        benchmark_scan,
        benchmark_streaming,
        build_moving_box_whiteout_scenario,
        phase_last_nonzero,
        phase_total,
        run_scan_counts,
        run_stream_trace,
    )
from helpers.render import save_preview_grid, write_summary, write_triptych_mp4
from v2e_jax import DVSParams


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pre_white_frames", type=int, default=135)
    p.add_argument("--whiteout_frames", type=int, default=15)
    p.add_argument("--post_white_frames", type=int, default=150)
    p.add_argument("--background_level", type=float, default=40.0)
    p.add_argument("--box_level", type=float, default=115.0)
    p.add_argument("--white_level", type=float, default=255.0)
    p.add_argument("--box_width", type=int, default=48)
    p.add_argument("--box_height", type=int, default=48)
    p.add_argument("--reacquire_lag_frames", type=int, default=10)
    p.add_argument("--sweep_px_per_frame", type=float, default=2.4)
    p.add_argument("--stripe_period_px", type=int, default=12)
    p.add_argument("--stripe_contrast", type=float, default=0.25)
    p.add_argument("--background_texture_amplitude", type=float, default=0.15)
    p.add_argument("--adaptation_rate_hz", type=float, default=4.0)
    p.add_argument("--pos_thres", type=float, default=0.22)
    p.add_argument("--neg_thres", type=float, default=0.22)
    p.add_argument("--cutoff_hz", type=float, default=80.0)
    p.add_argument("--max_events_per_step", type=int, default=1)
    p.add_argument("--benchmark_repeats", type=int, default=3)
    p.add_argument(
        "--display_scale",
        type=int,
        default=2,
        help="nearest-neighbor upscaling applied when encoding the triptych video",
    )
    p.add_argument(
        "--decay_frames",
        type=int,
        default=4,
        help="render-only event persistence buffer depth (triptych only; physics untouched)",
    )
    p.add_argument(
        "--reacquire_ratio",
        type=float,
        default=2.0,
        help="signal-to-clutter ratio that defines ROI reacquisition after whiteout",
    )
    p.add_argument(
        "--reacquire_consecutive",
        type=int,
        default=2,
        help="consecutive frames the ratio must be sustained to count as reacquired",
    )
    p.add_argument(
        "--reacquire_edge_band_px",
        type=int,
        default=4,
        help="half-width (px) of the box-edge signal band inside the ROI",
    )
    p.add_argument(
        "--triptych",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write the 3-panel adaptation_triptych.mp4 (original | adaptation ON | adaptation OFF)",
    )
    return p.parse_args(argv)


def _reacquire_frame(
    on_counts: np.ndarray,
    off_counts: np.ndarray,
    scenario,
    *,
    ratio_threshold: float = 2.0,
    consecutive_k: int = 2,
    edge_band_px: int = 4,
) -> tuple[int, int] | tuple[None, None]:
    """Find the first post-whiteout frame where ROI signal dominates clutter.

    Returns ``(absolute_frame_index, frames_since_whiteout_end)`` or ``(None, None)``
    if reacquisition never happens within the post-whiteout phase.

    "Signal" is the number of active (ON or OFF) pixels inside a tight
    ``edge_band_px`` column-band centered on the expected box-edge position
    (from ``scenario.edge_column_lookup``). "Clutter" is the remaining active
    pixel count elsewhere inside the ROI.
    """
    post_slice = scenario.phase_slices["post_white_motion"]
    y0, y1, x0, x1 = scenario.right_roi
    roi_width = x1 - x0
    streak = 0
    first_hit: int | None = None
    for t in range(post_slice.start, post_slice.stop):
        edge_x = int(scenario.edge_column_lookup[t])
        if edge_x < 0:
            streak = 0
            continue
        on_roi = on_counts[t, y0:y1, x0:x1] > 0
        off_roi = off_counts[t, y0:y1, x0:x1] > 0
        active = on_roi | off_roi
        band_lo = max(0, edge_x - int(edge_band_px))
        band_hi = min(roi_width, edge_x + int(edge_band_px) + 1)
        if band_hi <= band_lo:
            streak = 0
            continue
        signal = int(active[:, band_lo:band_hi].sum())
        total = int(active.sum())
        clutter = max(total - signal, 0)
        ratio = float(signal) / float(max(clutter, 1))
        if ratio >= float(ratio_threshold) and signal > 0:
            if first_hit is None:
                first_hit = t
            streak += 1
            if streak >= int(consecutive_k):
                return first_hit, first_hit - int(scenario.phase_slices["whiteout"].stop)
        else:
            streak = 0
            first_hit = None
    return None, None


def _phase_spans(phase_slices: dict[str, slice]) -> list[tuple[str, int, int]]:
    return [(name, phase.start, phase.stop) for name, phase in phase_slices.items()]


def _save_comparison_plot(out_path: Path, scenario, frames_np: np.ndarray, no_adapt, with_adapt) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    track_y, track_x = with_adapt.track_xy
    x_axis = np.arange(frames_np.shape[0], dtype=np.int32)
    luma_trace = frames_np[:, track_y, track_x]

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    phase_colors = {
        "pre_white_motion": "#d9d9d9",
        "whiteout": "#ffe08a",
        "post_white_motion": "#c7dcef",
    }
    for ax in axes:
        for name, start, stop in _phase_spans(scenario.phase_slices):
            ax.axvspan(start, stop - 1, color=phase_colors.get(name, "#f3f3f3"), alpha=0.35, linewidth=0)

    axes[0].plot(x_axis, luma_trace, color="#202020", linewidth=2.0, label="tracked pixel luma")
    axes[0].set_ylabel("luma")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Moving-box whiteout adaptation demo")

    axes[1].plot(x_axis, no_adapt.on_totals, color="#cf4446", linewidth=2.0, label="ON totals, adaptation off")
    axes[1].plot(x_axis, with_adapt.on_totals, color="#2da44e", linewidth=2.0, label="ON totals, adaptation on")
    axes[1].set_ylabel("ON events")
    axes[1].legend(loc="upper right")

    axes[2].plot(x_axis, no_adapt.off_totals, color="#cf4446", linewidth=2.0, label="OFF totals, adaptation off")
    axes[2].plot(x_axis, with_adapt.off_totals, color="#2da44e", linewidth=2.0, label="OFF totals, adaptation on")
    axes[2].set_ylabel("OFF events")
    axes[2].legend(loc="upper right")

    axes[3].plot(
        x_axis,
        no_adapt.ref_gap_trace,
        color="#cf4446",
        linewidth=2.0,
        label="|log_filt - ref_log|, adaptation off",
    )
    axes[3].plot(
        x_axis,
        with_adapt.ref_gap_trace,
        color="#2da44e",
        linewidth=2.0,
        label="|log_filt - ref_log|, adaptation on",
    )
    axes[3].set_ylabel("reference gap")
    axes[3].set_xlabel("frame index")
    axes[3].legend(loc="upper right")

    for name, start, stop in _phase_spans(scenario.phase_slices):
        axes[0].text(
            (start + stop - 1) * 0.5,
            axes[0].get_ylim()[1] * 0.96,
            name.replace("_", "\n"),
            ha="center",
            va="top",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_case_outputs(
    case_dir: Path,
    frames_np: np.ndarray,
    trace,
    fps: float,
    meta: dict[str, str],
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "on_counts.npy", trace.on_counts)
    np.save(case_dir / "off_counts.npy", trace.off_counts)
    np.save(case_dir / "on_totals.npy", trace.on_totals)
    np.save(case_dir / "off_totals.npy", trace.off_totals)
    np.save(case_dir / "ref_trace.npy", trace.ref_trace)
    np.save(case_dir / "log_filt_trace.npy", trace.log_filt_trace)
    np.save(case_dir / "ref_gap_trace.npy", trace.ref_gap_trace)
    save_preview_grid(
        trace.on_counts > 0,
        trace.off_counts > 0,
        case_dir / "preview_grid.png",
        stride=max(1, frames_np.shape[0] // 12),
    )
    write_summary(case_dir / "events_summary.txt", trace.on_totals, trace.off_totals, meta)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    args.out.mkdir(parents=True, exist_ok=True)

    scenario = build_moving_box_whiteout_scenario(
        width=args.width,
        height=args.height,
        fps=args.fps,
        pre_white_frames=args.pre_white_frames,
        whiteout_frames=args.whiteout_frames,
        post_white_frames=args.post_white_frames,
        background_level=args.background_level,
        box_level=args.box_level,
        white_level=args.white_level,
        box_width=args.box_width,
        box_height=args.box_height,
        reacquire_lag_frames=args.reacquire_lag_frames,
        sweep_px_per_frame=args.sweep_px_per_frame,
        stripe_period_px=args.stripe_period_px,
        stripe_contrast=args.stripe_contrast,
        background_texture_amplitude=args.background_texture_amplitude,
    )
    frames_np = scenario.frames
    track_xy = scenario.track_xy
    timestamps_np = np.arange(frames_np.shape[0], dtype=np.float32) / np.float32(scenario.fps)

    common_params = dict(
        pos_thres=args.pos_thres,
        neg_thres=args.neg_thres,
        sigma_thres=0.0,
        cutoff_hz=args.cutoff_hz,
        refractory_s=0.0,
        shot_noise_std=0.0,
        max_events_per_step=args.max_events_per_step,
    )
    params_no = DVSParams(adaptation_rate_hz=0.0, **common_params)
    params_yes = DVSParams(adaptation_rate_hz=args.adaptation_rate_hz, **common_params)

    no_adapt = run_stream_trace(frames_np, timestamps_np, params_no, seed=args.seed, track_xy=track_xy)
    with_adapt = run_stream_trace(frames_np, timestamps_np, params_yes, seed=args.seed, track_xy=track_xy)

    no_scan = run_scan_counts(frames_np, timestamps_np, params_no, seed=args.seed)
    yes_scan = run_scan_counts(frames_np, timestamps_np, params_yes, seed=args.seed)
    parity = {
        "adaptation_off": bool(np.array_equal(no_adapt.on_counts, no_scan[0]) and np.array_equal(no_adapt.off_counts, no_scan[1])),
        "adaptation_on": bool(np.array_equal(with_adapt.on_counts, yes_scan[0]) and np.array_equal(with_adapt.off_counts, yes_scan[1])),
    }

    whiteout_phase = scenario.phase_slices["whiteout"]
    post_phase = scenario.phase_slices["post_white_motion"]
    gap_tail = slice(max(whiteout_phase.stop - 3, whiteout_phase.start), whiteout_phase.stop)
    roi_y0, roi_y1, roi_x0, roi_x1 = scenario.right_roi

    no_reacq_frame, no_reacq_lag = _reacquire_frame(
        no_adapt.on_counts,
        no_adapt.off_counts,
        scenario,
        ratio_threshold=args.reacquire_ratio,
        consecutive_k=args.reacquire_consecutive,
        edge_band_px=args.reacquire_edge_band_px,
    )
    yes_reacq_frame, yes_reacq_lag = _reacquire_frame(
        with_adapt.on_counts,
        with_adapt.off_counts,
        scenario,
        ratio_threshold=args.reacquire_ratio,
        consecutive_k=args.reacquire_consecutive,
        edge_band_px=args.reacquire_edge_band_px,
    )

    metrics = {
        "scenario": scenario.name,
        "track_xy": [int(track_xy[0]), int(track_xy[1])],
        "right_roi_y0_y1_x0_x1": [int(roi_y0), int(roi_y1), int(roi_x0), int(roi_x1)],
        "stream_scan_parity": parity,
        "whiteout_on_total_no_adaptation": phase_total(no_adapt.on_totals, whiteout_phase),
        "whiteout_on_total_with_adaptation": phase_total(with_adapt.on_totals, whiteout_phase),
        "post_white_on_total_no_adaptation": phase_total(no_adapt.on_totals, post_phase),
        "post_white_on_total_with_adaptation": phase_total(with_adapt.on_totals, post_phase),
        "post_white_right_roi_on_total_no_adaptation": int(no_adapt.on_counts[post_phase, roi_y0:roi_y1, roi_x0:roi_x1].sum()),
        "post_white_right_roi_on_total_with_adaptation": int(with_adapt.on_counts[post_phase, roi_y0:roi_y1, roi_x0:roi_x1].sum()),
        "post_white_right_roi_off_total_no_adaptation": int(no_adapt.off_counts[post_phase, roi_y0:roi_y1, roi_x0:roi_x1].sum()),
        "post_white_right_roi_off_total_with_adaptation": int(with_adapt.off_counts[post_phase, roi_y0:roi_y1, roi_x0:roi_x1].sum()),
        "whiteout_gap_tail_mean_no_adaptation": float(np.mean(no_adapt.ref_gap_trace[gap_tail])),
        "whiteout_gap_tail_mean_with_adaptation": float(np.mean(with_adapt.ref_gap_trace[gap_tail])),
        "post_white_last_on_frame_no_adaptation": phase_last_nonzero(no_adapt.on_totals, post_phase),
        "post_white_last_on_frame_with_adaptation": phase_last_nonzero(with_adapt.on_totals, post_phase),
        "post_white_right_roi_reacquire_frame_no_adaptation": (-1 if no_reacq_frame is None else int(no_reacq_frame)),
        "post_white_right_roi_reacquire_frame_with_adaptation": (-1 if yes_reacq_frame is None else int(yes_reacq_frame)),
        "post_white_right_roi_reacquire_lag_frames_no_adaptation": (-1 if no_reacq_lag is None else int(no_reacq_lag)),
        "post_white_right_roi_reacquire_lag_frames_with_adaptation": (-1 if yes_reacq_lag is None else int(yes_reacq_lag)),
    }
    if no_reacq_lag is not None and yes_reacq_lag is not None:
        metrics["post_white_right_roi_reacquire_lag_delta_frames"] = int(no_reacq_lag - yes_reacq_lag)
    else:
        metrics["post_white_right_roi_reacquire_lag_delta_frames"] = -1
    metrics["whiteout_on_reduction_pct"] = float(
        100.0
        * (1.0 - (metrics["whiteout_on_total_with_adaptation"] / max(metrics["whiteout_on_total_no_adaptation"], 1)))
    )

    benchmarks = {
        "adaptation_off": {
            "streaming": benchmark_streaming(frames_np, timestamps_np, params_no, seed=args.seed, repeats=args.benchmark_repeats),
            "scan": benchmark_scan(frames_np, timestamps_np, params_no, seed=args.seed, repeats=args.benchmark_repeats),
        },
        "adaptation_on": {
            "streaming": benchmark_streaming(frames_np, timestamps_np, params_yes, seed=args.seed, repeats=args.benchmark_repeats),
            "scan": benchmark_scan(frames_np, timestamps_np, params_yes, seed=args.seed, repeats=args.benchmark_repeats),
        },
    }

    _write_case_outputs(
        args.out / "adaptation_off",
        frames_np,
        no_adapt,
        scenario.fps,
        {
            "scenario": scenario.name,
            "adaptation_rate_hz": "0.0",
            "track_xy": str(track_xy),
            "stream_scan_parity": str(parity["adaptation_off"]),
        },
    )
    _write_case_outputs(
        args.out / "adaptation_on",
        frames_np,
        with_adapt,
        scenario.fps,
        {
            "scenario": scenario.name,
            "adaptation_rate_hz": str(args.adaptation_rate_hz),
            "track_xy": str(track_xy),
            "stream_scan_parity": str(parity["adaptation_on"]),
        },
    )

    if args.triptych:
        write_triptych_mp4(
            frames_np,
            with_adapt.on_counts,
            with_adapt.off_counts,
            no_adapt.on_counts,
            no_adapt.off_counts,
            args.out / "adaptation_triptych.mp4",
            fps=scenario.fps,
            roi_xyxy=(int(roi_y0), int(roi_y1), int(roi_x0), int(roi_x1)),
            edge_column_lookup=scenario.edge_column_lookup,
            whiteout_start_frame=int(whiteout_phase.start),
            whiteout_stop_frame=int(whiteout_phase.stop),
            reacquired_on_frame=yes_reacq_frame,
            reacquired_off_frame=no_reacq_frame,
            decay_frames=args.decay_frames,
            display_scale=args.display_scale,
        )

    np.save(args.out / "synthetic_frames.npy", frames_np)
    np.save(args.out / "synthetic_timestamps.npy", timestamps_np)
    _save_comparison_plot(args.out / "adaptation_comparison.png", scenario, frames_np, no_adapt, with_adapt)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    (args.out / "benchmark.json").write_text(json.dumps(benchmarks, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {args.out} with adaptation comparison artifacts")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
