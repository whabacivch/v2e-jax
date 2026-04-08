#!/usr/bin/env python3
"""CLI: load KITTI-style frames or video, run streaming JAX DVS, save counts + preview.

Streaming pipeline
------------------
Frames are processed one interval at a time:

    for each pair (f0, f1):
        sub_frames, sub_ts = upsample_interval_linear(f0, f1, t0, t1, n_steps)
        for sub_frame, sub_t in zip(sub_frames, sub_ts):
            state, on_k, off_k = step_fn(state, sub_frame, sub_t, dt_sub)

``dvs_step`` is JIT-compiled once on the first call and runs at full GPU speed
for all subsequent frames — no Python overhead per frame after warm-up.

The full-sequence ``lax.scan`` path is also available via ``--mode scan``
for offline batch processing (faster for short clips, same physics).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT / "src", _ROOT):
    _s = str(_d)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from data.loaders import load_sequence, load_video
from helpers.render import (
    counts_per_timestep,
    infer_playback_fps,
    save_preview_grid,
    write_side_by_side_mp4,
    write_summary,
)
from v2e_jax import (
    DVSParams,
    VALID_SUBFRAME_SCHEDULES,
    build_threshold_maps,
    choose_adaptive_steps,
    dvs_init,
    make_dvs_step_fn,
    run_dvs_count_scan,
    temporal_upsample_adaptive_linear,
    temporal_upsample_linear,
    upsample_interval_linear,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_dir", type=Path, default=None, help="directory of ordered images")
    src.add_argument("--video", type=Path, default=None, help="video file (mp4/avi/...)")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--downscale", type=int, default=1, help="integer downscale factor")
    p.add_argument("--fps", type=float, default=10.0, help="FPS for image folders")
    p.add_argument("--video_fps_override", type=float, default=None)
    p.add_argument(
        "--steps_per_interval", type=int, default=1,
        help="sub-frames per original frame interval (temporal upsampling)"
    )
    p.add_argument(
        "--adaptive_upsample", action=argparse.BooleanOptionalAction, default=False,
        help="adapt steps_per_interval per interval based on motion score"
    )
    p.add_argument(
        "--subframe_schedule",
        choices=VALID_SUBFRAME_SCHEDULES,
        default="uniform",
        help="timing schedule for subframes within each source-frame interval",
    )
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pos_thres", type=float, default=0.15)
    p.add_argument("--neg_thres", type=float, default=0.15)
    p.add_argument("--sigma_thres", type=float, default=0.03)
    p.add_argument("--cutoff_hz", type=float, default=15.0)
    p.add_argument("--refractory_s", type=float, default=0.0005)
    p.add_argument("--shot_noise_std", type=float, default=0.0)
    p.add_argument(
        "--adaptation_rate_hz",
        type=float,
        default=0.0,
        help="reference adaptation rate toward the filtered signal; 0 disables recovery adaptation",
    )
    p.add_argument(
        "--mode", choices=["stream", "scan"], default="stream",
        help="stream: online frame-by-frame (default); scan: full lax.scan over clip"
    )
    p.add_argument(
        "--side_by_side", action=argparse.BooleanOptionalAction, default=True,
        help="write side_by_side.mp4"
    )
    p.add_argument("--playback_fps", type=float, default=None)
    return p.parse_args(argv)


def _run_streaming(
    frames_np: np.ndarray,
    ts_np: np.ndarray,
    params: DVSParams,
    key: jnp.ndarray,
    steps_per_interval: int,
    adaptive: bool,
    subframe_schedule: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream frames through dvs_step one sub-frame at a time.

    Returns ``(on_counts, off_counts, frames_out, timestamps_out)`` as numpy arrays.
    All sub-frame outputs are collected so downstream rendering sees the same
    resolution as the scan path.
    """
    h, w = frames_np.shape[1], frames_np.shape[2]
    key_t, key_r = jr.split(key)
    pos_map, neg_map = build_threshold_maps(key_t, (h, w), params)

    step_fn = make_dvs_step_fn(params, key_r)

    # Initialise state from the first frame
    f0 = jnp.asarray(frames_np[0], dtype=jnp.float32)
    t0 = jnp.float32(ts_np[0])
    state = dvs_init(f0, t0, pos_map, neg_map, params)

    on_list: list[np.ndarray] = []
    off_list: list[np.ndarray] = []
    frames_out: list[np.ndarray] = []
    ts_out: list[float] = []

    # Frame 0: no events on init frame, emit zeros to keep indexing aligned
    on_list.append(np.zeros((h, w), dtype=np.int16))
    off_list.append(np.zeros((h, w), dtype=np.int16))
    frames_out.append(frames_np[0])
    ts_out.append(float(ts_np[0]))

    t_count = frames_np.shape[0]
    for i in range(1, t_count):
        f1_np = frames_np[i]
        t1 = float(ts_np[i])
        t_prev = float(ts_np[i - 1])

        n = (
            choose_adaptive_steps(
                jnp.asarray(frames_np[i - 1], dtype=jnp.float32),
                jnp.asarray(f1_np, dtype=jnp.float32),
                steps_per_interval,
            )
            if adaptive
            else steps_per_interval
        )

        f0_jnp = jnp.asarray(frames_np[i - 1], dtype=jnp.float32)
        f1_jnp = jnp.asarray(f1_np, dtype=jnp.float32)
        sub_frames, sub_ts = upsample_interval_linear(
            f0_jnp, f1_jnp,
            jnp.float32(t_prev), jnp.float32(t1),
            n,
            subframe_schedule,
        )

        t_sub_prev = jnp.float32(t_prev)
        for s in range(n):
            sub_f = sub_frames[s]
            sub_t = sub_ts[s]
            dt_sub = sub_t - t_sub_prev
            state, on_k, off_k = step_fn(state, sub_f, sub_t, dt_sub)
            on_list.append(np.asarray(on_k))
            off_list.append(np.asarray(off_k))
            frames_out.append(np.asarray(sub_f))
            ts_out.append(float(sub_t))
            t_sub_prev = sub_t

    on_counts = np.stack(on_list, axis=0)
    off_counts = np.stack(off_list, axis=0)
    return on_counts, off_counts, np.stack(frames_out, axis=0), np.asarray(ts_out)


def _run_scan(
    frames_np: np.ndarray,
    ts_np: np.ndarray,
    params: DVSParams,
    key: jnp.ndarray,
    steps_per_interval: int,
    adaptive: bool,
    subframe_schedule: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full lax.scan over the entire (optionally upsampled) clip."""
    frames = jnp.asarray(frames_np, dtype=jnp.float32)
    timestamps = jnp.asarray(ts_np, dtype=jnp.float32)

    if steps_per_interval > 1:
        if adaptive:
            frames, timestamps, _ = temporal_upsample_adaptive_linear(
                frames,
                timestamps,
                steps_per_interval,
                schedule=subframe_schedule,
            )
        else:
            frames, timestamps = temporal_upsample_linear(
                frames,
                timestamps,
                steps_per_interval,
                schedule=subframe_schedule,
            )

    h, w = int(frames.shape[1]), int(frames.shape[2])
    key_t, key_r = jr.split(key)
    pos_map, neg_map = build_threshold_maps(key_t, (h, w), params)

    _, on_stack, off_stack = run_dvs_count_scan(frames, timestamps, pos_map, neg_map, key_r, params)
    jax.block_until_ready(on_stack)

    return (
        np.asarray(on_stack),
        np.asarray(off_stack),
        np.asarray(frames),
        np.asarray(timestamps),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.video is not None:
        frames_np, ts_np, fps_used = load_video(
            args.video,
            max_frames=args.max_frames,
            downscale=args.downscale,
            fps_override=args.video_fps_override,
        )
        fps_meta = fps_used
    else:
        assert args.input_dir is not None
        fps_meta = args.fps
        frames_np, ts_np = load_sequence(
            args.input_dir,
            max_frames=args.max_frames,
            downscale=args.downscale,
            fps=args.fps,
        )

    params = DVSParams(
        pos_thres=args.pos_thres,
        neg_thres=args.neg_thres,
        sigma_thres=args.sigma_thres,
        cutoff_hz=args.cutoff_hz,
        refractory_s=args.refractory_s,
        shot_noise_std=args.shot_noise_std,
        adaptation_rate_hz=args.adaptation_rate_hz,
    )
    key = jr.PRNGKey(args.seed)

    if args.mode == "stream":
        on_counts_np, off_counts_np, frames_out, ts_out = _run_streaming(
            frames_np, ts_np, params, key,
            args.steps_per_interval, args.adaptive_upsample, args.subframe_schedule,
        )
    else:
        on_counts_np, off_counts_np, frames_out, ts_out = _run_scan(
            frames_np, ts_np, params, key,
            args.steps_per_interval, args.adaptive_upsample, args.subframe_schedule,
        )

    on_masks_np = on_counts_np > 0
    off_masks_np = off_counts_np > 0
    on_totals, off_totals, tot = counts_per_timestep(on_counts_np, off_counts_np)

    np.save(args.out / "on_counts.npy", on_counts_np)
    np.save(args.out / "off_counts.npy", off_counts_np)
    np.save(args.out / "on_masks.npy", on_masks_np)
    np.save(args.out / "off_masks.npy", off_masks_np)
    np.save(args.out / "on_totals.npy", on_totals)
    np.save(args.out / "off_totals.npy", off_totals)
    np.save(args.out / "total_events.npy", tot)

    src_label = str(args.video.resolve()) if args.video is not None else str(args.input_dir.resolve())
    meta = {
        "input": src_label,
        "video_fps": str(fps_meta),
        "frames": str(frames_out.shape[0]),
        "height": str(frames_out.shape[1]),
        "width": str(frames_out.shape[2]),
        "steps_per_interval": str(args.steps_per_interval),
        "adaptive_upsample": str(args.adaptive_upsample),
        "subframe_schedule": args.subframe_schedule,
        "adaptation_rate_hz": str(args.adaptation_rate_hz),
        "mode": args.mode,
        "seed": str(args.seed),
        "jax_devices": str(jax.devices()),
    }

    if args.side_by_side:
        pb_fps = args.playback_fps if args.playback_fps is not None else infer_playback_fps(ts_out)
        sbs_path = args.out / "side_by_side.mp4"
        write_side_by_side_mp4(frames_out, on_masks_np, off_masks_np, sbs_path, fps=pb_fps)
        meta["side_by_side_mp4"] = str(sbs_path.resolve())
        meta["playback_fps"] = str(pb_fps)

    write_summary(args.out / "events_summary.txt", on_totals, off_totals, meta)
    save_preview_grid(
        on_masks_np,
        off_masks_np,
        args.out / "preview_grid.png",
        stride=max(1, frames_out.shape[0] // 12),
    )

    extra = " + side_by_side.mp4" if args.side_by_side else ""
    print(f"Wrote {args.out} (total events {int(tot.sum())}){extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
