#!/usr/bin/env python3
"""Real-time DVS simulation from a camera or video file at a target fps.

Sources
-------
--device INT        OpenCV device index (webcam, capture card, FPV via V4L2)
--video PATH        Any video file — simulates camera I/O via background thread
                    with frame dropping, identical code path to a real camera.

Visualisation modes  (--mode)
------------------------------
events   Default. Side-by-side: left=luma, right=green/red event overlay.
sharpen  Side-by-side: left=luma, right=event-sharpened luma. Event edges
         are blended into the grayscale image via unsharp masking on the
         ON+OFF activity map.
hdr      Side-by-side: left=luma, right=log-Retinex local contrast boost.
         Lifts shadows and compresses highlights guided by event activity.
motion   Side-by-side: left=luma, right=events coloured by motion type.
         Green = looming (approaching), red = receding, blue = lateral.
         Direct visual odometry (coarse-to-fine Gauss-Newton) is run each
         frame to derive the geometrically-grounded motion field.

The loop
--------
1. Background thread: cap.read() -> luma -> queue (drops if full, never blocks)
2. Main thread: dequeue -> upsample interval -> dvs_step (all on GPU)
3. Optional display: pull last sub-frame events to host for OpenCV window

Benchmark mode (--no_display --benchmark_frames N) prints per-frame timing
and overall throughput without any display overhead.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
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

try:
    import cv2
except ImportError:
    cv2 = None

from v2e_jax import (
    DVSParams,
    build_threshold_maps,
    dvs_init,
    make_dvs_step_fn,
    upsample_interval_linear,
)


def _opencv_has_gui() -> bool:
    """Return True when the installed OpenCV build exposes HighGUI windows."""
    if cv2 is None:
        return False
    try:
        info = cv2.getBuildInformation()
    except Exception:
        return False
    for line in info.splitlines():
        stripped = line.strip()
        if stripped.startswith("GUI:"):
            return not stripped.endswith("NONE")
    return False


def default_intrinsics_jax(h: int, w: int) -> jnp.ndarray:
    """Reasonable pinhole intrinsics for an uncalibrated webcam (~70° FoV)."""
    f = w / (2.0 * 0.700)
    return jnp.array([
        [f,  0., w / 2.0],
        [0., f,  h / 2.0],
        [0., 0., 1.      ],
    ], dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--device", type=int, default=None, help="V4L2/OpenCV camera index")
    src.add_argument("--video", type=Path, default=None, help="video file (simulates camera I/O)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=60.0, help="target capture fps")
    p.add_argument("--steps_per_interval", type=int, default=2, help="DVS sub-steps per frame")
    p.add_argument("--queue_size", type=int, default=4, help="frame queue depth (excess dropped)")
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
        help="reference adaptation rate toward the filtered signal; 0 disables",
    )
    p.add_argument(
        "--mode",
        choices=["events", "sharpen", "hdr", "motion"],
        default="events",
        help=(
            "visualisation mode: "
            "'events' = green/red event overlay (default), "
            "'sharpen' = event edge unsharp masking, "
            "'hdr' = log-Retinex local contrast boost (lifts shadows), "
            "'motion' = events coloured by looming/lateral/receding"
        ),
    )
    p.add_argument(
        "--display", action=argparse.BooleanOptionalAction, default=True,
        help="show OpenCV window (forces one host sync per frame)",
    )
    p.add_argument(
        "--benchmark_frames", type=int, default=None,
        help="if set, exit after N frames and print timing stats",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Camera / video capture thread
# ---------------------------------------------------------------------------

def _bgr_to_luma(bgr: np.ndarray) -> np.ndarray:
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _capture_thread(
    cap: cv2.VideoCapture,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
    target_w: int,
    target_h: int,
) -> None:
    """Decode frames in background. Drops frames if queue full — never stalls."""
    while not stop_event.is_set():
        ok, bgr = cap.read()
        if not ok:
            stop_event.set()
            break
        t = time.perf_counter()
        if bgr.shape[1] != target_w or bgr.shape[0] != target_h:
            bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        luma = _bgr_to_luma(bgr)
        try:
            frame_queue.put_nowait((luma, t))
        except queue.Full:
            pass  # drop — stale frame not worth blocking for


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _side_by_side(left_u8: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    """Concatenate a grey left panel and a BGR right panel with a divider."""
    h = left_u8.shape[0]
    left_bgr = cv2.merge([left_u8, left_u8, left_u8])
    divider = np.zeros((h, 2, 3), dtype=np.uint8)
    return np.concatenate([left_bgr, divider, right_bgr], axis=1)


def _vis_events(luma_np: np.ndarray, on_np: np.ndarray, off_np: np.ndarray) -> np.ndarray:
    """Standard green/red event overlay."""
    h, w = luma_np.shape
    g = np.clip(luma_np, 0, 255).astype(np.uint8)
    rgb = np.full((h, w, 3), 240, dtype=np.uint8)
    rgb[on_np] = (40, 200, 40)
    rgb[off_np] = (200, 40, 40)
    right = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return _side_by_side(g, right)


def _vis_sharpen(luma_np: np.ndarray, sharp_jax: jnp.ndarray) -> np.ndarray:
    """Event-sharpened luma on the right."""
    g = np.clip(luma_np, 0, 255).astype(np.uint8)
    sharp_np = np.asarray(jnp.clip(sharp_jax * 255.0, 0, 255)).astype(np.uint8)
    right = cv2.merge([sharp_np, sharp_np, sharp_np])
    return _side_by_side(g, right)


def _vis_motion(
    luma_np: np.ndarray,
    looming: np.ndarray,
    lateral: np.ndarray,
    event_mask: np.ndarray,
) -> np.ndarray:
    """Motion-coloured event overlay (green=approach, red=recede, blue=lateral)."""
    from v2e_jax.motion_field_vis import motion_colors_rgb_u8
    g = np.clip(luma_np, 0, 255).astype(np.uint8)
    colored_rgb = motion_colors_rgb_u8(looming, lateral, event_mask)
    right = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR)
    return _side_by_side(g, right)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if cv2 is None:
        print(
            'ERROR: run_camera requires OpenCV. Install camera support with: pip install -e ".[camera]"',
            file=sys.stderr,
        )
        return 2

    if args.display and not _opencv_has_gui():
        print(
            "WARNING: OpenCV was built without GUI support; disabling display. "
            "Install a GUI-enabled OpenCV build or rerun with --no-display.",
            file=sys.stderr,
        )
        args.display = False

    # --- open capture ---
    if args.video is not None:
        path = args.video.resolve()
        if not path.is_file():
            print(f"ERROR: video not found: {path}", file=sys.stderr)
            return 1
        cap = cv2.VideoCapture(str(path))
    else:
        cap = cv2.VideoCapture(args.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        print("ERROR: could not open capture source", file=sys.stderr)
        return 1

    ok, bgr0 = cap.read()
    if not ok:
        print("ERROR: could not read first frame", file=sys.stderr)
        return 1

    if bgr0.shape[1] != args.width or bgr0.shape[0] != args.height:
        bgr0 = cv2.resize(bgr0, (args.width, args.height), interpolation=cv2.INTER_AREA)
    luma0 = _bgr_to_luma(bgr0)
    h, w = luma0.shape

    # --- build sensor ---
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
    key_t, key_r = jr.split(key)
    pos_map, neg_map = build_threshold_maps(key_t, (h, w), params)

    t0 = jnp.float32(time.perf_counter())
    f0_jax = jnp.asarray(luma0)
    state = dvs_init(f0_jax, t0, pos_map, neg_map, params)
    step_fn = make_dvs_step_fn(params, key_r)

    # --- JIT warmup: DVS core (always) ---
    print(f"Warming up JIT [{args.mode} mode] ({h}x{w}, steps={args.steps_per_interval})...", flush=True)
    dt_warmup = jnp.float32(1.0 / args.fps / args.steps_per_interval)
    _s, _on, _off = step_fn(state, f0_jax, t0, dt_warmup)
    jax.block_until_ready(_on)
    _sf, _st = upsample_interval_linear(f0_jax, f0_jax, t0, t0, args.steps_per_interval)
    jax.block_until_ready(_sf)

    # --- JIT warmup: mode-specific ---
    _dummy_f = jnp.zeros((h, w), jnp.float32)
    _dummy_counts = jnp.zeros((h, w), jnp.int16)

    if args.mode == "sharpen":
        from v2e_jax.event_enhance import sharpen_luma_with_events
        jax.block_until_ready(
            sharpen_luma_with_events(_dummy_f, _dummy_counts, _dummy_counts)
        )

    elif args.mode == "hdr":
        from v2e_jax.event_enhance import hdr_local_contrast_boost
        jax.block_until_ready(
            hdr_local_contrast_boost(_dummy_f, _dummy_counts, _dummy_counts)
        )

    elif args.mode == "motion":
        from v2e_jax.direct_vo import direct_vo
        from v2e_jax.motion_field_vis import make_motion_field_fn

        K = default_intrinsics_jax(h, w)
        _motion_field_fn = make_motion_field_fn(K, (h, w))

        _xi_dummy = jnp.zeros(6, dtype=jnp.float32)
        _inv_depth_dummy = jnp.ones((h, w), dtype=jnp.float32)
        _loom, _lat = _motion_field_fn(_xi_dummy, _inv_depth_dummy)
        jax.block_until_ready(_loom)
        # Warm up direct_vo (coarse-to-fine GN — slower trace, only happens once)
        _xi_w, _inv_w, _ = direct_vo(_dummy_f, _dummy_f, 4, 5, 2.0, K)
        jax.block_until_ready(_xi_w)

    print("JIT ready. Starting capture loop.", flush=True)

    mode_labels = {
        "events":  "left:luma  right:events",
        "sharpen": "left:luma  right:event-sharpened",
        "hdr":     "left:luma  right:HDR local contrast (log-Retinex)",
        "motion":  "left:luma  right:motion-colors (green=approach red=recede blue=lateral)",
    }
    win_name = f"v2e_jax [{args.mode}] — {mode_labels[args.mode]}"

    if args.display:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, w * 2 + 2, h)

    frame_queue: queue.Queue = queue.Queue(maxsize=args.queue_size)
    stop_event = threading.Event()
    cap_thread = threading.Thread(
        target=_capture_thread,
        args=(cap, frame_queue, stop_event, w, h),
        daemon=True,
    )
    cap_thread.start()

    n_steps = args.steps_per_interval
    luma_prev = luma0
    t_prev = float(t0)

    frame_times: list[float] = []
    step_times: list[float] = []
    dropped = 0
    processed = 0
    prev_qsize = 0

    try:
        while not stop_event.is_set():
            try:
                luma_np, t_now = frame_queue.get(timeout=2.0)
            except queue.Empty:
                print("Source exhausted or stalled — exiting.")
                break

            dropped += max(0, frame_queue.qsize() - prev_qsize)
            prev_qsize = frame_queue.qsize()

            frame_start = time.perf_counter()

            f0_jax = jnp.asarray(luma_prev)
            f1_jax = jnp.asarray(luma_np)
            t0_jax = jnp.float32(t_prev)
            t1_jax = jnp.float32(t_now)
            dt_interval = t_now - t_prev
            dt_sub = jnp.float32(dt_interval / n_steps)

            sub_frames, sub_ts = upsample_interval_linear(f0_jax, f1_jax, t0_jax, t1_jax, n_steps)

            step_start = time.perf_counter()
            for s in range(n_steps):
                state, on_k, off_k = step_fn(state, sub_frames[s], sub_ts[s], dt_sub)
            jax.block_until_ready(on_k)
            step_end = time.perf_counter()

            luma_prev = luma_np
            t_prev = t_now
            processed += 1

            frame_times.append(step_end - frame_start)
            step_times.append(step_end - step_start)

            if args.display:
                on_np = np.asarray(on_k > 0)
                off_np = np.asarray(off_k > 0)
                n_ev = int((on_np | off_np).sum())

                if args.mode == "events":
                    vis = _vis_events(luma_np, on_np, off_np)

                elif args.mode == "sharpen":
                    luma01 = f1_jax / 255.0
                    sharp_jax = sharpen_luma_with_events(luma01, on_k, off_k)
                    vis = _vis_sharpen(luma_np, sharp_jax)

                elif args.mode == "hdr":
                    luma01 = f1_jax / 255.0
                    hdr_jax = hdr_local_contrast_boost(luma01, on_k, off_k)
                    vis = _vis_sharpen(luma_np, hdr_jax)  # same side-by-side layout

                elif args.mode == "motion":
                    xi, inv_depth, _info = direct_vo(f0_jax, f1_jax, 4, 5, 2.0, K)
                    looming, lateral = _motion_field_fn(xi, inv_depth)
                    vis = _vis_motion(
                        luma_np,
                        np.asarray(looming),
                        np.asarray(lateral),
                        on_np | off_np,
                    )

                fps_live = 1.0 / (frame_times[-1] + 1e-9)
                cv2.putText(
                    vis,
                    f"events:{n_ev} | step {step_times[-1]*1000:.1f}ms | {fps_live:.0f}fps",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 220, 255), 1, cv2.LINE_AA,
                )
                cv2.imshow(win_name, vis)
                key_pressed = cv2.waitKey(1) & 0xFF
                if key_pressed == ord("q") or key_pressed == 27:
                    break

            if args.benchmark_frames is not None and processed >= args.benchmark_frames:
                break

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    if frame_times:
        ft = np.array(frame_times) * 1000
        st = np.array(step_times) * 1000
        print(f"\n{'='*52}")
        print(f"  Mode             : {args.mode}")
        print(f"  Frames processed : {processed}")
        print(f"  Frames dropped   : {dropped}")
        print(f"  Resolution       : {w}x{h}")
        print(f"  steps/interval   : {n_steps}")
        print(f"  JAX devices      : {jax.devices()}")
        print(f"{'─'*52}")
        print(f"  Frame latency (total loop, ms)")
        print(f"    mean  {ft.mean():.2f}   median {np.median(ft):.2f}   p99 {np.percentile(ft,99):.2f}")
        print(f"  DVS step time (GPU kernel + sync, ms)")
        print(f"    mean  {st.mean():.2f}   median {np.median(st):.2f}   p99 {np.percentile(st,99):.2f}")
        print(f"  Effective throughput : {1000/ft.mean():.1f} fps")
        print(f"  Sub-frame throughput : {1000/st.mean()*n_steps:.1f} sub-fps")
        print(f"{'='*52}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
