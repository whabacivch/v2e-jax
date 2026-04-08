"""Temporal upsampling backends for DVS simulation.

Streaming-first design
----------------------
The primary API is **interval-level**: given a pair of adjacent frames
``(f0, f1)`` and their timestamps, produce ``n_steps`` interpolated sub-frames.
These functions are fully vectorised and ``@jax.jit``-compatible — they trace
once and run without Python loops, so you can call them inside a streaming
outer loop without retracing overhead (as long as ``n_steps`` and ``(H, W)``
are constant). The timing schedule within an interval is configurable.

    sub_frames, sub_ts = upsample_interval_linear(f0, f1, t0, t1, n_steps)
    for i in range(n_steps):
        state, on_k, off_k = step_fn(state, sub_frames[i], sub_ts[i], dt_sub)

Batch wrappers
--------------
``temporal_upsample_linear`` / ``temporal_upsample_motion_compensated`` are
retained for offline pipelines. The adaptive variant remains Python-loop based
because it computes ``n_steps`` dynamically per interval; use it for
pre-generation, not inside a traced loop.

Motion-compensated notes
-------------------------
Flow estimation is *not* included. Pass pre-computed dense flow arrays from
any external estimator (RAFT, GMFlow, etc.) as ``flow_yx``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


VALID_SUBFRAME_SCHEDULES = ("uniform", "ease_in", "ease_out", "ease_in_out")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_base_grid(h: int, w: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.meshgrid(
        jnp.arange(h, dtype=jnp.float32),
        jnp.arange(w, dtype=jnp.float32),
        indexing="ij",
    )


def _bilinear_sample(img: jnp.ndarray, yy: jnp.ndarray, xx: jnp.ndarray) -> jnp.ndarray:
    """Sample ``img`` (H,W) at float coordinates, bilinear with border clamping."""
    h, w = img.shape
    y0 = jnp.clip(jnp.floor(yy).astype(jnp.int32), 0, h - 1)
    x0 = jnp.clip(jnp.floor(xx).astype(jnp.int32), 0, w - 1)
    y1 = jnp.clip(y0 + 1, 0, h - 1)
    x1 = jnp.clip(x0 + 1, 0, w - 1)
    wy = yy - y0.astype(jnp.float32)
    wx = xx - x0.astype(jnp.float32)
    return (
        (1.0 - wy) * (1.0 - wx) * img[y0, x0]
        + (1.0 - wy) * wx * img[y0, x1]
        + wy * (1.0 - wx) * img[y1, x0]
        + wy * wx * img[y1, x1]
    )


def _warp_image(img: jnp.ndarray, flow_yx: jnp.ndarray) -> jnp.ndarray:
    """Backward-warp ``img`` (H,W) using dense ``flow_yx`` (H,W,2) in (dy,dx) order."""
    h, w = img.shape
    yy, xx = _make_base_grid(h, w)
    return _bilinear_sample(img, yy - flow_yx[..., 0], xx - flow_yx[..., 1])


def _frame_motion_score(f0: jnp.ndarray, f1: jnp.ndarray) -> jnp.ndarray:
    """Cheap normalised motion proxy for adaptive step-count selection."""
    diff = jnp.abs(f1 - f0)
    denom = jnp.maximum(jnp.mean(jnp.abs(f0)) + jnp.mean(jnp.abs(f1)), 1e-6)
    return jnp.mean(diff) / denom


def _normalize_schedule_name(schedule: str) -> str:
    schedule = schedule.strip().lower().replace("-", "_")
    if schedule not in VALID_SUBFRAME_SCHEDULES:
        raise ValueError(
            f"unknown subframe schedule {schedule!r}; "
            f"expected one of {VALID_SUBFRAME_SCHEDULES}"
        )
    return schedule


def _schedule_alphas(n_steps: int, schedule: str) -> jnp.ndarray:
    """Return monotonically increasing alphas in ``(0, 1]`` for one interval."""
    schedule = _normalize_schedule_name(schedule)
    base = jnp.linspace(1.0 / n_steps, 1.0, n_steps, dtype=jnp.float32)
    if schedule == "uniform":
        return base
    if schedule == "ease_in":
        return base * base
    if schedule == "ease_out":
        return 1.0 - (1.0 - base) * (1.0 - base)
    # cosine easing gives denser early/late subframes and a smoother mid-interval ramp.
    return 0.5 - 0.5 * jnp.cos(jnp.pi * base)


# ---------------------------------------------------------------------------
# Streaming / JIT-able interval API  (primary interface)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnums=(4, 5))
def upsample_interval_linear(
    f0: jnp.ndarray,
    f1: jnp.ndarray,
    t0: jnp.ndarray,
    t1: jnp.ndarray,
    n_steps: int,
    schedule: str = "uniform",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Linearly interpolate one interval into ``n_steps`` sub-frames.

    Parameters
    ----------
    f0, f1 : (H, W) float32
        Bounding frames.
    t0, t1 : scalar float32
        Timestamps in seconds.
    n_steps : int
        Number of output sub-frames. ``n_steps=1`` returns just ``f1`` / ``t1``.

    Returns
    -------
    frames : (n_steps, H, W) float32
    timestamps : (n_steps,) float32

    Notes
    -----
    Sub-frames span ``(t0, t1]`` — ``f0`` is *not* repeated in the output
    (it was already yielded as the last sub-frame of the previous interval).
    Alphas are ``1/n .. n/n`` so the final sub-frame is always exactly ``f1``.

    ``n_steps`` and ``schedule`` must be static so this JITs cleanly.
    """
    alphas = _schedule_alphas(n_steps, schedule)
    frames = (1.0 - alphas[:, None, None]) * f0 + alphas[:, None, None] * f1
    timestamps = (1.0 - alphas) * t0 + alphas * t1
    return frames, timestamps


@functools.partial(jax.jit, static_argnums=(4, 7))
def upsample_interval_motion_compensated(
    f0: jnp.ndarray,
    f1: jnp.ndarray,
    t0: jnp.ndarray,
    t1: jnp.ndarray,
    n_steps: int,
    flow01: jnp.ndarray,
    flow10: jnp.ndarray,
    schedule: str = "uniform",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Motion-compensated interpolation for one interval.

    Parameters
    ----------
    f0, f1 : (H, W) float32
    t0, t1 : scalar float32
    n_steps : int  (static Python int for JIT)
    flow01 : (H, W, 2)
        Forward optical flow ``f0 -> f1`` in ``(dy, dx)`` pixel units.
    flow10 : (H, W, 2)
        Backward optical flow ``f1 -> f0``.

    Returns
    -------
    frames : (n_steps, H, W) float32
    timestamps : (n_steps,) float32

    Notes
    -----
    Uses bidirectional warping: warp ``f0`` forward by ``a*flow01`` and
    ``f1`` backward by ``(1-a)*flow10``, blend linearly. Final sub-frame
    is always the exact ``f1`` (no warp artefacts at the boundary).

    If you only have one flow direction, negate it for the other — that is
    a coarse approximation, not true inverse-flow warping for large motions.
    """
    alphas = _schedule_alphas(n_steps, schedule)

    def interp_one(a: jnp.ndarray) -> jnp.ndarray:
        warp0 = _warp_image(f0, a * flow01)
        warp1 = _warp_image(f1, (1.0 - a) * flow10)
        return (1.0 - a) * warp0 + a * warp1

    # vmap over alphas — one warp per alpha, fully on-device
    frames_mid = jax.vmap(interp_one)(alphas)                   # (n_steps, H, W)
    # Force final sub-frame to exact f1 (no float accumulation error)
    frames = frames_mid.at[-1].set(f1)
    timestamps = (1.0 - alphas) * t0 + alphas * t1
    return frames, timestamps


def choose_adaptive_steps(
    f0: jnp.ndarray,
    f1: jnp.ndarray,
    base_steps: int,
    *,
    max_steps: int = 8,
    motion_scale: float = 24.0,
) -> int:
    """Return a static Python int for adaptive step count based on motion score.

    This runs on the host (materialises a scalar). Call it once per interval
    *before* calling :func:`upsample_interval_linear`, then pass the result
    as ``n_steps``. JAX will retrace for each distinct ``n_steps`` value but
    that's bounded by ``max_steps``.
    """
    score = float(_frame_motion_score(f0, f1))
    n = int(round(base_steps + motion_scale * score))
    return max(1, min(n, max_steps))


# ---------------------------------------------------------------------------
# Batch / offline wrappers  (kept for backwards compatibility)
# ---------------------------------------------------------------------------

def temporal_upsample_linear(
    frames: jnp.ndarray,
    timestamps: jnp.ndarray,
    steps_per_interval: int,
    *,
    schedule: str = "uniform",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate a full clip uniformly. Returns ``(T_out, H, W)`` and ``(T_out,)``.

    Output length: ``(T-1) * steps_per_interval + 1`` (endpoints preserved).
    If ``steps_per_interval <= 1``, returns inputs unchanged.
    """
    schedule = _normalize_schedule_name(schedule)
    if steps_per_interval <= 1 or frames.shape[0] < 2:
        return frames, timestamps

    out_frames = [frames[0]]
    out_ts = [timestamps[0]]
    for i in range(frames.shape[0] - 1):
        sub_f, sub_t = upsample_interval_linear(
            frames[i],
            frames[i + 1],
            timestamps[i],
            timestamps[i + 1],
            steps_per_interval,
            schedule,
        )
        out_frames.extend(sub_f)
        out_ts.extend(sub_t)
    return jnp.stack(out_frames, axis=0), jnp.stack(out_ts, axis=0)


def temporal_upsample_adaptive_linear(
    frames: jnp.ndarray,
    timestamps: jnp.ndarray,
    base_steps_per_interval: int,
    *,
    max_steps_per_interval: int = 8,
    motion_scale: float = 24.0,
    schedule: str = "uniform",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Adaptive linear upsampling. Returns ``(out_frames, out_timestamps, steps_array)``.

    Per-interval step counts are chosen by motion score. This function is
    Python-loop based and not JIT-compatible; use it for offline pre-generation.
    For online use, call :func:`choose_adaptive_steps` +
    :func:`upsample_interval_linear` per interval.
    """
    schedule = _normalize_schedule_name(schedule)
    t_in = frames.shape[0]
    if t_in < 2:
        return frames, timestamps, jnp.zeros((0,), dtype=jnp.int32)

    out_frames = [frames[0]]
    out_ts = [timestamps[0]]
    steps = []

    for i in range(t_in - 1):
        n = choose_adaptive_steps(
            frames[i], frames[i + 1],
            base_steps_per_interval,
            max_steps=max_steps_per_interval,
            motion_scale=motion_scale,
        )
        steps.append(n)
        sub_f, sub_t = upsample_interval_linear(
            frames[i], frames[i + 1],
            timestamps[i], timestamps[i + 1],
            n,
            schedule,
        )
        out_frames.extend(sub_f)
        out_ts.extend(sub_t)

    return (
        jnp.stack(out_frames, axis=0),
        jnp.stack(out_ts, axis=0),
        jnp.asarray(steps, dtype=jnp.int32),
    )


def temporal_upsample_motion_compensated(
    frames: jnp.ndarray,
    timestamps: jnp.ndarray,
    steps_per_interval: int,
    forward_flows: jnp.ndarray | None = None,
    backward_flows: jnp.ndarray | None = None,
    *,
    schedule: str = "uniform",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Motion-compensated temporal upsampling for a full clip.

    Parameters
    ----------
    frames : (T, H, W) float32
    timestamps : (T,) float32
    steps_per_interval : int
    forward_flows : (T-1, H, W, 2) or None
        Dense flow ``frame_i -> frame_{i+1}`` in ``(dy, dx)`` pixel units.
    backward_flows : (T-1, H, W, 2) or None
        Dense flow ``frame_{i+1} -> frame_i``.

    Notes
    -----
    Falls back to linear interpolation when no flows are provided.
    If only one direction is given, the other is approximated by negation.
    """
    schedule = _normalize_schedule_name(schedule)
    if steps_per_interval <= 1 or frames.shape[0] < 2:
        return frames, timestamps

    if forward_flows is None and backward_flows is None:
        return temporal_upsample_linear(frames, timestamps, steps_per_interval, schedule=schedule)

    if forward_flows is None:
        forward_flows = -backward_flows
    if backward_flows is None:
        backward_flows = -forward_flows

    t_in = frames.shape[0]
    out_frames = [frames[0]]
    out_ts = [timestamps[0]]

    for i in range(t_in - 1):
        sub_f, sub_t = upsample_interval_motion_compensated(
            frames[i], frames[i + 1],
            timestamps[i], timestamps[i + 1],
            steps_per_interval,
            forward_flows[i],
            backward_flows[i],
            schedule=schedule,
        )
        out_frames.extend(sub_f)
        out_ts.extend(sub_t)

    return jnp.stack(out_frames, axis=0), jnp.stack(out_ts, axis=0)
