"""Shared helpers for synthetic adaptation demos and validation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from v2e_jax import DVSParams, build_threshold_maps, dvs_init, make_dvs_step_fn, run_dvs_count_scan


@dataclass(frozen=True)
class Scenario:
    """Synthetic brightness schedule and phase metadata."""

    name: str
    levels: np.ndarray
    phase_slices: dict[str, slice]
    fps: float


@dataclass(frozen=True)
class MovingBoxScenario:
    """Moving-box whiteout clip metadata.

    ``right_roi`` is ``(y0, y1, x0, x1)`` in absolute frame coordinates (half-open,
    numpy-slice conventions). ``edge_column_lookup[t]`` is the leading box-edge
    x coordinate *inside the ROI* (``0 .. roi_width - 1``) at frame ``t``, or
    ``-1`` if the box is not inside the ROI on that frame. Both are consumed by
    the triptych renderer (for the overlay rectangle) and the reacquire-frame
    metric (for the signal-band slice) in ``adaptation_demo``.
    """

    name: str
    frames: np.ndarray
    phase_slices: dict[str, slice]
    fps: float
    track_xy: tuple[int, int]
    right_roi: tuple[int, int, int, int]
    edge_column_lookup: np.ndarray


@dataclass(frozen=True)
class StreamTrace:
    """Outputs captured from a streaming run."""

    on_counts: np.ndarray
    off_counts: np.ndarray
    on_totals: np.ndarray
    off_totals: np.ndarray
    ref_trace: np.ndarray
    log_filt_trace: np.ndarray
    ref_gap_trace: np.ndarray
    timestamps: np.ndarray
    track_xy: tuple[int, int]


def build_outside_to_inside_scenario(
    *,
    fps: float = 30.0,
    baseline_frames: int = 60,
    bright_hold_frames: int = 90,
    recovery_frames: int = 90,
    reentry_frames: int = 30,
    settle_frames: int = 30,
    indoor_level: float = 85.0,
    outdoor_level: float = 320.0,
    reentry_level: float = 120.0,
) -> Scenario:
    """Return a simple outside-to-inside brightness schedule."""
    levels_parts = [
        np.full((baseline_frames,), indoor_level, dtype=np.float32),
        np.full((bright_hold_frames,), outdoor_level, dtype=np.float32),
        np.full((recovery_frames,), indoor_level, dtype=np.float32),
        np.full((reentry_frames,), reentry_level, dtype=np.float32),
        np.full((settle_frames,), indoor_level, dtype=np.float32),
    ]
    levels = np.concatenate(levels_parts, axis=0)

    cursor = 0
    phase_slices: dict[str, slice] = {}
    phase_defs = [
        ("indoor_baseline", baseline_frames),
        ("outdoor_hold", bright_hold_frames),
        ("indoor_recovery", recovery_frames),
        ("indoor_reentry", reentry_frames),
        ("settle", settle_frames),
    ]
    for name, length in phase_defs:
        phase_slices[name] = slice(cursor, cursor + length)
        cursor += length

    return Scenario(
        name="outside_to_inside",
        levels=levels,
        phase_slices=phase_slices,
        fps=float(fps),
    )


def build_moving_box_whiteout_scenario(
    *,
    width: int,
    height: int,
    fps: float = 30.0,
    pre_white_frames: int = 135,
    whiteout_frames: int = 15,
    post_white_frames: int = 150,
    background_level: float = 40.0,
    box_level: float = 170.0,
    white_level: float = 255.0,
    box_width: int = 48,
    box_height: int = 48,
    margin_x: int = 16,
    reacquire_lag_frames: int = 18,
    sweep_px_per_frame: float = 2.4,
    stripe_period_px: int = 12,
    stripe_contrast: float = 0.25,
    background_texture_amplitude: float = 0.15,
) -> MovingBoxScenario:
    """Build a continuous moving-box clip with a brief full-frame whiteout.

    The scene is intentionally designed to showcase post-whiteout recovery:

    - the background is a dim, static 2D sinusoid around ``background_level``
      with amplitude ``background_texture_amplitude * background_level`` — low
      enough not to trigger events on its own at the default thresholds, but
      non-flat so the box-edge sweep has scene context to recover against;
    - the box has internal vertical stripes alternating between
      ``(1 - stripe_contrast) * box_level`` and ``(1 + stripe_contrast) * box_level``
      so reacquisition provides multiple crisp edges, not a single rectangle;
    - the linear motion schedule is chosen so the box leading edge enters the
      right-side ROI roughly ``reacquire_lag_frames`` after the whiteout ends.

    Returns a ``MovingBoxScenario`` with:

    - ``right_roi``: ``(y0, y1, x0, x1)`` ints in absolute frame coords;
    - ``edge_column_lookup``: ``(T,)`` int32, the box leading-edge x coordinate
      *inside the ROI* at each frame, or ``-1`` if the box is not in the ROI.
    """
    total_frames = int(pre_white_frames + whiteout_frames + post_white_frames)

    # ------------------------------------------------------------------
    # Static textured background (dim, non-flat, event-neutral on its own)
    # ------------------------------------------------------------------
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    y_norm = yy / max(height - 1, 1)
    x_norm = xx / max(width - 1, 1)
    texture = (
        np.sin(2.0 * np.pi * 1.5 * x_norm) * 0.6
        + np.sin(2.0 * np.pi * 1.1 * y_norm + 0.7) * 0.4
    )
    amp = float(background_texture_amplitude) * float(background_level)
    background = np.clip(
        np.float32(background_level) + np.float32(amp) * texture.astype(np.float32),
        0.0,
        255.0,
    ).astype(np.float32)
    frames = np.broadcast_to(background[None, :, :], (total_frames, height, width)).copy()

    # ------------------------------------------------------------------
    # Striped box template (recomputed once, pasted each frame at x0)
    # ------------------------------------------------------------------
    stripe_period = max(2, int(stripe_period_px))
    stripe_lo = np.float32(max(0.0, (1.0 - stripe_contrast) * box_level))
    stripe_hi = np.float32(min(255.0, (1.0 + stripe_contrast) * box_level))
    col_phase = (np.arange(box_width, dtype=np.int32) // (stripe_period // 2)) & 1
    box_row = np.where(col_phase.astype(bool), stripe_hi, stripe_lo).astype(np.float32)
    box_template = np.broadcast_to(box_row[None, :], (box_height, box_width)).copy()

    # ------------------------------------------------------------------
    # Motion schedule: fixed-speed linear sweep, back-solved so that the
    # box leading edge enters the ROI shortly after the whiteout ends.
    #
    # Unlike the earlier implementation, the conceptual motion is not clamped
    # to the frame bounds. The box can be partially visible near either edge,
    # and only the visible intersection is pasted into the frame. This keeps
    # the motion continuous while still letting us back-solve the start point
    # from a fixed sweep speed.
    # ------------------------------------------------------------------
    y0 = int(max(0, min(height - box_height, height // 2 - box_height // 2)))

    white_start = int(pre_white_frames)
    white_stop = int(pre_white_frames + whiteout_frames)
    roi_x0_abs = int(max(0, width * 2 // 3))
    roi_x1_abs = int(width)

    target_t = float(max(0.0, min(total_frames - 1, white_stop + max(0, reacquire_lag_frames))))
    target_leading_abs = float(roi_x0_abs + max(2, min(box_width // 6, 10)))
    target_box_x = target_leading_abs - float(box_width - 1)
    sweep = float(sweep_px_per_frame)
    start_x = target_box_x - sweep * target_t
    x_positions_f = start_x + sweep * np.arange(total_frames, dtype=np.float32)
    x_positions = np.round(x_positions_f).astype(np.int32)

    # ------------------------------------------------------------------
    # Paste box + apply whiteout
    # ------------------------------------------------------------------
    for t in range(total_frames):
        if white_start <= t < white_stop:
            frames[t, :, :] = np.float32(white_level)
            continue
        x0 = int(x_positions[t])
        x1 = x0 + box_width
        vis_x0 = max(0, x0)
        vis_x1 = min(width, x1)
        if vis_x1 <= vis_x0:
            continue
        src_x0 = vis_x0 - x0
        src_x1 = src_x0 + (vis_x1 - vis_x0)
        frames[t, y0:y0 + box_height, vis_x0:vis_x1] = box_template[:, src_x0:src_x1]

    phase_slices = {
        "pre_white_motion": slice(0, pre_white_frames),
        "whiteout": slice(white_start, white_stop),
        "post_white_motion": slice(white_stop, total_frames),
    }

    # ROI: vertical span loose around the box track, x covers the right third.
    roi_y0 = int(max(0, y0 - box_height // 2))
    roi_y1 = int(min(height, y0 + box_height + box_height // 2))
    right_roi = (roi_y0, roi_y1, roi_x0_abs, roi_x1_abs)

    # track_xy is (y, x) to match StreamTrace convention (see run_stream_trace).
    track_xy = (
        int(max(0, min(height - 1, y0 + box_height // 2))),
        int(max(0, min(width - 1, (roi_x0_abs + roi_x1_abs) // 2))),
    )

    # edge_column_lookup: leading edge x in ROI-local coords, -1 if outside ROI.
    edge_column_lookup = np.full((total_frames,), -1, dtype=np.int32)
    for t in range(total_frames):
        if white_start <= t < white_stop:
            continue
        leading_abs = int(x_positions[t]) + box_width - 1
        if roi_x0_abs <= leading_abs < roi_x1_abs:
            edge_column_lookup[t] = int(leading_abs - roi_x0_abs)

    return MovingBoxScenario(
        name="moving_box_whiteout",
        frames=frames,
        phase_slices=phase_slices,
        fps=float(fps),
        track_xy=track_xy,
        right_roi=right_roi,
        edge_column_lookup=edge_column_lookup,
    )


def make_uniform_frames(levels: np.ndarray, height: int, width: int) -> np.ndarray:
    """Broadcast a scalar luma level per frame to a full frame stack."""
    clipped = np.clip(levels.astype(np.float32), 0.0, 255.0)
    return np.broadcast_to(clipped[:, None, None], (clipped.shape[0], height, width)).copy()


def make_textured_room_frames(levels: np.ndarray, height: int, width: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Create a textured room/window scene whose bright regions can saturate outdoors."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    y = yy / max(height - 1, 1)
    x = xx / max(width - 1, 1)

    wall = 0.18 + 0.08 * (1.0 - y) + 0.03 * np.cos(2.0 * np.pi * x)
    floor = 0.10 + 0.22 * y + 0.02 * np.sin(8.0 * np.pi * x)
    vignette = 1.0 - 0.30 * ((x - 0.5) ** 2 + (y - 0.55) ** 2)
    template = (wall + floor) * vignette

    window_y0 = int(round(0.16 * height))
    window_y1 = int(round(0.52 * height))
    window_x0 = int(round(0.34 * width))
    window_x1 = int(round(0.72 * width))
    template[window_y0:window_y1, window_x0:window_x1] += 0.68

    frame_thick = max(2, height // 64)
    template[window_y0:window_y0 + frame_thick, window_x0:window_x1] = 0.92
    template[window_y1 - frame_thick:window_y1, window_x0:window_x1] = 0.92
    template[window_y0:window_y1, window_x0:window_x0 + frame_thick] = 0.92
    template[window_y0:window_y1, window_x1 - frame_thick:window_x1] = 0.92

    mullion_x = (window_x0 + window_x1) // 2
    mullion_y = (window_y0 + window_y1) // 2
    template[window_y0:window_y1, mullion_x - frame_thick // 2:mullion_x + frame_thick // 2 + 1] = 0.88
    template[mullion_y - frame_thick // 2:mullion_y + frame_thick // 2 + 1, window_x0:window_x1] = 0.88

    blind_step = max(5, height // 24)
    for row in range(window_y0 + blind_step, window_y1, blind_step):
        template[row:row + 1, window_x0 + frame_thick:window_x1 - frame_thick] *= 0.78

    doorway_y0 = int(round(0.28 * height))
    doorway_y1 = int(round(0.96 * height))
    doorway_x0 = int(round(0.08 * width))
    doorway_x1 = int(round(0.22 * width))
    template[doorway_y0:doorway_y1, doorway_x0:doorway_x1] += 0.06
    template[doorway_y0:doorway_y0 + frame_thick, doorway_x0:doorway_x1] = 0.55
    template[doorway_y0:doorway_y1, doorway_x0:doorway_x0 + frame_thick] = 0.52
    template[doorway_y0:doorway_y1, doorway_x1 - frame_thick:doorway_x1] = 0.52

    desk_y0 = int(round(0.68 * height))
    desk_y1 = int(round(0.82 * height))
    desk_x0 = int(round(0.54 * width))
    desk_x1 = int(round(0.90 * width))
    template[desk_y0:desk_y1, desk_x0:desk_x1] += 0.10
    template[desk_y0:desk_y0 + frame_thick, desk_x0:desk_x1] = 0.62

    lamp_y0 = int(round(0.34 * height))
    lamp_y1 = int(round(0.66 * height))
    lamp_x0 = int(round(0.77 * width))
    lamp_x1 = int(round(0.86 * width))
    template[lamp_y0:lamp_y1, lamp_x0:lamp_x1] += 0.08
    template[lamp_y0:lamp_y0 + frame_thick, lamp_x0:lamp_x1] = 0.70

    picture_y0 = int(round(0.18 * height))
    picture_y1 = int(round(0.36 * height))
    picture_x0 = int(round(0.12 * width))
    picture_x1 = int(round(0.26 * width))
    template[picture_y0:picture_y1, picture_x0:picture_x1] += 0.05
    template[picture_y0:picture_y0 + frame_thick, picture_x0:picture_x1] = 0.64
    template[picture_y1 - frame_thick:picture_y1, picture_x0:picture_x1] = 0.64
    template[picture_y0:picture_y1, picture_x0:picture_x0 + frame_thick] = 0.64
    template[picture_y0:picture_y1, picture_x1 - frame_thick:picture_x1] = 0.64

    shelf_y0 = int(round(0.44 * height))
    shelf_y1 = int(round(0.72 * height))
    shelf_x0 = int(round(0.02 * width))
    shelf_x1 = int(round(0.10 * width))
    template[shelf_y0:shelf_y1, shelf_x0:shelf_x1] += 0.09
    book_step = max(3, width // 80)
    for col in range(shelf_x0 + book_step, shelf_x1, book_step):
        template[shelf_y0:shelf_y1, col:col + 1] *= 0.72

    crown_y = int(round(0.12 * height))
    template[crown_y:crown_y + 1, :] = 0.58

    floorboard_step = max(6, height // 18)
    for row in range(int(round(0.74 * height)), height, floorboard_step):
        template[row:row + 1, :] *= 0.82

    template = np.clip(template, 0.05, 1.0).astype(np.float32)
    frames = np.stack(
        [np.clip(template * float(level), 0.0, 255.0) for level in levels.astype(np.float32)],
        axis=0,
    ).astype(np.float32)

    track_xy = (min(max(window_y0 + 2, 0), height - 1), min(max(width // 2, 0), width - 1))
    return frames, track_xy


def make_timestamps(frame_count: int, fps: float) -> np.ndarray:
    return (np.arange(frame_count, dtype=np.float32) / np.float32(fps)).astype(np.float32)


def run_stream_trace(
    frames_np: np.ndarray,
    timestamps_np: np.ndarray,
    params: DVSParams,
    *,
    seed: int = 0,
    track_xy: tuple[int, int] | None = None,
) -> StreamTrace:
    """Run the streaming kernel and record counts plus one tracked pixel state."""
    if frames_np.ndim != 3:
        raise ValueError("expected frames_np with shape (T,H,W)")
    height, width = int(frames_np.shape[1]), int(frames_np.shape[2])
    if track_xy is None:
        track_xy = (height // 2, width // 2)
    y_idx, x_idx = track_xy

    key = jr.PRNGKey(seed)
    key_t, key_r = jr.split(key)
    pos_map, neg_map = build_threshold_maps(key_t, (height, width), params)
    step_fn = make_dvs_step_fn(params, key_r)

    frame0 = jnp.asarray(frames_np[0], dtype=jnp.float32)
    state = dvs_init(frame0, jnp.float32(timestamps_np[0]), pos_map, neg_map, params)

    on_frames = [np.zeros((height, width), dtype=np.int16)]
    off_frames = [np.zeros((height, width), dtype=np.int16)]
    ref_trace = [float(np.asarray(state.ref_log[y_idx, x_idx]))]
    log_filt_trace = [float(np.asarray(state.log_filt[y_idx, x_idx]))]
    ref_gap_trace = [abs(ref_trace[-1] - log_filt_trace[-1])]

    for i in range(1, frames_np.shape[0]):
        dt = jnp.float32(timestamps_np[i] - timestamps_np[i - 1])
        state, on_k, off_k = step_fn(
            state,
            jnp.asarray(frames_np[i], dtype=jnp.float32),
            jnp.float32(timestamps_np[i]),
            dt,
        )
        on_np = np.asarray(on_k, dtype=np.int16)
        off_np = np.asarray(off_k, dtype=np.int16)
        on_frames.append(on_np)
        off_frames.append(off_np)
        ref_now = float(np.asarray(state.ref_log[y_idx, x_idx]))
        log_filt_now = float(np.asarray(state.log_filt[y_idx, x_idx]))
        ref_trace.append(ref_now)
        log_filt_trace.append(log_filt_now)
        ref_gap_trace.append(abs(ref_now - log_filt_now))

    on_counts = np.stack(on_frames, axis=0)
    off_counts = np.stack(off_frames, axis=0)
    return StreamTrace(
        on_counts=on_counts,
        off_counts=off_counts,
        on_totals=on_counts.sum(axis=(1, 2)).astype(np.int64),
        off_totals=off_counts.sum(axis=(1, 2)).astype(np.int64),
        ref_trace=np.asarray(ref_trace, dtype=np.float32),
        log_filt_trace=np.asarray(log_filt_trace, dtype=np.float32),
        ref_gap_trace=np.asarray(ref_gap_trace, dtype=np.float32),
        timestamps=np.asarray(timestamps_np, dtype=np.float32),
        track_xy=track_xy,
    )


def run_scan_counts(
    frames_np: np.ndarray,
    timestamps_np: np.ndarray,
    params: DVSParams,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the batch scan helper with the same maps and key split as streaming."""
    height, width = int(frames_np.shape[1]), int(frames_np.shape[2])
    key = jr.PRNGKey(seed)
    key_t, key_r = jr.split(key)
    pos_map, neg_map = build_threshold_maps(key_t, (height, width), params)
    _, on_counts, off_counts = run_dvs_count_scan(
        jnp.asarray(frames_np, dtype=jnp.float32),
        jnp.asarray(timestamps_np, dtype=jnp.float32),
        pos_map,
        neg_map,
        key_r,
        params,
    )
    jax.block_until_ready(on_counts)
    return np.asarray(on_counts, dtype=np.int16), np.asarray(off_counts, dtype=np.int16)


def benchmark_streaming(
    frames_np: np.ndarray,
    timestamps_np: np.ndarray,
    params: DVSParams,
    *,
    seed: int = 0,
    repeats: int = 10,
) -> dict[str, float]:
    """Measure steady-state streaming runtime over the full synthetic clip."""
    if frames_np.shape[0] < 2:
        raise ValueError("need at least two frames for benchmark_streaming")

    height, width = int(frames_np.shape[1]), int(frames_np.shape[2])
    key = jr.PRNGKey(seed)
    key_t, key_r = jr.split(key)
    pos_map, neg_map = build_threshold_maps(key_t, (height, width), params)
    step_fn = make_dvs_step_fn(params, key_r)

    state = dvs_init(jnp.asarray(frames_np[0], dtype=jnp.float32), jnp.float32(timestamps_np[0]), pos_map, neg_map, params)
    for i in range(1, frames_np.shape[0]):
        state, on_k, off_k = step_fn(
            state,
            jnp.asarray(frames_np[i], dtype=jnp.float32),
            jnp.float32(timestamps_np[i]),
            jnp.float32(timestamps_np[i] - timestamps_np[i - 1]),
        )
    jax.block_until_ready(state.log_filt)
    jax.block_until_ready(on_k)
    jax.block_until_ready(off_k)

    timings = []
    for _ in range(max(repeats, 1)):
        state = dvs_init(jnp.asarray(frames_np[0], dtype=jnp.float32), jnp.float32(timestamps_np[0]), pos_map, neg_map, params)
        start = perf_counter()
        for i in range(1, frames_np.shape[0]):
            state, on_k, off_k = step_fn(
                state,
                jnp.asarray(frames_np[i], dtype=jnp.float32),
                jnp.float32(timestamps_np[i]),
                jnp.float32(timestamps_np[i] - timestamps_np[i - 1]),
            )
        jax.block_until_ready(state.log_filt)
        jax.block_until_ready(on_k)
        jax.block_until_ready(off_k)
        timings.append(perf_counter() - start)

    timings_np = np.asarray(timings, dtype=np.float64)
    timing = float(np.median(timings_np))
    simulated_steps = max(frames_np.shape[0] - 1, 1)
    return {
        "steady_state_ms_per_clip": float(1e3 * timing),
        "steady_state_ms_per_step": float(1e3 * timing / simulated_steps),
        "steady_state_steps_per_second": float(simulated_steps / timing),
    }


def benchmark_scan(
    frames_np: np.ndarray,
    timestamps_np: np.ndarray,
    params: DVSParams,
    *,
    seed: int = 0,
    repeats: int = 10,
) -> dict[str, float]:
    """Measure steady-state batch scan runtime over the full synthetic clip."""
    height, width = int(frames_np.shape[1]), int(frames_np.shape[2])
    key = jr.PRNGKey(seed)
    key_t, key_r = jr.split(key)
    pos_map, neg_map = build_threshold_maps(key_t, (height, width), params)
    frames = jnp.asarray(frames_np, dtype=jnp.float32)
    timestamps = jnp.asarray(timestamps_np, dtype=jnp.float32)

    _, on_counts, off_counts = run_dvs_count_scan(frames, timestamps, pos_map, neg_map, key_r, params)
    jax.block_until_ready(on_counts)
    jax.block_until_ready(off_counts)

    timings = []
    for _ in range(max(repeats, 1)):
        start = perf_counter()
        _, on_counts, off_counts = run_dvs_count_scan(frames, timestamps, pos_map, neg_map, key_r, params)
        jax.block_until_ready(on_counts)
        jax.block_until_ready(off_counts)
        timings.append(perf_counter() - start)

    timings_np = np.asarray(timings, dtype=np.float64)
    timing = float(np.median(timings_np))
    simulated_steps = max(frames_np.shape[0] - 1, 1)
    return {
        "steady_state_ms_per_clip": float(1e3 * timing),
        "steady_state_ms_per_step": float(1e3 * timing / simulated_steps),
        "steady_state_steps_per_second": float(simulated_steps / timing),
    }


def phase_total(values: np.ndarray, phase_slice: slice) -> int:
    return int(np.asarray(values[phase_slice]).sum())


def phase_last_nonzero(values: np.ndarray, phase_slice: slice) -> int:
    phase_values = np.asarray(values[phase_slice])
    nz = np.flatnonzero(phase_values > 0)
    return int(nz[-1]) if nz.size else -1
