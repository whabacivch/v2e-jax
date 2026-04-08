"""JAX DVS emulator: log-domain event simulation with streaming-first design.

Architecture
------------
The primary API is **streaming**: one frame at a time, zero retracing.

    state = dvs_init(frame0, t0, pos_map, neg_map, params)
    for frame, t in source:
        state, on_k, off_k = dvs_step(state, frame, t)

``dvs_step`` is decorated with ``@jax.jit`` and traces once for a given
``(H, W)`` — subsequent calls at the same resolution pay no Python overhead.

Batch / offline helpers
-----------------------
``run_dvs_count_scan`` / ``run_dvs_dense_scan`` remain available for offline
pipelines. They are now thin wrappers that call ``dvs_step`` inside
``lax.scan`` while preserving the streaming init contract: ``frame0`` only
initializes state and emits no events.

Event model improvements over v1
---------------------------------
- Multi-event threshold crossings per step (``max_events_per_step``).
- Refractory masking is per-pixel with float timestamp tracking.
- Explicit reference adaptation supports recovery in long-running scenes.
- Reference ``ref_log`` advances by the quantised threshold multiple, not
  snapped to ``log_filt``, so threshold FPN is preserved across firings.
- Noise key is folded from a monotone counter, safe for non-sequential use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class SparseEvents(NamedTuple):
    """Packed sparse events."""

    t: jnp.ndarray
    y: jnp.ndarray
    x: jnp.ndarray
    p: jnp.ndarray
    count: jnp.ndarray


@dataclass(frozen=True)
class DVSParams:
    """Log-domain DVS-style sensor parameters."""

    pos_thres: float = 0.15
    neg_thres: float = 0.15
    sigma_thres: float = 0.03
    cutoff_hz: float = 15.0
    refractory_s: float = 0.0005
    shot_noise_std: float = 0.0
    adaptation_rate_hz: float = 0.0
    max_events_per_step: int = 4
    min_threshold: float = 1e-4
    log_eps: float = 1e-3


class DVSSensorState(NamedTuple):
    """Streaming carry: all per-pixel state for one sensor instance.

    Fields
    ------
    log_filt : (H, W) float32
        Current low-pass filtered log-luminance.
    ref_log : (H, W) float32
        Per-pixel log reference level (updated on firing).
    last_t : (H, W) float32
        Per-pixel timestamp of last event (seconds), for refractory gating.
    pos_map : (H, W) float32
        Per-pixel positive contrast threshold (fixed after init).
    neg_map : (H, W) float32
        Per-pixel negative contrast threshold (fixed after init).
    step_count : scalar int32
        Monotone counter used as noise key; avoids host-side integer carry.
    """

    log_filt: jnp.ndarray
    ref_log: jnp.ndarray
    last_t: jnp.ndarray
    pos_map: jnp.ndarray
    neg_map: jnp.ndarray
    step_count: jnp.ndarray


# ---------------------------------------------------------------------------
# Threshold map construction
# ---------------------------------------------------------------------------

def build_threshold_maps(
    key: jnp.ndarray,
    shape: tuple[int, int],
    params: DVSParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-pixel log-contrast thresholds: nominal ± Gaussian FPN."""
    h, w = shape
    min_thr = jnp.float32(params.min_threshold)
    key_p, key_n = jr.split(key)
    pos_map = jnp.float32(params.pos_thres) + jnp.float32(params.sigma_thres) * jr.normal(
        key_p, (h, w)
    )
    neg_map = jnp.float32(params.neg_thres) + jnp.float32(params.sigma_thres) * jr.normal(
        key_n, (h, w)
    )
    return jnp.maximum(pos_map, min_thr), jnp.maximum(neg_map, min_thr)


# ---------------------------------------------------------------------------
# Core helpers (JIT-compiled)
# ---------------------------------------------------------------------------

@jax.jit
def _count_threshold_crossings(
    delta: jnp.ndarray,
    pos_map: jnp.ndarray,
    neg_map: jnp.ndarray,
    max_events_per_step: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return dense ON/OFF event counts per pixel for one step."""
    max_e = max_events_per_step.astype(jnp.float32)
    pos_raw = jnp.maximum(delta, 0.0) / pos_map
    neg_raw = jnp.maximum(-delta, 0.0) / neg_map
    pos_k = jnp.minimum(jnp.floor(pos_raw), max_e).astype(jnp.int16)
    neg_k = jnp.minimum(jnp.floor(neg_raw), max_e).astype(jnp.int16)
    return pos_k, neg_k


# ---------------------------------------------------------------------------
# Streaming API  (primary interface)
# ---------------------------------------------------------------------------

def dvs_init(
    frame0: jnp.ndarray,
    t0: float | jnp.ndarray,
    pos_map: jnp.ndarray,
    neg_map: jnp.ndarray,
    params: DVSParams,
) -> DVSSensorState:
    """Initialise sensor state from the first frame.

    Parameters
    ----------
    frame0 : (H, W) float32
        First luma frame (e.g. [0, 255]).
    t0 : scalar
        Timestamp for frame0 in seconds.
    pos_map, neg_map : (H, W) float32
        Per-pixel threshold maps from :func:`build_threshold_maps`.
    params : DVSParams
        Sensor parameters (only ``log_eps`` used here).
    """
    log0 = jnp.log(jnp.maximum(frame0, jnp.float32(params.log_eps)))
    h, w = frame0.shape
    return DVSSensorState(
        log_filt=log0,
        ref_log=log0,
        last_t=jnp.full((h, w), jnp.float32(-1e9)),
        pos_map=pos_map,
        neg_map=neg_map,
        step_count=jnp.zeros((), dtype=jnp.int32),
    )


@jax.jit
def dvs_step(
    state: DVSSensorState,
    frame: jnp.ndarray,
    t_now: jnp.ndarray,
    key: jnp.ndarray,
    cutoff_hz: jnp.ndarray,
    refractory_s: jnp.ndarray,
    shot_noise_std: jnp.ndarray,
    adaptation_rate_hz: jnp.ndarray,
    max_events_per_step: jnp.ndarray,
    log_eps: jnp.ndarray,
    dt: jnp.ndarray,
) -> tuple[DVSSensorState, jnp.ndarray, jnp.ndarray]:
    """Simulate one frame step. Returns ``(new_state, on_k, off_k)``.

    Parameters
    ----------
    state : DVSSensorState
        Current sensor state (from :func:`dvs_init` or previous step).
    frame : (H, W) float32
        Current luma frame.
    t_now : scalar float32
        Timestamp of ``frame`` in seconds.
    key : PRNGKey
        Base key; per-step noise keys are derived via ``fold_in(key, step_count)``.
    cutoff_hz, refractory_s, shot_noise_std, adaptation_rate_hz : scalar float32
        Physical sensor parameters as 0-dim JAX arrays (so this JITs cleanly
        without retracing on value changes — pass the same scalar shapes always).
    max_events_per_step : scalar int32
    log_eps : scalar float32
    dt : scalar float32
        Time delta since previous frame (seconds). Caller computes this to
        avoid carrying ``t_prev`` inside state and to support non-uniform fps.

    Notes
    -----
    All scalar params are 0-dim arrays, not Python scalars, so ``@jax.jit``
    traces once per ``(H, W)`` resolution and never retraces on param changes.
    Wrap Python scalars with ``jnp.float32(x)`` before calling.
    """
    log_filt_prev, ref_log, last_t, pos_map, neg_map, step_count = state

    log_in = jnp.log(jnp.maximum(frame, log_eps))

    tau = 1.0 / (2.0 * jnp.pi * cutoff_hz)
    alpha = jnp.exp(-dt / jnp.maximum(tau, jnp.float32(1e-9)))
    log_filt = alpha * log_filt_prev + (1.0 - alpha) * log_in

    key_step = jr.fold_in(key, step_count)
    key_noise = key_step

    h, w = frame.shape
    noise = jr.normal(key_noise, (h, w)) * shot_noise_std
    adapt_alpha = 1.0 - jnp.exp(-dt * jnp.maximum(adaptation_rate_hz, jnp.float32(0.0)))
    ref_adapted = ref_log + adapt_alpha * (log_filt - ref_log)
    delta = (log_filt + noise) - ref_adapted

    on_k, off_k = _count_threshold_crossings(delta, pos_map, neg_map, max_events_per_step)

    refr_ok = (t_now - last_t) >= refractory_s
    on_k = jnp.where(refr_ok, on_k, jnp.zeros_like(on_k))
    off_k = jnp.where(refr_ok, off_k, jnp.zeros_like(off_k))

    fired = (on_k > 0) | (off_k > 0)
    ref_step = (
        ref_adapted
        + on_k.astype(jnp.float32) * pos_map
        - off_k.astype(jnp.float32) * neg_map
    )
    new_ref = jnp.where(fired, ref_step, ref_adapted)
    new_last_t = jnp.where(fired, t_now, last_t)

    new_state = DVSSensorState(
        log_filt=log_filt,
        ref_log=new_ref,
        last_t=new_last_t,
        pos_map=pos_map,
        neg_map=neg_map,
        step_count=step_count + 1,
    )
    return new_state, on_k, off_k


def make_dvs_step_fn(params: DVSParams, key: jnp.ndarray):
    """Return a convenience wrapper around :func:`dvs_step` with params bound.

    Usage::

        step_fn = make_dvs_step_fn(params, key)
        state = dvs_init(frame0, t0, pos_map, neg_map, params)
        t_prev = t0
        for frame, t in source:
            state, on_k, off_k = step_fn(state, frame, t, t - t_prev)
            t_prev = t

    The returned ``step_fn`` is already JIT-compiled (it calls :func:`dvs_step`
    which is ``@jax.jit``). No extra ``jax.jit`` wrapper needed.
    """
    cutoff_hz = jnp.float32(params.cutoff_hz)
    refractory_s = jnp.float32(params.refractory_s)
    shot_noise_std = jnp.float32(params.shot_noise_std)
    adaptation_rate_hz = jnp.float32(params.adaptation_rate_hz)
    max_events_per_step = jnp.int32(params.max_events_per_step)
    log_eps = jnp.float32(params.log_eps)

    def step_fn(
        state: DVSSensorState,
        frame: jnp.ndarray,
        t_now: jnp.ndarray,
        dt: jnp.ndarray,
    ) -> tuple[DVSSensorState, jnp.ndarray, jnp.ndarray]:
        return dvs_step(
            state, frame, t_now, key,
            cutoff_hz, refractory_s, shot_noise_std, adaptation_rate_hz,
            max_events_per_step, log_eps, dt,
        )

    return step_fn


# ---------------------------------------------------------------------------
# Offline / batch helpers  (lax.scan wrappers — bit-identical to streaming)
# ---------------------------------------------------------------------------

@jax.jit
def run_dvs_count_scan_jit(
    frames: jnp.ndarray,
    timestamps: jnp.ndarray,
    pos_map: jnp.ndarray,
    neg_map: jnp.ndarray,
    key: jnp.ndarray,
    cutoff_hz: jnp.ndarray,
    refractory_s: jnp.ndarray,
    shot_noise_std: jnp.ndarray,
    adaptation_rate_hz: jnp.ndarray,
    max_events_per_step: jnp.ndarray,
    log_eps: jnp.ndarray,
) -> tuple[DVSSensorState, jnp.ndarray, jnp.ndarray]:
    """Offline count scan with streaming semantics.

    ``frames[0]`` / ``timestamps[0]`` initialize the sensor state and emit no
    events, matching :func:`dvs_init` + repeated :func:`dvs_step` in streaming
    mode. Returned count tensors therefore always have a leading all-zero slice
    at index 0 for alignment with the input frame stack.
    """
    init_state = DVSSensorState(
        log_filt=jnp.log(jnp.maximum(frames[0], log_eps)),
        ref_log=jnp.log(jnp.maximum(frames[0], log_eps)),
        last_t=jnp.full(frames.shape[1:], jnp.float32(-1e9)),
        pos_map=pos_map,
        neg_map=neg_map,
        step_count=jnp.zeros((), dtype=jnp.int32),
    )

    def scan_step(state, scan_inputs):
        frame, t_now, dt = scan_inputs
        new_state, on_k, off_k = dvs_step(
            state, frame, t_now, key,
            cutoff_hz, refractory_s, shot_noise_std, adaptation_rate_hz,
            max_events_per_step, log_eps, dt,
        )
        return new_state, (on_k, off_k)

    scan_inputs = (
        frames[1:],
        timestamps[1:],
        timestamps[1:] - timestamps[:-1],
    )
    final_state, (on_tail, off_tail) = lax.scan(scan_step, init_state, scan_inputs)
    zero_counts = jnp.zeros((1, *frames.shape[1:]), dtype=jnp.int16)
    on_counts = jnp.concatenate([zero_counts, on_tail], axis=0)
    off_counts = jnp.concatenate([zero_counts, off_tail], axis=0)
    return final_state, on_counts, off_counts


@jax.jit
def run_dvs_dense_scan_jit(
    frames: jnp.ndarray,
    timestamps: jnp.ndarray,
    pos_map: jnp.ndarray,
    neg_map: jnp.ndarray,
    key: jnp.ndarray,
    cutoff_hz: jnp.ndarray,
    refractory_s: jnp.ndarray,
    shot_noise_std: jnp.ndarray,
    adaptation_rate_hz: jnp.ndarray,
    max_events_per_step: jnp.ndarray,
    log_eps: jnp.ndarray,
) -> tuple[DVSSensorState, jnp.ndarray, jnp.ndarray]:
    """Legacy boolean-mask scan. Returns ``(final_state, on_masks, off_masks)``."""
    final_state, on_counts, off_counts = run_dvs_count_scan_jit(
        frames, timestamps, pos_map, neg_map, key,
        cutoff_hz, refractory_s, shot_noise_std, adaptation_rate_hz,
        max_events_per_step, log_eps,
    )
    return final_state, on_counts > 0, off_counts > 0


def run_dvs_count_scan(
    frames: jnp.ndarray,
    timestamps: jnp.ndarray,
    pos_map: jnp.ndarray,
    neg_map: jnp.ndarray,
    key: jnp.ndarray,
    params: DVSParams,
) -> tuple[DVSSensorState, jnp.ndarray, jnp.ndarray]:
    """Non-JIT wrapper: returns integer ON/OFF count stacks ``(T,H,W)``."""
    return run_dvs_count_scan_jit(
        frames, timestamps, pos_map, neg_map, key,
        jnp.float32(params.cutoff_hz),
        jnp.float32(params.refractory_s),
        jnp.float32(params.shot_noise_std),
        jnp.float32(params.adaptation_rate_hz),
        jnp.int32(params.max_events_per_step),
        jnp.float32(params.log_eps),
    )


def run_dvs_dense_scan(
    frames: jnp.ndarray,
    timestamps: jnp.ndarray,
    pos_map: jnp.ndarray,
    neg_map: jnp.ndarray,
    key: jnp.ndarray,
    params: DVSParams,
) -> tuple[DVSSensorState, jnp.ndarray, jnp.ndarray]:
    """Non-JIT wrapper: returns boolean ON/OFF mask stacks ``(T,H,W)``."""
    return run_dvs_dense_scan_jit(
        frames, timestamps, pos_map, neg_map, key,
        jnp.float32(params.cutoff_hz),
        jnp.float32(params.refractory_s),
        jnp.float32(params.shot_noise_std),
        jnp.float32(params.adaptation_rate_hz),
        jnp.int32(params.max_events_per_step),
        jnp.float32(params.log_eps),
    )


# ---------------------------------------------------------------------------
# Sparse event packing  (post-processing, host-side)
# ---------------------------------------------------------------------------

def dense_counts_to_sparse_events(
    on_counts: jnp.ndarray,
    off_counts: jnp.ndarray,
    timestamps: jnp.ndarray,
    *,
    max_events: int | None = None,
) -> SparseEvents:
    """Pack dense count tensors into sparse event arrays.

    Notes
    -----
    - One sparse entry per event count (repeats pixel coords for count > 1).
    - Events within a sub-frame share the sub-frame timestamp; if you fed
      sub-frames through the streaming API each gets its own ``t``, giving
      you fine-grained intra-frame timing for free.
    - Truncation via ``max_events`` is order-of-appearance, not time-sorted.
    - This runs on the host (uses ``jnp.where`` which forces materialisation).
      Keep it as post-processing, not inside a JIT-traced loop.
    """
    if on_counts.shape != off_counts.shape:
        raise ValueError("on_counts and off_counts must have the same shape")
    if on_counts.ndim != 3:
        raise ValueError("expected (T,H,W) count tensors")
    if timestamps.shape[0] != on_counts.shape[0]:
        raise ValueError("timestamps length must match T")

    t_on, y_on, x_on = jnp.where(on_counts > 0)
    t_off, y_off, x_off = jnp.where(off_counts > 0)

    c_on = on_counts[t_on, y_on, x_on].astype(jnp.int32)
    c_off = off_counts[t_off, y_off, x_off].astype(jnp.int32)

    t_idx = jnp.concatenate([jnp.repeat(t_on, c_on), jnp.repeat(t_off, c_off)])
    y = jnp.concatenate([jnp.repeat(y_on, c_on), jnp.repeat(y_off, c_off)])
    x = jnp.concatenate([jnp.repeat(x_on, c_on), jnp.repeat(x_off, c_off)])
    p = jnp.concatenate([
        jnp.ones((int(jnp.sum(c_on)),), dtype=jnp.int8),
        -jnp.ones((int(jnp.sum(c_off)),), dtype=jnp.int8),
    ])
    count = jnp.ones_like(p, dtype=jnp.int16)
    t = timestamps[t_idx]

    if max_events is not None:
        t = t[:max_events]
        y = y[:max_events]
        x = x[:max_events]
        p = p[:max_events]
        count = count[:max_events]

    return SparseEvents(t=t, y=y, x=x, p=p, count=count)
