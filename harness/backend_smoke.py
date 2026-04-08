#!/usr/bin/env python3
"""Minimal backend smoke test for Linux CPU / NVIDIA / ROCm JAX installs."""

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

import jax
import jax.numpy as jnp
import jax.random as jr

from v2e_jax import DVSParams, build_threshold_maps, dvs_init, make_dvs_step_fn, run_dvs_count_scan


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--height", type=int, default=32)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    h, w = args.height, args.width
    t_count = max(2, int(args.frames))
    key = jr.PRNGKey(args.seed)
    key_frames, key_thr, key_step = jr.split(key, 3)

    frames = jr.uniform(key_frames, (t_count, h, w), minval=16.0, maxval=224.0, dtype=jnp.float32)
    timestamps = jnp.linspace(0.0, 0.030 * jnp.float32(t_count - 1), t_count, dtype=jnp.float32)
    params = DVSParams(pos_thres=0.15, neg_thres=0.15, cutoff_hz=20.0, max_events_per_step=2)
    pos_map, neg_map = build_threshold_maps(key_thr, (h, w), params)

    state = dvs_init(frames[0], timestamps[0], pos_map, neg_map, params)
    step_fn = make_dvs_step_fn(params, key_step)
    stream_on = []
    stream_off = []
    t_prev = timestamps[0]
    for i in range(1, t_count):
        state, on_k, off_k = step_fn(state, frames[i], timestamps[i], timestamps[i] - t_prev)
        stream_on.append(on_k)
        stream_off.append(off_k)
        t_prev = timestamps[i]

    stream_on_sum = int(jnp.stack(stream_on).sum()) if stream_on else 0
    stream_off_sum = int(jnp.stack(stream_off).sum()) if stream_off else 0

    _, scan_on, scan_off = run_dvs_count_scan(frames, timestamps, pos_map, neg_map, key_step, params)
    jax.block_until_ready(scan_on)
    jax.block_until_ready(scan_off)

    result = {
        "default_backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "frame_shape": [t_count, h, w],
        "stream_on_sum": stream_on_sum,
        "stream_off_sum": stream_off_sum,
        "scan_on_sum": int(scan_on.sum()),
        "scan_off_sum": int(scan_off.sum()),
        "stream_scan_match": bool(jnp.array_equal(jnp.stack([jnp.zeros((h, w), dtype=jnp.int16), *stream_on]), scan_on))
        and bool(jnp.array_equal(jnp.stack([jnp.zeros((h, w), dtype=jnp.int16), *stream_off]), scan_off)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
