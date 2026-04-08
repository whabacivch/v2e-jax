# v2e-jax

A JAX-native DVS emulator inspired by [SensorsINI/v2e](https://github.com/SensorsINI/v2e), redesigned for streaming execution, GPU acceleration, and simpler integration into modern JAX pipelines.

## Project status

`v2e-jax` should be read as a **standalone reinterpretation**, not a drop-in replacement for upstream `v2e`.

It keeps the same broad problem setting and several core ideas from `v2e`:

1. log-domain event generation from frame intensity changes,
2. per-pixel threshold variation,
3. low-pass photoreceptor state,
4. ON/OFF event emission from threshold crossings.

But it also makes deliberate architectural and modeling changes in service of streaming and hardware acceleration. If you need the most faithful implementation of published `v2e` behavior, use upstream. If you want a compact JAX-first implementation that is easy to run online and extend, this project is aimed at that use case.

## Why this exists

The original `v2e` is a strong reference implementation, but it is also tightly coupled to a PyTorch-era workflow: larger Python control flow, heavier interpolation dependencies, and less natural support for JAX/XLA-style streaming.

This project exists to explore a different tradeoff:

1. **Streaming-first**: process one frame at a time with a small explicit state object and minimal host overhead after JIT warmup.
2. **JAX-native execution**: make the hot path easy to run on GPU/accelerator hardware through JAX/XLA.
3. **Cleaner temporal upsampling**: replace the SuperSloMo dependency with pure-JAX linear and flow-based interpolation hooks.
4. **Practical long-running behavior**: support explicit recovery dynamics for persistent high-contrast or saturated streaming scenes rather than assuming short offline clips only.

---

## Relationship to upstream v2e

This repo is **based on ideas from** `SensorsINI/v2e`, but it is **not a faithful port** of the upstream codebase.

Important differences:

- The tone-mapping and photoreceptor pipeline are simplified relative to upstream `lin_log` plus intensity-dependent low-pass filtering.
- The emulator exposes a compact functional step API (`dvs_init`, `dvs_step`, `make_dvs_step_fn`) rather than upstream's larger monolithic emulator class.
- Temporal upsampling is deliberately simpler and more composable than the original SuperSloMo path.
- Some state-update behavior, especially reference adaptation, is tuned for streaming practicality rather than strict behavioral parity with upstream `v2e`.

So the right mental model is:

- `v2e`: mature reference simulator with richer legacy behavior,
- `v2e-jax`: smaller JAX-oriented implementation for online and accelerator-backed workflows.

## Current model notes

### Core emulator (`dvs_core.py`)

**Functional streaming API**

The core design is an explicit state transition:

```python
state = dvs_init(frame0, t0, pos_map, neg_map, params)
state, on_k, off_k = dvs_step(...)
```

This makes the emulator easy to run frame-by-frame, wrap in `jax.jit`, and compose into larger streaming systems.

**Threshold-FPN-preserving reference updates**

When a pixel fires multiple times in one step, the reference level advances by the quantised threshold multiple:

```
ref_log += on_k * pos_map - off_k * neg_map
```

This preserves each pixel's individual threshold across multiple firings. A pixel with a high `pos_map` that fires twice advances by `2 * pos_map`, not by a snap to the filtered value.

**Explicit reference adaptation**

The current implementation uses an explicit deterministic adaptation term to relax `ref_log` toward the filtered signal:

```
ref_log <- ref_log + alpha_adapt * (log_filt - ref_log)
```

with

```
alpha_adapt = 1 - exp(-dt * adaptation_rate_hz)
```

This makes the recovery mechanism legible in the model itself: if the internal reference gets pinned far from the current scene statistics, it can recover continuously over time even without waiting for a reset. `adaptation_rate_hz=0` disables this behavior.

That choice is intentional for long-running online use, but it is also one of the main places where this project diverges from upstream `v2e`.

The intended claim here is modest and practical: adaptation is meant to improve
post-saturation recovery quality in streaming use, not to create a dramatic
"works vs does not work" split. In the bundled moving-box whiteout demo, the
current implementation produces cleaner ROI structure after the white flash and
a slightly earlier reacquire metric, while both variants still eventually pick
the target back up.

**Refractory gating**

Tracked per-pixel via a float32 timestamp (`last_t`). The gate is `(t_now - last_t) >= refractory_s` applied as a dense mask before event counts are finalised.

**Noise key management**

Per-step noise keys are derived by folding a monotone step counter into the base key (`jr.fold_in(key, step_count)`). This is deterministic and safe for non-sequential or parallel use without host-side integer bookkeeping.

**Event counts, not booleans**

`dvs_step` returns integer ON/OFF count arrays (`int16`) per pixel per step, not boolean masks. This correctly represents multiple threshold crossings in a single interval. Boolean masks (`on_counts > 0`) are derived from these.

### Upsampling (`upsample.py`)

The original v2e uses SuperSloMo, a pre-trained frame interpolation network. This requires a learned model download and a PyTorch dependency.

This project provides two pure-JAX alternatives:

| Method | Description |
|--------|-------------|
| `upsample_interval_linear` | Linear blend between adjacent frames. No external dependency. |
| `upsample_interval_motion_compensated` | Bidirectional warp using dense optical flow. Plug in RAFT, GMFlow, or any estimator. |

Both are `@jax.jit`-compatible and vectorised with `jax.vmap`. `choose_adaptive_steps` selects step count per interval from a motion score, bounding the retrace count to `max_steps`.

The CLI also supports named subframe timing schedules via `--subframe_schedule`:

- `uniform`: evenly spaced subframes,
- `ease_in`: denser subframes early in the interval,
- `ease_out`: denser subframes late in the interval,
- `ease_in_out`: denser subframes near both ends for smoother transitions.

---

## Layout

| Path | Purpose |
|------|---------|
| `src/v2e_jax/dvs_core.py` | DVS emulator: streaming API + offline `lax.scan` wrappers |
| `src/v2e_jax/upsample.py` | Temporal upsampling: linear, motion-compensated, adaptive |
| `data/loaders.py` | KITTI image folder and video file loaders |
| `helpers/render.py` | Event overlays, side-by-side MP4, summary stats |
| `harness/run_clip.py` | CLI: run emulator on a video file or image folder |
| `harness/run_camera.py` | CLI: real-time emulation from webcam or video (threaded capture) |
| `harness/make_sample_video.py` | Generate a synthetic test video |
| `harness/grad_smoke.py` | Toolchain check: Equinox + `jax.grad` |
| `harness/adaptation_demo.py` | Moving-box whiteout triptych demo + micro-benchmark for adaptation |
| `harness/adaptation_check.py` | Deterministic numerical checks for adaptation and scan/stream parity |

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

The base install is intentionally minimal: JAX + NumPy + Pillow.

Optional extras:

```bash
pip install -e ".[camera]"    # OpenCV-backed camera/video CLIs
pip install -e ".[viz]"       # plots + MP4 rendering
pip install -e ".[training]"  # Equinox grad smoke
pip install -e ".[full]"      # all optional tooling
```

If you prefer requirements files:

```bash
pip install -r requirements.txt
pip install -r requirements-full.txt   # optional full tooling
```

### Linux backend quickstarts

**Linux CPU**

```bash
pip install --upgrade pip
pip install --upgrade jax
pip install -e .
v2e-jax-backend-smoke
```

**Linux NVIDIA GPU**

```bash
pip install --upgrade pip
pip install --upgrade "jax[cuda13]"
pip install -e ".[full]"
v2e-jax-backend-smoke
```

**Linux AMD ROCm GPU**

```bash
pip install --upgrade pip
pip install --upgrade "jax[rocm7-local]"
pip install -e ".[full]"
v2e-jax-backend-smoke
```

The JAX backend command above is the one to trust first:

```bash
python3 -c "import jax; print(jax.devices())"
```

`v2e-jax` itself does not contain CUDA- or ROCm-specific kernels. If JAX can
see the target device, the same emulator code path should run there. In this
repo, Linux CPU is exercised directly in CI-style smoke runs; NVIDIA and ROCm
are documented and packaging-ready, but still need hardware validation on those
actual targets.

---

## Usage

**From a video file:**

```bash
python3 harness/run_clip.py --video clip.mp4 --out /tmp/dvs_out --downscale 2
```

**From a KITTI image folder:**

```bash
python3 harness/run_clip.py --input_dir /path/to/kitti/image_2 --fps 10 --max_frames 120 \
  --steps_per_interval 4 --out /tmp/dvs_out
```

**With a non-uniform subframe schedule:**

```bash
python3 harness/run_clip.py --video clip.mp4 --mode scan --steps_per_interval 4 \
  --subframe_schedule ease_in_out --out /tmp/dvs_out
```

**With explicit recovery adaptation enabled:**

```bash
python3 harness/run_clip.py --video clip.mp4 --steps_per_interval 4 \
  --adaptation_rate_hz 2.0 --out /tmp/dvs_out
```

**Real-time from webcam:**

```bash
python3 harness/run_camera.py --device 0 --fps 60 --steps_per_interval 2
```

If your OpenCV build lacks GUI support, `run_camera.py` will warn and fall back
to `--no-display` instead of crashing.

**Offline batch scan (`lax.scan` wrapper over the same step kernel):**

```bash
python3 harness/run_clip.py --video clip.mp4 --mode scan --out /tmp/dvs_out
```

**Synthetic moving-box whiteout adaptation demo:**

```bash
v2e-jax-adaptation-demo --out /tmp/v2e_jax_adaptation_demo
```

This writes:

- `adaptation_triptych.mp4` — a 3-panel composite with the original luma on the left and ROI-zoomed `adaptation ON` / `adaptation OFF` event views stacked on the right, a labeled ROI rectangle, an expected-edge guide line, reacquire status text, and a render-only event persistence buffer (see `--decay_frames`, default 4) so settling behaviour is legible instead of flickery;
- `adaptation_off/` and `adaptation_on/` — per-case `.npy` stacks, `preview_grid.png`, and `events_summary.txt`;
- `adaptation_comparison.png` with ON/OFF totals and tracked reference-gap plots,
- `metrics.json` with recovery metrics — including a right-ROI reacquire-frame metric (`post_white_right_roi_reacquire_frame_{no,with}_adaptation`, `post_white_right_roi_reacquire_lag_frames_*`, and the headline `post_white_right_roi_reacquire_lag_delta_frames`),
- `benchmark.json` with a small steady-state streaming/scan timing comparison.

The default demo is deliberately framed as a **small enhancement** story:

- the box moves continuously underneath a short full-frame whiteout,
- both variants eventually recover it,
- `adaptation ON` is expected to recover the right-side ROI a bit more cleanly and a bit earlier,
- `adaptation OFF` typically shows more residual clutter and a slightly slower reacquire metric.

On the current default scene, the difference is modest rather than dramatic: the
ROI reacquire metric improves by about one frame, while post-whiteout ROI clutter
and total event load drop more noticeably. That is the level of claim this repo
is intended to support.

**Deterministic adaptation check:**

```bash
v2e-jax-adaptation-check
```

This verifies that, on a fixed bright-hold to indoor-recovery stress case:

- stream and scan still match exactly,
- adaptation reduces repeated ON activity during the saturated bright hold,
- adaptation reduces the tracked `|log_filt - ref_log|` gap near the end of that hold,
- adaptation quiets the OFF tail sooner after returning indoors.

Outputs under `--out`:

- `on_counts.npy`, `off_counts.npy`: dense `(T, H, W)` event count stacks,
- `on_masks.npy`, `off_masks.npy`: boolean activity masks derived from counts,
- `on_totals.npy`, `off_totals.npy`, `total_events.npy`: per-frame totals,
- `events_summary.txt`, `preview_grid.png`, `side_by_side.mp4`.

---

## Streaming API

```python
import jax.numpy as jnp
import jax.random as jr
from v2e_jax import DVSParams, build_threshold_maps, dvs_init, make_dvs_step_fn

params = DVSParams(pos_thres=0.15, neg_thres=0.15, cutoff_hz=15.0)
key = jr.PRNGKey(0)
key_t, key_r = jr.split(key)

pos_map, neg_map = build_threshold_maps(key_t, (H, W), params)
state = dvs_init(frame0, t0, pos_map, neg_map, params)
step_fn = make_dvs_step_fn(params, key_r)  # JIT-compiled, params bound

t_prev = t0
for frame, t in source:
    state, on_k, off_k = step_fn(state, frame, t, t - t_prev)
    t_prev = t
    # on_k, off_k: (H, W) int16 event count arrays
```

`dvs_step` traces once per `(H, W)`. All subsequent calls at the same resolution reuse the compiled kernel.

---

## Limitations

This repo is still experimental. A few things are intentionally unfinished or simplified:

- it is not yet a feature-complete replacement for upstream `v2e`,
- it does not currently aim for exact event-by-event parity with upstream,
- some model choices, including reference adaptation, are optimized for streaming practicality rather than historical fidelity,
- the adaptation effect demonstrated here is currently best read as a small streaming-oriented recovery enhancement, not a dramatic failure-vs-success boundary,
- benchmarking is still lightweight compared with a polished standalone release, though a synthetic adaptation stress demo and check are now included.

---

## Credits

This project was directly inspired by [SensorsINI/v2e](https://github.com/SensorsINI/v2e) by Tobi Delbruck, Yuhuang Hu, and contributors. Original paper:

> Hu, Y., Liu, S., & Delbruck, T. (2021). v2e: From Video Frames to Realistic DVS Events. CVPR Workshop on Event-based Vision.
