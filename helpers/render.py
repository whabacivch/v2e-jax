"""Visualize dense ON/OFF masks: counts, RGB overlays, optional PNG grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def counts_per_timestep(
    on_stack: np.ndarray,
    off_stack: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-frame ON counts, OFF counts, total."""
    on_c = on_stack.sum(axis=(1, 2)).astype(np.int64)
    off_c = off_stack.sum(axis=(1, 2)).astype(np.int64)
    return on_c, off_c, on_c + off_c


def overlay_events_rgb(
    on_frame: np.ndarray,
    off_frame: np.ndarray,
) -> np.ndarray:
    """``(H,W)`` bool masks -> ``(H,W,3)`` float in [0,1]. Green=ON, red=OFF."""
    h, w = on_frame.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)
    o = on_frame.astype(bool)
    f = off_frame.astype(bool)
    rgb[o, 0] = 0.15
    rgb[o, 1] = 0.95
    rgb[o, 2] = 0.15
    rgb[f, 0] = 0.95
    rgb[f, 1] = 0.15
    rgb[f, 2] = 0.15
    both = o & f
    rgb[both, :] = (0.9, 0.9, 0.2)
    return rgb


def save_preview_grid(
    on_stack: np.ndarray,
    off_stack: np.ndarray,
    out_path: Path,
    *,
    stride: int = 1,
    max_panels: int = 12,
) -> None:
    """Save a matplotlib grid of event overlays (first ``max_panels`` frames at ``stride``)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = on_stack.shape[0]
    idxs = list(range(0, t, stride))[:max_panels]
    n = len(idxs)
    if n == 0:
        return
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, idxs):
        rgb = overlay_events_rgb(on_stack[i], off_stack[i])
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"t={i}")
        ax.axis("off")
    for j in range(len(idxs), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def infer_playback_fps(timestamps: np.ndarray) -> float:
    """Median spacing ``dt`` → ``1/dt`` for encoding the simulated timeline."""
    if timestamps.size < 2:
        return 30.0
    dts = np.diff(timestamps.astype(np.float64))
    dt = float(np.median(dts)) if dts.size else 1.0 / 30.0
    return 1.0 / max(dt, 1e-9)


def luma_to_bgr_u8(luma: np.ndarray) -> np.ndarray:
    """``(H,W)`` float luma → ``(H,W,3)`` uint8 BGR."""
    try:
        import cv2
    except ImportError as e:
        raise ImportError("side-by-side video requires opencv: pip install opencv-python-headless") from e
    g = np.clip(luma, 0.0, 255.0).astype(np.uint8)
    return cv2.merge([g, g, g])


def overlay_to_bgr_u8(on_frame: np.ndarray, off_frame: np.ndarray) -> np.ndarray:
    """Event overlay as BGR uint8 for OpenCV."""
    try:
        import cv2
    except ImportError as e:
        raise ImportError("side-by-side video requires opencv: pip install opencv-python-headless") from e
    rgb = overlay_events_rgb(on_frame, off_frame)
    u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)


def write_side_by_side_mp4(
    frames_luma: np.ndarray,
    on_stack: np.ndarray,
    off_stack: np.ndarray,
    out_path: Path,
    *,
    fps: float,
    divider_px: int = 2,
    display_scale: int = 1,
) -> None:
    """Encode ``[ original luma | event RGB ]`` per timestep (same height, concatenated width).

    ``frames_luma`` shape ``(T,H,W)``; ``on_stack`` / ``off_stack`` ``(T,H,W)`` bool or 0/1.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError("side-by-side video requires opencv: pip install opencv-python-headless") from e

    t_count = int(frames_luma.shape[0])
    if t_count == 0:
        return
    h, w = int(frames_luma.shape[1]), int(frames_luma.shape[2])
    div = max(0, int(divider_px))
    scale = max(1, int(display_scale))
    h_out = h * scale
    w_out = w * scale
    div_out = div * scale
    w_full = w_out * 2 + div_out
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), fourcc, float(max(fps, 1e-3)), (w_full, h_out))
    if not vw.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {out_path}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = min(h, w) / 480.0 * 0.55
    th = max(1, int(round(fs * 2)))
    for ti in range(t_count):
        left = luma_to_bgr_u8(np.asarray(frames_luma[ti]))
        right = overlay_to_bgr_u8(on_stack[ti], off_stack[ti])
        if scale > 1:
            left = cv2.resize(left, (w_out, h_out), interpolation=cv2.INTER_NEAREST)
            right = cv2.resize(right, (w_out, h_out), interpolation=cv2.INTER_NEAREST)
        cv2.putText(left, "Original (luma)", (8, 24), font, fs, (40, 220, 255), th, cv2.LINE_AA)
        cv2.putText(right, "Events (green=ON red=OFF)", (8, 24), font, fs, (40, 40, 240), th, cv2.LINE_AA)
        if div_out > 0:
            bar = np.zeros((h_out, div_out, 3), dtype=np.uint8)
            frame = np.concatenate([left, bar, right], axis=1)
        else:
            frame = np.concatenate([left, right], axis=1)
        vw.write(frame)
    vw.release()


def write_summary(
    path: Path,
    on_c: np.ndarray,
    off_c: np.ndarray,
    meta: dict[str, str],
) -> None:
    lines = [f"{k}: {v}" for k, v in sorted(meta.items())]
    lines.append("")
    lines.append("frame_idx on_count off_count total")
    for i in range(len(on_c)):
        lines.append(f"{i} {int(on_c[i])} {int(off_c[i])} {int(on_c[i] + off_c[i])}")
    path.write_text("\n".join(lines) + "\n")


def decayed_event_rgb(
    on_window: np.ndarray,
    off_window: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Composite the last ``N`` frames of ON/OFF counts into one ``(H,W,3)`` RGB.

    ``on_window`` / ``off_window`` are ``(N, H, W)``, **newest frame at index 0**.
    ``weights`` is ``(N,)`` in the same order (typically exponential decay,
    already normalised to peak 1.0). Newest events are drawn at full opacity
    on a white canvas; older events blend toward white according to their
    weight so settling is legible but noise fades out.

    Colours match :func:`overlay_events_rgb`: green=ON, red=OFF, yellow-ish
    mix when both polarities are active in the same pixel on the newest frame.
    """
    if on_window.shape != off_window.shape:
        raise ValueError("on_window and off_window must have the same shape")
    if on_window.ndim != 3:
        raise ValueError("expected (N,H,W) windows")
    n, h, w = on_window.shape
    if weights.shape != (n,):
        raise ValueError("weights must match the window length")

    rgb = np.ones((h, w, 3), dtype=np.float32)
    on_color = np.array([0.15, 0.95, 0.15], dtype=np.float32)
    off_color = np.array([0.95, 0.15, 0.15], dtype=np.float32)

    # Iterate oldest -> newest so the newest events overwrite faded ones.
    for i in range(n - 1, -1, -1):
        alpha = float(np.clip(weights[i], 0.0, 1.0))
        if alpha <= 0.0:
            continue
        on_mask = (on_window[i] > 0).astype(np.float32) * alpha
        off_mask = (off_window[i] > 0).astype(np.float32) * alpha
        # Blend ON
        on_a = on_mask[..., None]
        rgb = rgb * (1.0 - on_a) + on_color[None, None, :] * on_a
        # Blend OFF on top
        off_a = off_mask[..., None]
        rgb = rgb * (1.0 - off_a) + off_color[None, None, :] * off_a

    return np.clip(rgb, 0.0, 1.0)


def _decayed_rgb_to_bgr_u8(rgb: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError as e:
        raise ImportError("triptych requires opencv: pip install opencv-python-headless") from e
    u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)


def write_triptych_mp4(
    frames_luma: np.ndarray,
    on_counts_on: np.ndarray,
    off_counts_on: np.ndarray,
    on_counts_off: np.ndarray,
    off_counts_off: np.ndarray,
    out_path: Path,
    *,
    fps: float,
    roi_xyxy: tuple[int, int, int, int],
    edge_column_lookup: np.ndarray | None = None,
    whiteout_start_frame: int,
    whiteout_stop_frame: int,
    reacquired_on_frame: int | None = None,
    reacquired_off_frame: int | None = None,
    decay_frames: int = 4,
    display_scale: int = 1,
    divider_px: int = 2,
) -> None:
    """Write a 3-panel comparison video for the adaptation demo.

    Layout per timestep, with each ``(H, W)`` pane upscaled by ``display_scale``:

        +-----------------+-----------------+
        |                 |  adaptation ON  |
        |    Original     +-----------------+
        |     (luma)      |  adaptation OFF |
        +-----------------+-----------------+

    The left pane shows the full original frame with a green ROI rectangle.
    The right panes show a zoomed view of that ROI for adaptation ON/OFF, plus
    a yellow guide line at the expected box-edge column when available.
    The right panes also show per-frame ROI event counts and use a
    ``decay_frames`` rolling buffer so settling behaviour is legible.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError("triptych requires opencv: pip install opencv-python-headless") from e

    if frames_luma.ndim != 3:
        raise ValueError("frames_luma must be (T,H,W)")
    t_count, h, w = int(frames_luma.shape[0]), int(frames_luma.shape[1]), int(frames_luma.shape[2])
    for name, arr in (
        ("on_counts_on", on_counts_on),
        ("off_counts_on", off_counts_on),
        ("on_counts_off", on_counts_off),
        ("off_counts_off", off_counts_off),
    ):
        if arr.shape != (t_count, h, w):
            raise ValueError(f"{name} shape {arr.shape} does not match frames_luma {frames_luma.shape}")
    if t_count == 0:
        return

    scale = max(1, int(display_scale))
    div = max(0, int(divider_px))
    decay_n = max(1, int(decay_frames))

    # Exponential decay weights (newest-first), normalised so newest weight == 1.
    decay_base = 0.55
    weights = np.asarray([decay_base ** i for i in range(decay_n)], dtype=np.float32)

    # Output sizing: left column is full height 2*h, right column is two h stacks.
    pane_h = h * scale
    pane_w = w * scale
    right_col_h = pane_h * 2 + div * scale  # internal horizontal divider
    out_h = right_col_h
    out_w = pane_w + div * scale + pane_w  # left + vertical divider + right
    # Left column is stretched to full right_col_h so the two columns line up.

    y0, y1, x0, x1 = roi_xyxy
    roi_h = max(1, int(y1 - y0))
    roi_w = max(1, int(x1 - x0))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), fourcc, float(max(fps, 1e-3)), (out_w, out_h))
    if not vw.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {out_path}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = min(pane_h, pane_w) / 480.0 * 0.55
    th = max(1, int(round(fs * 2)))
    roi_color_bgr = (60, 220, 60)
    label_color = (40, 220, 255)
    sub_color = (200, 200, 200)

    def _scale_and_label(
        bgr: np.ndarray,
        label: str,
        sub: str | None,
        *,
        edge_local_x: int | None = None,
        reacquired_frame: int | None = None,
        ti: int,
    ) -> np.ndarray:
        bgr = cv2.resize(bgr, (pane_w, pane_h), interpolation=cv2.INTER_NEAREST)
        if edge_local_x is not None and edge_local_x >= 0:
            guide_x = int(round((edge_local_x + 0.5) * pane_w / float(max(roi_w, 1))))
            guide_x = max(0, min(pane_w - 1, guide_x))
            cv2.line(bgr, (guide_x, 0), (guide_x, pane_h - 1), (40, 210, 230), max(1, scale), cv2.LINE_AA)
        cv2.rectangle(bgr, (0, 0), (pane_w - 1, pane_h - 1), roi_color_bgr, max(1, scale))
        cv2.putText(bgr, label, (8, 22), font, fs, label_color, th, cv2.LINE_AA)
        if sub is not None:
            cv2.putText(bgr, sub, (8, 22 + int(fs * 30)), font, fs * 0.9, sub_color, max(1, th - 0), cv2.LINE_AA)
        if reacquired_frame is None:
            reacq_text = "reacq: n/a"
        elif reacquired_frame < 0:
            reacq_text = "reacq: none"
        elif ti < reacquired_frame:
            reacq_text = f"reacq pending ({reacquired_frame - ti}f)"
        else:
            reacq_text = f"reacquired +{reacquired_frame - int(whiteout_stop_frame)}f"
        cv2.putText(
            bgr,
            reacq_text,
            (8, 22 + int(fs * 58)),
            font,
            fs * 0.8,
            sub_color,
            max(1, th - 0),
            cv2.LINE_AA,
        )
        return bgr

    vdiv = np.zeros((out_h, div * scale, 3), dtype=np.uint8) if div > 0 else None
    hdiv = np.zeros((div * scale, pane_w, 3), dtype=np.uint8) if div > 0 else None

    for ti in range(t_count):
        # --- build decay windows (newest at index 0) ---
        lo = max(0, ti - decay_n + 1)
        idxs = list(range(ti, lo - 1, -1))
        pad = decay_n - len(idxs)

        def _window(counts: np.ndarray) -> np.ndarray:
            w_arr = np.zeros((decay_n, h, w), dtype=counts.dtype)
            for k, src_i in enumerate(idxs):
                w_arr[k] = counts[src_i]
            # padding slots stay zero
            _ = pad  # explicit no-op for readability
            return w_arr

        on_win_on = _window(on_counts_on)
        off_win_on = _window(off_counts_on)
        on_win_off = _window(on_counts_off)
        off_win_off = _window(off_counts_off)

        rgb_on = decayed_event_rgb(on_win_on, off_win_on, weights)
        rgb_off = decayed_event_rgb(on_win_off, off_win_off, weights)

        bgr_on = _decayed_rgb_to_bgr_u8(rgb_on)
        bgr_off = _decayed_rgb_to_bgr_u8(rgb_off)

        # --- left pane: luma, vertically stretched ---
        left_small = luma_to_bgr_u8(np.asarray(frames_luma[ti]))
        if scale > 1:
            left_small = cv2.resize(left_small, (pane_w, pane_h), interpolation=cv2.INTER_NEAREST)
        # Stretch to full right-column height
        left = cv2.resize(left_small, (pane_w, out_h), interpolation=cv2.INTER_NEAREST)

        # Draw ROI on left (on the stretched canvas -> scale y accordingly)
        y_scale = out_h / float(pane_h)
        rx0 = int(x0 * scale)
        rx1 = int(x1 * scale) - 1
        ry0 = int(y0 * scale * y_scale)
        ry1 = int(y1 * scale * y_scale) - 1
        cv2.rectangle(left, (rx0, ry0), (rx1, ry1), roi_color_bgr, max(1, scale))

        # Frame captions
        rel = ti - int(whiteout_stop_frame)
        if int(whiteout_start_frame) <= ti < int(whiteout_stop_frame):
            sub_caption = f"frame {ti}  WHITEOUT"
        elif rel >= 0:
            sub_caption = f"frame {ti}  +{rel}f since whiteout"
        else:
            sub_caption = f"frame {ti}  {rel}f before whiteout"

        cv2.putText(left, "Original (luma)", (8, 26), font, fs, label_color, th, cv2.LINE_AA)
        cv2.putText(left, sub_caption, (8, 26 + int(fs * 32)), font, fs * 0.85, sub_color, max(1, th), cv2.LINE_AA)

        # --- right panes: event overlays ---
        roi_on_events = int(
            (on_counts_on[ti, y0:y1, x0:x1] > 0).sum()
            + (off_counts_on[ti, y0:y1, x0:x1] > 0).sum()
        )
        roi_off_events = int(
            (on_counts_off[ti, y0:y1, x0:x1] > 0).sum()
            + (off_counts_off[ti, y0:y1, x0:x1] > 0).sum()
        )
        edge_local_x = None if edge_column_lookup is None else int(edge_column_lookup[ti])
        bgr_on_roi = bgr_on[y0:y1, x0:x1]
        bgr_off_roi = bgr_off[y0:y1, x0:x1]
        bgr_on = _scale_and_label(
            bgr_on_roi,
            "adaptation ON (ROI)",
            f"ROI events: {roi_on_events}",
            edge_local_x=edge_local_x if edge_local_x >= 0 else None,
            reacquired_frame=reacquired_on_frame,
            ti=ti,
        )
        bgr_off = _scale_and_label(
            bgr_off_roi,
            "adaptation OFF (ROI)",
            f"ROI events: {roi_off_events}",
            edge_local_x=edge_local_x if edge_local_x >= 0 else None,
            reacquired_frame=reacquired_off_frame,
            ti=ti,
        )

        # --- stack right column ---
        if hdiv is not None:
            right_col = np.concatenate([bgr_on, hdiv, bgr_off], axis=0)
        else:
            right_col = np.concatenate([bgr_on, bgr_off], axis=0)

        # Make sure right column matches left column height (rounding safety)
        if right_col.shape[0] != out_h:
            right_col = cv2.resize(right_col, (pane_w, out_h), interpolation=cv2.INTER_NEAREST)

        # --- full frame ---
        if vdiv is not None:
            frame = np.concatenate([left, vdiv, right_col], axis=1)
        else:
            frame = np.concatenate([left, right_col], axis=1)

        vw.write(frame)

    vw.release()
