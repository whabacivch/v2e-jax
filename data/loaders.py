"""Load ordered image sequences (KITTI-style) to float32 luma arrays."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError as e:
    raise ImportError("install pillow: pip install pillow") from e


def _natural_sort_paths(paths: list[Path]) -> list[Path]:
    def key(p: Path) -> tuple[list[int | str], str]:
        s = p.stem
        parts = re.split(r"(\d+)", s)
        k: list[int | str] = []
        for part in parts:
            if part.isdigit():
                k.append(int(part))
            elif part:
                k.append(part)
        return k, p.name

    return sorted(paths, key=key)


def list_image_files(
    input_dir: Path,
    *,
    pattern: str = "*.png",
    extra_globs: tuple[str, ...] = ("*.jpg", "*.jpeg"),
) -> list[Path]:
    """Return sorted image paths under ``input_dir`` (natural numeric order)."""
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {input_dir}")
    files: list[Path] = []
    files.extend(input_dir.glob(pattern))
    for g in extra_globs:
        files.extend(input_dir.glob(g))
    # de-dupe
    uniq = sorted(set(files), key=lambda p: str(p))
    return _natural_sort_paths(uniq)


def rgb_to_luma_u8(rgb: np.ndarray) -> np.ndarray:
    """RGB HWC uint8 or float -> single channel luma float32 in [0, 255]."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected HWC with C>=3, got {rgb.shape}")
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def load_grayscale_hw(path: Path) -> np.ndarray:
    """Load image as 2D float32 luma [0, 255]."""
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        return rgb_to_luma_u8(arr)
    raise ValueError(f"unsupported array shape {arr.shape} for {path}")


def load_sequence(
    input_dir: Path,
    *,
    max_frames: int | None = None,
    downscale: int = 1,
    pattern: str = "*.png",
    fps: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load KITTI-style frame folder to ``(T, H, W)`` float32 and ``timestamps`` (T,) seconds.

    Assumes uniform frame spacing ``1/fps`` (KITTI tracking is ~10 Hz).
    ``downscale`` > 1 uses PIL area resize (integer factor).
    """
    paths = list_image_files(input_dir, pattern=pattern)
    if not paths:
        raise FileNotFoundError(f"no images matched in {input_dir}")
    if max_frames is not None:
        paths = paths[: max(0, max_frames)]

    frames: list[np.ndarray] = []
    for p in paths:
        g = load_grayscale_hw(p)
        if downscale > 1:
            h, w = g.shape
            nh, nw = h // downscale, w // downscale
            pil = Image.fromarray(g.astype(np.uint8), mode="L")
            pil = pil.resize((nw, nh), Image.Resampling.BOX)
            g = np.array(pil, dtype=np.float32)
        frames.append(g)

    t = len(frames)
    if t == 0:
        raise RuntimeError("empty sequence")
    stack = np.stack(frames, axis=0)
    dt = 1.0 / float(fps)
    timestamps = np.arange(t, dtype=np.float64) * dt
    return stack, timestamps


def load_video(
    video_path: Path,
    *,
    max_frames: int | None = None,
    downscale: int = 1,
    fps_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load frames from a video file (OpenCV). Returns ``(T,H,W)`` float32 luma, timestamps, fps_used.

    Luma in [0, 255]. Timestamps use uniform spacing ``1/fps`` from container metadata or
    ``fps_override``.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError("video loading requires opencv: pip install opencv-python-headless") from e

    path = video_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")

    fps_native = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps_override is not None:
        fps = float(fps_override)
    elif fps_native > 1e-3:
        fps = fps_native
    else:
        fps = 30.0

    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        # BGR -> luma (same weights as RGB)
        b = bgr[..., 0].astype(np.float32)
        g = bgr[..., 1].astype(np.float32)
        r = bgr[..., 2].astype(np.float32)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        if downscale > 1:
            h, w = gray.shape
            nh, nw = h // downscale, w // downscale
            gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        frames.append(gray)
        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()
    t = len(frames)
    if t == 0:
        raise RuntimeError(f"no frames decoded from {path}")

    stack = np.stack(frames, axis=0)
    dt = 1.0 / fps
    timestamps = np.arange(t, dtype=np.float64) * dt
    return stack, timestamps, fps
