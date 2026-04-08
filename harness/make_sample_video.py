#!/usr/bin/env python3
"""Write a short synthetic MP4 (moving bright square) for testing ``run_clip.py --video``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        raise SystemExit(f"need opencv and numpy: {e}") from e

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("/tmp/v2e_jax_sample.mp4"))
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    args = p.parse_args(argv)

    w, h = args.width, args.height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(args.out), fourcc, args.fps, (w, h), isColor=True)

    for t in range(args.frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (40, 40, 45)
        x = int((t / max(args.frames - 1, 1)) * (w - 80)) + 20
        y = h // 2 - 25
        cv2.rectangle(frame, (x, y), (x + 50, y + 50), (220, 200, 80), -1)
        vw.write(frame)

    vw.release()
    print(f"Wrote {args.out.resolve()} ({args.frames} frames @ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
