#!/usr/bin/env python3
"""Optional: Equinox module + scalar loss + ``jax.grad`` (toolchain smoke; DVS events are discrete)."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp


class ScaleModel(eqx.Module):
    """Trivial learnable scalar; loss couples to frame sum so gradients are non-trivial."""

    w: jax.Array


def loss_fn(model: ScaleModel, x: jnp.ndarray) -> jnp.ndarray:
    return model.w * jnp.sum(x * x)


def main() -> None:
    model = ScaleModel(jnp.array(0.5, dtype=jnp.float32))
    x = jnp.ones((4, 8, 8), dtype=jnp.float32)
    loss = loss_fn(model, x)
    grads = eqx.filter_grad(loss_fn)(model, x)
    w_grad = grads.w
    out_path = Path(__file__).resolve().parent / "grad_smoke.txt"
    out_path.write_text(
        f"loss={float(loss)}\n"
        f"w_grad={float(jnp.asarray(w_grad))}\n"
        f"devices={jax.devices()}\n"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
