"""Event-driven image enhancement for streaming pipelines.

Public functions:

  sharpen_luma_with_events   — event-guided unsharp masking: boost edges at
                               ON+OFF active pixels.

  hdr_local_contrast_boost   — event-guided log-Retinex: separates reflectance
                               from illumination via log(luma) - log(blur(luma)),
                               lifting shadows and compressing highlights.
                               Events gate the strength per-pixel.

Both are JIT-compiled and streaming-native (one frame at a time, no state).
FFT convolution primitives are self-contained (no deps beyond JAX).

Ported from dvo_jax/fused_flow.py + event_corners.py, taking only what is
needed here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# FFT convolution primitives
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _fft_conv2d(img: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """2-D SAME convolution via FFT. img and kernel must be 2-D float32."""
    h, w = img.shape
    kh, kw = kernel.shape
    ph = _next_pow2(h + kh - 1)
    pw = _next_pow2(w + kw - 1)
    F_img = jnp.fft.rfft2(img, s=(ph, pw))
    F_ker = jnp.fft.rfft2(kernel, s=(ph, pw))
    out_full = jnp.fft.irfft2(F_img * F_ker, s=(ph, pw))
    pad_top = (kh - 1) // 2
    pad_left = (kw - 1) // 2
    return out_full[pad_top:pad_top + h, pad_left:pad_left + w]


def _gaussian_kernel(radius: int, sigma: float) -> jnp.ndarray:
    """Normalised 2-D Gaussian of size (2*radius+1, 2*radius+1)."""
    k = 2 * radius + 1
    ax = jnp.arange(k, dtype=jnp.float32) - radius
    xx, yy = jnp.meshgrid(ax, ax)
    g = jnp.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return g / jnp.sum(g)


# ---------------------------------------------------------------------------
# Sharpening
# ---------------------------------------------------------------------------

@jax.jit
def sharpen_luma_with_events(
    luma: jnp.ndarray,
    on_k: jnp.ndarray,
    off_k: jnp.ndarray,
    edge_weight: float = 0.4,
    blur_sigma: float = 1.0,
) -> jnp.ndarray:
    """Overlay event edge structure onto grayscale luma.

    Parameters
    ----------
    luma : (H, W) float32 in [0, 1]
    on_k, off_k : (H, W) int16 event count maps from dvs_step
    edge_weight : blend strength (0 = no change, 1 = full enhancement)
    blur_sigma : Gaussian sigma applied to activity before Sobel

    Returns
    -------
    sharpened : (H, W) float32 clipped to [0, 1]
    """
    activity = on_k.astype(jnp.float32) + off_k.astype(jnp.float32)
    act_max = jnp.maximum(jnp.max(activity), 1.0)
    activity_norm = activity / act_max

    g = _gaussian_kernel(2, blur_sigma)
    activity_smooth = _fft_conv2d(activity_norm, g)

    sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=jnp.float32)
    sobel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=jnp.float32)
    Ix = _fft_conv2d(activity_smooth, sobel_x)
    Iy = _fft_conv2d(activity_smooth, sobel_y)
    edge_mag = jnp.sqrt(Ix * Ix + Iy * Iy)
    edge_norm = edge_mag / jnp.maximum(jnp.max(edge_mag), 1e-6)

    # Unsharp-mask style: boost luma at event-active edges
    sharpened = luma + edge_weight * edge_norm * (1.0 - luma)

    # Darken slightly at strong OFF-polarity edges (shadows / trailing edges)
    off_smooth = _fft_conv2d(off_k.astype(jnp.float32) / act_max, g)
    sharpened = sharpened - 0.3 * edge_weight * off_smooth

    return jnp.clip(sharpened, 0.0, 1.0)


# ---------------------------------------------------------------------------
# HDR-like local contrast boost (log-Retinex)
# ---------------------------------------------------------------------------

@jax.jit
def hdr_local_contrast_boost(
    luma: jnp.ndarray,
    on_k: jnp.ndarray,
    off_k: jnp.ndarray,
    blur_radius: int = 32,
    strength: float = 0.85,
) -> jnp.ndarray:
    """Event-guided HDR-like local contrast boost via log-Retinex.

    Computes log(luma) - log(local_blur(luma)) to separate per-pixel
    reflectance from the local illumination field, then maps back to [0, 1]
    via sigmoid.  This lifts shadows and compresses highlights without
    blowing out bright regions.  Event activity gates the blend strength so
    the effect is strongest where the scene is changing.

    Parameters
    ----------
    luma : (H, W) float32 in [0, 1]
    on_k, off_k : (H, W) int16 event count maps from dvs_step
    blur_radius : Gaussian radius for the local illumination estimate —
                  larger = more aggressive shadow lifting (default 32)
    strength : overall blend strength into the Retinex output (default 0.85)

    Returns
    -------
    enhanced : (H, W) float32 in [0, 1]
    """
    eps = jnp.float32(1e-3)

    # Log-Retinex: remove local illumination estimate
    log_luma = jnp.log(luma + eps)
    g_large = _gaussian_kernel(blur_radius, blur_radius * 0.5)
    log_local = _fft_conv2d(log_luma, g_large)
    retinex = log_luma - log_local          # local contrast in log domain

    # Map to [0, 1]: sigmoid centred at 0 (average luminance → 0.5)
    retinex_01 = jax.nn.sigmoid(retinex * 2.5)

    # Event activity gates per-pixel blend strength
    activity = on_k.astype(jnp.float32) + off_k.astype(jnp.float32)
    act_max = jnp.maximum(jnp.max(activity), 1.0)
    g_small = _gaussian_kernel(4, 2.0)
    act_smooth = _fft_conv2d(activity / act_max, g_small)

    # 40% base effect everywhere + 60% extra at event-active pixels
    alpha = strength * jnp.clip(0.4 + 0.6 * act_smooth, 0.0, 1.0)

    enhanced = alpha * retinex_01 + (1.0 - alpha) * luma
    return jnp.clip(enhanced, 0.0, 1.0)
