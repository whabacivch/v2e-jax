"""Direct visual odometry in JAX.

Solves for relative camera pose ξ ∈ se(3) and (optionally) per-pixel inverse
depth from two consecutive frames by minimising photometric error directly.
No intermediate flow field, no feature matching, no host round-trips — pure
GPU pixel math.

Algorithm
---------
Given f0, f1, intrinsics K, initial inverse depth ρ:

    minimize  Σ ρ(p)·||f1(π(K · exp(ξ) · π⁻¹(p, 1/ρ(p)))) - f0(p)||²
              over ξ ∈ se(3)

where π is projection, π⁻¹ is unprojection. ρ(p) is the per-pixel weighting
(here using inverse depth so that pixels with no reliable depth contribute
less and the system stays well-conditioned).

We solve via **coarse-to-fine Gauss-Newton**:
  - Build a Gaussian pyramid for both frames (n_levels)
  - At the coarsest level, initialise ξ = 0, ρ = constant
  - At each level: 5–10 GN iterations on ξ, then upsample ξ to next level
  - Final output: full-resolution ξ and ρ

This is essentially DSO/LSD-SLAM's tracking step, simplified for two frames
at a time and JIT-compiled in JAX.

Ported from jax_dvs_kitti/direct_vo.py — only the import is changed
(SE(3) primitives come from motion_field_vis which is already in v2e_jax).

References
----------
- Engel, Schöps, Cremers (2014), LSD-SLAM
- Engel, Koltun, Cremers (2016), Direct Sparse Odometry (DSO)
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp

# SE(3) primitives live in motion_field_vis in this package
from .motion_field_vis import exp_so3, skew


class VOInfo(NamedTuple):
    """Diagnostics from one ``direct_vo`` call.

    All fields are JAX arrays so the whole struct stays on-device until the
    caller explicitly pulls it.

    Fields
    ------
    residual_rms : scalar float32
        Weighted RMS photometric residual at the finest level after the last
        iteration. Lower is better; spikes flag tracking failure.
    valid_frac : scalar float32
        Fraction of pixels that survived the gradient + in-bounds masks.
    rot_observability : scalar float32
        Smallest eigenvalue of the rotation 3×3 block of H_n.
    trans_observability : scalar float32
        Smallest eigenvalue of the translation 3×3 block of H_n. Low under
        pure-forward motion (FOE degeneracy).
    condition_number : scalar float32
        λ_max / λ_min of the full 6×6 H_n.
    dynamic_residual : (H, W) float32
        Per-pixel |residual|·weight after convergence — high values flag
        independently moving objects.
    """
    residual_rms: jnp.ndarray
    valid_frac: jnp.ndarray
    rot_observability: jnp.ndarray
    trans_observability: jnp.ndarray
    condition_number: jnp.ndarray
    dynamic_residual: jnp.ndarray


# ---------------------------------------------------------------------------
# SE(3) exponential (local, uses primitives from motion_field_vis)
# ---------------------------------------------------------------------------

def exp_se3(xi: jnp.ndarray) -> jnp.ndarray:
    """se(3) exponential map: 6-vector → 4×4 SE(3) matrix.

    xi = [ω; v] where ω is rotation (3,) and v is translation (3,).
    """
    omega = xi[:3]
    v = xi[3:]
    theta = jnp.linalg.norm(omega)
    R = exp_so3(omega)

    omega_safe = omega / jnp.maximum(theta, 1e-8)
    K_skew = skew(omega_safe)
    theta_sq = jnp.maximum(theta * theta, 1e-12)
    theta_cu = jnp.maximum(theta * theta * theta, 1e-12)
    V = (
        jnp.eye(3)
        + ((1.0 - jnp.cos(theta)) / theta_sq) * (theta * K_skew)
        + ((theta - jnp.sin(theta)) / theta_cu) * (theta * theta * (K_skew @ K_skew))
    )
    V = jnp.where(theta < 1e-6, jnp.eye(3), V)
    t = V @ v

    T = jnp.eye(4, dtype=jnp.float32)
    T = T.at[:3, :3].set(R)
    T = T.at[:3, 3].set(t)
    return T


# ---------------------------------------------------------------------------
# Pyramid construction
# ---------------------------------------------------------------------------

def _gaussian_blur_2x(img: jnp.ndarray) -> jnp.ndarray:
    """3×3 Gaussian blur via separable 1-D pass."""
    kernel_1d = jnp.array([1.0, 2.0, 1.0], dtype=jnp.float32) / 4.0
    img_pad = jnp.pad(img, ((0, 0), (1, 1)), mode="edge")
    horiz = (
        kernel_1d[0] * img_pad[:, :-2]
        + kernel_1d[1] * img_pad[:, 1:-1]
        + kernel_1d[2] * img_pad[:, 2:]
    )
    horiz_pad = jnp.pad(horiz, ((1, 1), (0, 0)), mode="edge")
    return (
        kernel_1d[0] * horiz_pad[:-2, :]
        + kernel_1d[1] * horiz_pad[1:-1, :]
        + kernel_1d[2] * horiz_pad[2:, :]
    )


def _downsample_2x(img: jnp.ndarray) -> jnp.ndarray:
    return _gaussian_blur_2x(img)[::2, ::2]


def build_pyramid(img: jnp.ndarray, n_levels: int) -> list[jnp.ndarray]:
    """Gaussian pyramid. Level 0 = original, level n-1 = coarsest."""
    levels = [img]
    for _ in range(n_levels - 1):
        levels.append(_downsample_2x(levels[-1]))
    return levels


def scale_intrinsics(K: jnp.ndarray, scale: float) -> jnp.ndarray:
    return K.at[:2, :].multiply(scale)


# ---------------------------------------------------------------------------
# Bilinear sampling
# ---------------------------------------------------------------------------

def _bilinear_sample(img: jnp.ndarray, vv: jnp.ndarray, uu: jnp.ndarray) -> jnp.ndarray:
    h, w = img.shape
    v0 = jnp.clip(jnp.floor(vv).astype(jnp.int32), 0, h - 1)
    u0 = jnp.clip(jnp.floor(uu).astype(jnp.int32), 0, w - 1)
    v1 = jnp.clip(v0 + 1, 0, h - 1)
    u1 = jnp.clip(u0 + 1, 0, w - 1)
    wv = vv - v0.astype(jnp.float32)
    wu = uu - u0.astype(jnp.float32)
    return (
        (1.0 - wv) * (1.0 - wu) * img[v0, u0]
        + (1.0 - wv) * wu        * img[v0, u1]
        + wv         * (1.0 - wu) * img[v1, u0]
        + wv         * wu          * img[v1, u1]
    )


# ---------------------------------------------------------------------------
# Pixelwise warp + Jacobian
# ---------------------------------------------------------------------------

def _project_and_jacobian(
    pts_3d: jnp.ndarray,
    K: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project (H,W,3) points → (H,W,2) pixels + (H,W,2,6) pose Jacobian."""
    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
    X = pts_3d[..., 0]
    Y = pts_3d[..., 1]
    Z = jnp.maximum(pts_3d[..., 2], 1e-3)
    inv_Z = 1.0 / Z
    inv_Z_sq = inv_Z * inv_Z

    u = fx * X * inv_Z + cx
    v = fy * Y * inv_Z + cy
    pix = jnp.stack([u, v], axis=-1)

    duv_dP = jnp.zeros(pts_3d.shape[:-1] + (2, 3), dtype=jnp.float32)
    duv_dP = duv_dP.at[..., 0, 0].set(fx * inv_Z)
    duv_dP = duv_dP.at[..., 0, 2].set(-fx * X * inv_Z_sq)
    duv_dP = duv_dP.at[..., 1, 1].set(fy * inv_Z)
    duv_dP = duv_dP.at[..., 1, 2].set(-fy * Y * inv_Z_sq)

    zeros = jnp.zeros_like(X)
    P_cross = jnp.stack([
        jnp.stack([zeros, -Z, Y], axis=-1),
        jnp.stack([Z, zeros, -X], axis=-1),
        jnp.stack([-Y, X, zeros], axis=-1),
    ], axis=-2)
    eye3 = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), P_cross.shape)
    dP_dxi = jnp.concatenate([-P_cross, eye3], axis=-1)

    duv_dxi = jnp.einsum("hwij,hwjk->hwik", duv_dP, dP_dxi)
    return pix, duv_dxi


def _gauss_newton_step(
    f0: jnp.ndarray,
    f1: jnp.ndarray,
    inv_depth: jnp.ndarray,
    xi: jnp.ndarray,
    K: jnp.ndarray,
    grad_thresh: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    h, w = f0.shape
    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]

    vv, uu = jnp.meshgrid(
        jnp.arange(h, dtype=jnp.float32),
        jnp.arange(w, dtype=jnp.float32),
        indexing="ij",
    )
    Z = 1.0 / jnp.maximum(inv_depth, 1e-3)
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    pts0 = jnp.stack([X, Y, Z], axis=-1)

    T = exp_se3(xi)
    R = T[:3, :3]; t_vec = T[:3, 3]
    pts1 = pts0 @ R.T + t_vec[None, None, :]

    pix1, duv_dxi = _project_and_jacobian(pts1, K)
    u1 = pix1[..., 0]; v1 = pix1[..., 1]

    f1_warped = _bilinear_sample(f1, v1, u1)
    residual = f1_warped - f0

    f1_du = 0.5 * (
        _bilinear_sample(f1, v1, jnp.clip(u1 + 1, 0, w - 1))
        - _bilinear_sample(f1, v1, jnp.clip(u1 - 1, 0, w - 1))
    )
    f1_dv = 0.5 * (
        _bilinear_sample(f1, jnp.clip(v1 + 1, 0, h - 1), u1)
        - _bilinear_sample(f1, jnp.clip(v1 - 1, 0, h - 1), u1)
    )

    J = (
        f1_du[..., None] * duv_dxi[..., 0, :]
        + f1_dv[..., None] * duv_dxi[..., 1, :]
    )

    grad_mag = jnp.sqrt(f1_du * f1_du + f1_dv * f1_dv + 1e-8)
    valid = (grad_mag > grad_thresh).astype(jnp.float32)
    in_bounds = (
        (u1 >= 0) & (u1 < w - 1) & (v1 >= 0) & (v1 < h - 1) & (Z > 0.0)
    ).astype(jnp.float32)
    weight = valid * in_bounds

    sigma = 10.0
    huber_w = 1.0 / (1.0 + (residual / sigma) ** 2)
    w_total = weight * huber_w

    Jw = J * w_total[..., None]
    H_n = jnp.einsum("hwi,hwj->ij", Jw, J)
    b = -jnp.einsum("hwi,hw->i", Jw, residual)

    damping = 1e-3 * jnp.trace(H_n) / 6.0
    H_damped = H_n + damping * jnp.eye(6, dtype=jnp.float32)
    delta_xi = jnp.linalg.solve(H_damped, b)

    res_norm = jnp.sqrt(jnp.mean(w_total * residual * residual) + 1e-12)
    weight_sum = jnp.sum(w_total)
    return delta_xi, res_norm, H_n, weight_sum, residual, w_total


# ---------------------------------------------------------------------------
# Observability diagnostics
# ---------------------------------------------------------------------------

def _compute_vo_info(
    H_n: jnp.ndarray,
    residual: jnp.ndarray,
    w_total: jnp.ndarray,
    n_pixels: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    residual_rms = jnp.sqrt(
        jnp.sum(w_total * residual * residual) / jnp.maximum(jnp.sum(w_total), 1.0)
    )
    valid_frac = jnp.sum((w_total > 0).astype(jnp.float32)) / jnp.float32(n_pixels)

    rot_block = H_n[:3, :3]
    trans_block = H_n[3:, 3:]
    rot_obs = jnp.min(jnp.linalg.eigvalsh(rot_block + 1e-8 * jnp.eye(3)))
    trans_obs = jnp.min(jnp.linalg.eigvalsh(trans_block + 1e-8 * jnp.eye(3)))

    full_eigs = jnp.linalg.eigvalsh(H_n + 1e-8 * jnp.eye(6))
    condition_number = jnp.max(full_eigs) / jnp.maximum(jnp.min(full_eigs), 1e-12)

    return residual_rms, valid_frac, rot_obs, trans_obs, condition_number


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def direct_vo(
    f0: jnp.ndarray,
    f1: jnp.ndarray,
    n_levels: int,
    n_iters_per_level: int,
    grad_thresh: float,
    K: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, VOInfo]:
    """Estimate relative pose ξ ∈ se(3) between two frames via direct VO.

    Parameters
    ----------
    f0, f1 : (H, W) float32 luma — any consistent scale (e.g. [0, 255])
    n_levels : int (static) — pyramid depth, e.g. 4
    n_iters_per_level : int (static) — GN iterations per level, e.g. 5
    grad_thresh : float (static) — minimum gradient magnitude to include a pixel
    K : (3, 3) float32 — camera intrinsics at full resolution

    Returns
    -------
    xi : (6,) float32 — twist [ωx,ωy,ωz,vx,vy,vz] from f0 → f1
    inv_depth : (H, W) float32 — constant placeholder (future: joint solve)
    info : VOInfo — convergence diagnostics
    """
    pyr0 = build_pyramid(f0, n_levels)
    pyr1 = build_pyramid(f1, n_levels)

    xi = jnp.zeros(6, dtype=jnp.float32)

    H_n_final = jnp.zeros((6, 6), dtype=jnp.float32)
    residual_final = jnp.zeros_like(f0)
    w_total_final = jnp.zeros_like(f0)

    for level in reversed(range(n_levels)):
        scale = 1.0 / (2 ** level)
        K_lvl = scale_intrinsics(K, scale)
        f0_l = pyr0[level]
        f1_l = pyr1[level]
        inv_depth = jnp.ones_like(f0_l)

        for _ in range(n_iters_per_level):
            delta_xi, _res, H_n, _ws, residual, w_total = _gauss_newton_step(
                f0_l, f1_l, inv_depth, xi, K_lvl, grad_thresh,
            )
            xi = xi + delta_xi

        if level == 0:
            H_n_final = H_n
            residual_final = residual
            w_total_final = w_total

    h, w = f0.shape
    residual_rms, valid_frac, rot_obs, trans_obs, cond_num = _compute_vo_info(
        H_n_final, residual_final, w_total_final, h * w,
    )
    dynamic_residual = jnp.abs(residual_final) * w_total_final

    info = VOInfo(
        residual_rms=residual_rms,
        valid_frac=valid_frac,
        rot_observability=rot_obs,
        trans_observability=trans_obs,
        condition_number=cond_num,
        dynamic_residual=dynamic_residual,
    )

    return xi, jnp.ones_like(f0), info


def xi_to_R_t(xi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Decompose twist into (R, t)."""
    T = exp_se3(xi)
    return T[:3, :3], T[:3, 3]
