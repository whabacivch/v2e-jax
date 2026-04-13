"""Motion visualization helpers for geometry-based and flow-based pipelines.

Two paths are supported:

1. Pose + inverse depth + intrinsics -> per-pixel looming/lateral fields
2. Dense optical flow (+ optional intrinsics) -> per-pixel looming/lateral
   proxies suitable for webcam or video pipelines without depth

The geometry path is generic: build rays once from any pinhole intrinsics
matrix, or supply precomputed rays for a different camera model. The flow path
uses intrinsics only to normalize image-plane motion and define the optical
center used for radial vs tangential decomposition.

The RGB recipe maps looming/lateral magnitudes to red/green/blue weights, with
an optional magenta tint for dynamic regions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "skew",
    "exp_so3",
    "exp_se3",
    "camera_rays_from_intrinsics",
    "motion_field_from_rays",
    "motion_field_from_pose",
    "make_motion_field_fn",
    "motion_field_from_flow",
    "motion_colors_rgb_u8",
]


# ---------------------------------------------------------------------------
# SO(3) / se(3)
# ---------------------------------------------------------------------------

def skew(v: jnp.ndarray) -> jnp.ndarray:
    """3-vector -> 3x3 skew-symmetric matrix."""
    dtype = jnp.result_type(v, jnp.float32)
    v = jnp.asarray(v, dtype=dtype)
    x, y, z = v[0], v[1], v[2]
    return jnp.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=dtype,
    )


def exp_so3(omega: jnp.ndarray) -> jnp.ndarray:
    """Rodrigues map: rotation vector (axis x angle, rad) -> SO(3) matrix (3, 3)."""
    dtype = jnp.result_type(omega, jnp.float32)
    omega = jnp.asarray(omega, dtype=dtype)
    theta = jnp.linalg.norm(omega)
    omega_safe = omega / jnp.maximum(theta, 1e-8)
    k = skew(omega_safe)
    eye = jnp.eye(3, dtype=dtype)
    r = eye + jnp.sin(theta) * k + (1.0 - jnp.cos(theta)) * (k @ k)
    return jnp.where(theta < 1e-8, eye, r)


def exp_se3(xi: jnp.ndarray) -> jnp.ndarray:
    """se(3) exponential map: ``xi = [ω; v]`` (6,) -> 4x4 SE(3).

    Translation uses the left Jacobian V(ω) with t = V @ v.
    """
    dtype = jnp.result_type(xi, jnp.float32)
    xi = jnp.asarray(xi, dtype=dtype)
    omega = xi[:3]
    v = xi[3:]
    theta = jnp.linalg.norm(omega)
    r = exp_so3(omega)

    omega_safe = omega / jnp.maximum(theta, 1e-8)
    k = skew(omega_safe)
    theta_sq = jnp.maximum(theta * theta, 1e-12)
    theta_cu = jnp.maximum(theta * theta * theta, 1e-12)
    v_mat = (
        jnp.eye(3, dtype=dtype)
        + ((1.0 - jnp.cos(theta)) / theta_sq) * (theta * k)
        + ((theta - jnp.sin(theta)) / theta_cu) * (theta * theta * (k @ k))
    )
    v_mat = jnp.where(theta < 1e-6, jnp.eye(3, dtype=dtype), v_mat)
    t = v_mat @ v

    out = jnp.eye(4, dtype=dtype)
    out = out.at[:3, :3].set(r)
    out = out.at[:3, 3].set(t)
    return out


# ---------------------------------------------------------------------------
# Camera rays / motion field
# ---------------------------------------------------------------------------

def camera_rays_from_intrinsics(
    intrinsics: jnp.ndarray,
    shape: tuple[int, int],
    *,
    dtype: jnp.dtype | None = None,
) -> jnp.ndarray:
    """Return unnormalized camera rays for a rectified pinhole camera.

    The output is an ``(H, W, 3)`` ray field with z=1, suitable for reuse
    across many motion-field renders.
    """
    if dtype is None:
        dtype = jnp.result_type(intrinsics, jnp.float32)
    intrinsics = jnp.asarray(intrinsics, dtype=dtype)
    h, w = shape
    vv, uu = jnp.meshgrid(
        jnp.arange(h, dtype=dtype),
        jnp.arange(w, dtype=dtype),
        indexing="ij",
    )
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    return jnp.stack(
        [
            (uu - cx) / fx,
            (vv - cy) / fy,
            jnp.ones((h, w), dtype=dtype),
        ],
        axis=-1,
    )


@jax.jit
def motion_field_from_rays(
    xi: jnp.ndarray,
    inv_depth: jnp.ndarray,
    rays: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-pixel looming and lateral motion proxies from an inter-frame pose.

    ``rays`` should be the cached ``(H, W, 3)`` output of
    :func:`camera_rays_from_intrinsics`, or any equivalent camera-space ray
    field for a non-pinhole model.

    For pixel p with inverse depth rho(p): unproject to P0, map with exp(xi) to
    P1, delta = P1 - P0. Both arrays are normalized by ||P0||.

    looming
        ``-delta_z / ||P0||`` - positive means depth decreased.
    lateral
        ``sqrt(delta_x^2 + delta_y^2) / ||P0||`` - in-plane magnitude.
    """
    dtype = jnp.result_type(xi, inv_depth, rays, jnp.float32)
    xi = jnp.asarray(xi, dtype=dtype)
    inv_depth = jnp.asarray(inv_depth, dtype=dtype)
    rays = jnp.asarray(rays, dtype=dtype)

    z = 1.0 / jnp.maximum(inv_depth, jnp.asarray(1e-3, dtype=dtype))
    pts0 = rays * z[..., None]

    t_mat = exp_se3(xi)
    rot = t_mat[:3, :3]
    t = t_mat[:3, 3]
    pts1 = pts0 @ rot.T + t[None, None, :]
    delta = pts1 - pts0

    norm0 = jnp.sqrt(jnp.sum(pts0 * pts0, axis=-1) + jnp.asarray(1e-6, dtype=dtype))
    looming = -delta[..., 2] / norm0
    lateral = jnp.sqrt(delta[..., 0] ** 2 + delta[..., 1] ** 2) / norm0
    return looming, lateral


def motion_field_from_pose(
    xi: jnp.ndarray,
    inv_depth: jnp.ndarray,
    k_mat: jnp.ndarray,
    *,
    rays: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convenience wrapper that accepts intrinsics and optional cached rays.

    For repeated rendering with a fixed camera, prefer:

    1. ``rays = camera_rays_from_intrinsics(K, (H, W))``
    2. ``render = make_motion_field_fn(K, (H, W))`` or
       ``motion_field_from_rays(xi, inv_depth, rays)``.
    """
    if rays is None:
        rays = camera_rays_from_intrinsics(k_mat, inv_depth.shape, dtype=inv_depth.dtype)
    return motion_field_from_rays(xi, inv_depth, rays)


def make_motion_field_fn(
    intrinsics: jnp.ndarray,
    shape: tuple[int, int],
    *,
    dtype: jnp.dtype | None = None,
):
    """Build a JIT-compiled renderer bound to one camera and image shape."""
    rays = camera_rays_from_intrinsics(intrinsics, shape, dtype=dtype)

    @jax.jit
    def _render(
        xi: jnp.ndarray,
        inv_depth: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return motion_field_from_rays(xi, inv_depth, rays)

    return _render


def motion_field_from_flow(
    flow: jnp.ndarray,
    intrinsics: jnp.ndarray | None = None,
    *,
    flow_order: str = "xy",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-pixel looming/lateral proxies from dense optical flow.

    Parameters
    ----------
    flow : (H, W, 2)
        Dense optical flow in pixel units. ``flow_order="xy"`` expects
        ``[..., 0] = dx, [..., 1] = dy`` which matches OpenCV. Set
        ``flow_order="yx"`` for package-internal ``(dy, dx)`` flow tensors.
    intrinsics : (3, 3) or None
        Optional pinhole intrinsics used to normalize flow to camera units and
        to define the principal point. When omitted, the image center and raw
        pixel units are used.
    flow_order : {"xy", "yx"}
        Component ordering of ``flow``.

    Returns
    -------
    looming : (H, W)
        Signed radial flow. Positive means outward expansion from the optical
        center, negative means contraction.
    lateral : (H, W)
        Tangential flow magnitude around the optical center.

    Notes
    -----
    This is not a depth-aware motion decomposition. It is a 2D optical-flow
    interpretation that provides a webcam-friendly approximation aligned with
    the existing looming/lateral color recipe.
    """
    if flow_order not in {"xy", "yx"}:
        raise ValueError(f"Unsupported flow_order={flow_order!r}; expected 'xy' or 'yx'")

    dtype = jnp.result_type(flow, intrinsics if intrinsics is not None else jnp.float32, jnp.float32)
    flow = jnp.asarray(flow, dtype=dtype)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"Expected flow with shape (H, W, 2); got {flow.shape}")

    h, w, _ = flow.shape
    if flow_order == "xy":
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
    else:
        flow_y = flow[..., 0]
        flow_x = flow[..., 1]

    if intrinsics is None:
        fx = fy = jnp.asarray(1.0, dtype=dtype)
        cx = jnp.asarray((w - 1) * 0.5, dtype=dtype)
        cy = jnp.asarray((h - 1) * 0.5, dtype=dtype)
    else:
        intrinsics = jnp.asarray(intrinsics, dtype=dtype)
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

    vv, uu = jnp.meshgrid(
        jnp.arange(h, dtype=dtype),
        jnp.arange(w, dtype=dtype),
        indexing="ij",
    )
    x = (uu - cx) / fx
    y = (vv - cy) / fy
    flow_x = flow_x / fx
    flow_y = flow_y / fy

    radial_norm = jnp.sqrt(x * x + y * y)
    radial_safe = jnp.maximum(radial_norm, jnp.asarray(1e-6, dtype=dtype))
    radial_x = x / radial_safe
    radial_y = y / radial_safe

    looming = flow_x * radial_x + flow_y * radial_y
    tangential = -flow_x * radial_y + flow_y * radial_x
    lateral = jnp.abs(tangential)

    near_center = radial_norm < jnp.asarray(1e-6, dtype=dtype)
    speed = jnp.sqrt(flow_x * flow_x + flow_y * flow_y)
    looming = jnp.where(near_center, jnp.asarray(0.0, dtype=dtype), looming)
    lateral = jnp.where(near_center, speed, lateral)
    return looming, lateral


# ---------------------------------------------------------------------------
# RGB overlay (NumPy)
# ---------------------------------------------------------------------------

def motion_colors_rgb_u8(
    looming: np.ndarray,
    lateral: np.ndarray,
    event_mask: np.ndarray,
    *,
    dynamic_mask: np.ndarray | None = None,
    background: int = 240,
) -> np.ndarray:
    """Map (looming, lateral) to RGB uint8 on ``event_mask``; else gray."""
    looming = np.asarray(looming)
    lateral = np.asarray(lateral)
    event_mask = np.asarray(event_mask, dtype=bool)

    h, w = event_mask.shape
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[...] = background
    if not event_mask.any():
        return rgb

    loom = looming[event_mask].astype(np.float32)
    latr = lateral[event_mask].astype(np.float32)
    loom_pos = np.maximum(loom, 0.0)
    loom_neg = np.maximum(-loom, 0.0)
    total = loom_pos + loom_neg + latr + 1e-6
    w_g = loom_pos / total
    w_r = loom_neg / total
    w_b = latr / total
    r_ch = (40 + 200 * w_r).astype(np.uint8)
    g_ch = (40 + 200 * w_g).astype(np.uint8)
    b_ch = (40 + 200 * w_b).astype(np.uint8)
    rgb[event_mask] = np.stack([r_ch, g_ch, b_ch], axis=-1)

    if dynamic_mask is not None:
        dyn = np.asarray(dynamic_mask, dtype=bool) & event_mask
        if dyn.any():
            base = rgb[dyn].astype(np.float32)
            magenta = np.array([230.0, 40.0, 230.0], dtype=np.float32)
            rgb[dyn] = np.clip(0.4 * base + 0.6 * magenta, 0, 255).astype(np.uint8)

    return rgb
