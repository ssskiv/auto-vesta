"""Camera geometry. Pure numpy, no ROS imports.

Frames: optical = x right, y down, z forward.  body (base_link / base_footprint) = x forward, y left, z up.
Transforms are 4x4 and map points of the child frame into the parent frame (T_parent_child).
Nothing here hard-codes extrinsics: callers pass them (from TF / vehicle.yaml).
"""
import numpy as np

MODEL_W, MODEL_H = 640, 320
CONTOUR_RAYS, CONTOUR_BINS, BIN_PX = 37, 46, 10
CONTOUR_START = (319, 320)  # (row, col) of the ray origin in model pixels
NO_HIT_BIN = CONTOUR_BINS - 1  # 45: ray found nothing; never an obstacle


# ---------------------------------------------------------------- transforms
def rpy_to_R(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def make_T(xyz, rpy=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(*rpy)
    T[:3, 3] = xyz
    return T


# URDF convention for an optical frame hanging off an x-forward body link: rpy = (-pi/2, 0, -pi/2)
OPTICAL_RPY = (-np.pi / 2, 0.0, -np.pi / 2)


def body_to_optical_T(xyz, pitch_down=0.0):
    """T_parent_optical for a camera at xyz whose optical axis points forward, pitched down by pitch_down rad."""
    T = make_T(xyz, (0.0, pitch_down, 0.0)) @ make_T((0, 0, 0), OPTICAL_RPY)
    return T


def transform(T, pts):
    pts = np.asarray(pts, dtype=float)
    return pts @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------- intrinsics
def scale_intrinsics(K, w, h, out_w=MODEL_W, out_h=MODEL_H, mode='resize', crop_top=0, crop_bottom=0):
    """K of the original w x h image -> K of the 640x320 model input (fx*640/W, fy*320/H, cx, cy likewise)."""
    K = np.array(K, dtype=float).reshape(3, 3).copy()
    if mode == 'crop':
        K[1, 2] -= crop_top
        h = h - crop_top - crop_bottom
    sx, sy = out_w / w, out_h / h
    K[0] *= sx
    K[1] *= sy
    return K


def model_to_image_coords(x, y, w, h, mode='resize', crop_top=0, crop_bottom=0):
    """640x320 model pixel coords -> original image pixel coords (float u, v). Plain scaling, the exact inverse
    of scale_intrinsics (same convention as fx' = fx*640/W, cx' = cx*640/W)."""
    hh = h - crop_top - crop_bottom if mode == 'crop' else h
    u = np.asarray(x, float) * (w / MODEL_W)
    v = np.asarray(y, float) * (hh / MODEL_H) + (crop_top if mode == 'crop' else 0)
    return u, v


# ---------------------------------------------------------------- depth
def median_depth(depth, u, v, win=5):
    """Median of the valid (finite, >0) depths in a win x win window around (u, v). NaN if none valid."""
    H, W = depth.shape
    r = win // 2
    u = np.rint(u).astype(int)
    v = np.rint(v).astype(int)
    out = np.full(u.shape, np.nan)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not ok.any():
        return out
    offs = np.arange(-r, r + 1)
    vv = np.clip(v[ok, None, None] + offs[None, :, None], 0, H - 1)
    uu = np.clip(u[ok, None, None] + offs[None, None, :], 0, W - 1)
    patch = depth[vv, uu].reshape(vv.shape[0], -1).astype(float)
    patch[~np.isfinite(patch) | (patch <= 0)] = np.nan
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            out[ok] = np.nanmedian(patch, axis=1)
    return out


def backproject(u, v, z, K):
    """Pixel (u, v) with depth z (along optical axis) -> optical-frame point."""
    u, v, z = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(z, float))
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1)


def project(pts_cam, K):
    p = np.asarray(pts_cam, float)
    return K[0, 0] * p[..., 0] / p[..., 2] + K[0, 2], K[1, 1] * p[..., 1] / p[..., 2] + K[1, 2]


def horizon_row(K, T_ground_cam):
    """Image row of the horizon (rays with zero upward component in the ground frame), at u = cx.
    Returns None when the optical axis y is parallel to the ground (no horizon in image)."""
    R = T_ground_cam[:3, :3]
    if abs(R[2, 1]) < 1e-9:
        return None
    return K[1, 2] - K[1, 1] * R[2, 2] / R[2, 1]


def above_horizon(v, K, T_ground_cam, margin_px=2.0):
    """True for pixel rows at or above the horizon (+margin): nothing there may be projected."""
    h = horizon_row(K, T_ground_cam)
    if h is None:
        return np.zeros(np.shape(v), bool)
    return np.asarray(v, float) < h + margin_px


def pixel_to_ground(u, v, K, T_ground_cam, ground_z=0.0):
    """Intersect the ray through pixel (u, v) with the plane z = ground_z of the ground frame.
    Returns ground-frame points, NaN where the ray does not hit the plane in front of the camera."""
    d_cam = backproject(u, v, 1.0, K)
    d = d_cam @ T_ground_cam[:3, :3].T
    o = T_ground_cam[:3, 3]
    with np.errstate(all='ignore'):
        s = (ground_z - o[2]) / d[..., 2]
    s = np.where(np.isfinite(s) & (s > 0), s, np.nan)
    return o + d * s[..., None]


# ---------------------------------------------------------------- ground plane
def fit_ground_plane(pts, up_prior, iters=200, thresh=0.08, max_tilt_deg=25.0, seed=0, min_pts=30):
    """RANSAC plane n.p + d = 0 on 3D points (any frame). up_prior = expected unit up vector in that frame
    (from TF/IMU); candidate planes tilted more than max_tilt_deg from it are rejected.
    Returns (n, d, inlier_mask) with n pointing the same way as up_prior, or None."""
    pts = np.asarray(pts, float)
    if len(pts) < min_pts:
        return None
    up = np.asarray(up_prior, float) / np.linalg.norm(up_prior)
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    cos_max = np.cos(np.deg2rad(max_tilt_deg))
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n @ up < 0:
            n = -n
        if n @ up < cos_max:
            continue
        d = -n @ s[0]
        inl = np.abs(pts @ n + d) < thresh
        c = inl.sum()
        if c > best_n:
            best, best_n = (n, d, inl), c
    if best is None or best_n < min_pts:
        return None
    n, d, inl = best
    # least-squares refit on inliers
    P = pts[inl]
    c = P.mean(0)
    n2 = np.linalg.svd(P - c, full_matrices=False)[2][-1]
    if n2 @ up < 0:
        n2 = -n2
    d2 = -n2 @ c
    return n2, d2, np.abs(pts @ n2 + d2) < thresh


def height_above_plane(pts, n, d):
    return np.asarray(pts, float) @ n + d


# ---------------------------------------------------------------- contour
def contour_endpoints(bins):
    """bins: (37,) ints 0..45. Returns rows, cols (float, clamped into the 640x320 image) and clear_only.

    Rays go left -> right (angle linspace(pi, 0, 37)) from (row 319, col 320); end = start + 10*bin px.
    clear_only is True for bin 45 (nothing found) and for rays that leave the image (image-edge hit):
    those endpoints may clear space but must never mark an obstacle.
    """
    bins = np.asarray(bins).astype(int)
    th = np.linspace(np.pi, 0, CONTOUR_RAYS)
    r = BIN_PX * bins
    row = CONTOUR_START[0] - r * np.sin(th)
    col = CONTOUR_START[1] + r * np.cos(th)
    outside = (row < 0) | (row > MODEL_H - 1) | (col < 0) | (col > MODEL_W - 1)
    clear_only = (bins >= NO_HIT_BIN) | outside
    return np.clip(row, 0, MODEL_H - 1), np.clip(col, 0, MODEL_W - 1), clear_only
