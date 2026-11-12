"""Network outputs + depth -> point clouds per class. Pure numpy, no ROS imports.

All returned point sets are in the camera OPTICAL frame (z forward), ready to be stamped with the image time.
Heights are measured against a ground plane fitted per frame in the footprint frame (RANSAC on pixels the freespace
network calls drivable, with a prior on the up direction), never against a hard-coded z.
"""
from dataclasses import dataclass, field

import numpy as np

from . import geometry as G

CLASS_NAMES = {1: 'manmade', 2: 'vru', 3: 'vehicle', 4: 'natural'}  # objseg: 0 bg, 1 man-made, 2 vulnerable living,
#                                                                    3 vehicle, 4 natural obstacle


@dataclass
class Params:
    stride: int = 4                 # model-pixel grid stride for dense clouds
    win: int = 5                    # median-depth window
    min_depth: float = 1.5
    max_depth: float = 20.0         # stereo error ~ z^2: trust depth only to ~20 m
    obstacle_h_min: float = 0.15
    obstacle_h_max: float = 2.2
    ground_h: float = 0.15          # |h| below this = flat ground
    horizon_margin_px: float = 3.0
    ransac_iters: int = 150
    ransac_thresh: float = 0.08
    ransac_max_pts: int = 1500
    max_tilt_deg: float = 20.0
    class_h_min: float = 0.05       # class points need h above this ...
    class_h_max: float = 2.5
    tall_h: float = 0.3             # a column needs a point this high for its ground-contact point to count
    lowest_h_min: float = -0.3      # ... except the lowest pixel of each column per class (ground contact)
    contour_marks_nondrivable: bool = True
    ego_box: tuple = None           # (xmin, xmax, ymin, ymax) in the ground frame: points inside are the car itself
    preprocess_mode: str = 'resize'
    crop_top: int = 0
    crop_bottom: int = 0


@dataclass
class FrameClouds:
    clouds: dict = field(default_factory=dict)   # name -> (N,3) float32, optical frame
    plane: tuple = None                          # (n, d) in the ground frame, or None (nominal plane used)
    stats: dict = field(default_factory=dict)


def _lift(u, v, depth, K, p):
    z = G.median_depth(depth, u, v, p.win)
    ok = np.isfinite(z) & (z >= p.min_depth) & (z <= p.max_depth)
    pc = np.full(u.shape + (3,), np.nan)
    pc[ok] = G.backproject(u[ok], v[ok], z[ok], K)
    return pc, ok


def build_clouds(seg, contour, objseg, depth, K, T_ground_cam, up_ground=(0, 0, 1), p=None, rng_seed=0):
    """seg/objseg (320,640) uint8 or None, contour (37,) or None, depth (H,W) float32 metres (NaN invalid),
    K full-resolution intrinsics, T_ground_cam = T_footprint_optical. Returns FrameClouds."""
    p = p or Params()
    Himg, Wimg = depth.shape
    out = FrameClouds()
    ys, xs = np.mgrid[0:G.MODEL_H:p.stride, 0:G.MODEL_W:p.stride]
    u, v = G.model_to_image_coords(xs, ys, Wimg, Himg, p.preprocess_mode, p.crop_top, p.crop_bottom)
    below = ~G.above_horizon(v, K, T_ground_cam, p.horizon_margin_px)
    # lift depth only below the horizon (nothing above it is projected anyway): ~half the work
    pc = np.full(u.shape + (3,), np.nan)
    ok = np.zeros(u.shape, bool)
    pc[below], ok[below] = _lift(u[below], v[below], depth, K, p)
    pg = np.full_like(pc, np.nan)
    pg[ok] = G.transform(T_ground_cam, pc[ok])
    if p.ego_box is not None:   # hood/bumper/mirrors seen by the camera are not obstacles
        x0, x1, y0, y1 = p.ego_box
        ego = ok & (pg[..., 0] > x0) & (pg[..., 0] < x1) & (pg[..., 1] > y0) & (pg[..., 1] < y1)
        ok &= ~ego

    # --- ground plane from drivable pixels
    up = np.asarray(up_ground, float)
    plane = None
    if seg is not None:
        drv = ok & (seg[ys, xs] == 1) & (pc[..., 2] < 20.0)
        pts = pg[drv]
        if len(pts) > p.ransac_max_pts:
            pts = pts[np.random.default_rng(rng_seed).choice(len(pts), p.ransac_max_pts, replace=False)]
        fit = G.fit_ground_plane(pts, up, p.ransac_iters, p.ransac_thresh, p.max_tilt_deg, seed=rng_seed)
        if fit is not None and abs(fit[1]) < 0.6:
            plane = (fit[0], fit[1])
    n, d = plane if plane is not None else (up / np.linalg.norm(up), 0.0)
    out.plane = plane
    h = np.full(ok.shape, np.nan)
    h[ok] = pg[ok] @ n + d

    f32 = lambda m: pc[m].astype(np.float32)
    # --- geometric obstacles: anything standing above the ground, from depth alone
    out.clouds['geometric'] = f32(ok & (h > p.obstacle_h_min) & (h < p.obstacle_h_max))
    if seg is not None:
        s = seg[ys, xs]
        nd = ok & (s == 0) & (np.abs(h) < p.ground_h)
        out.clouds['nondrivable'] = f32(nd)
        out.clouds['clear'] = f32(ok & (s == 1) & (h < p.ground_h) & ((xs // p.stride + ys // p.stride) % 2 == 0))
    # --- semantic classes
    if objseg is not None:
        o = objseg[ys, xs]
        for c, name in CLASS_NAMES.items():
            mc = ok & (o == c)
            keep = mc & (h > p.class_h_min) & (h < p.class_h_max)
            # lowest valid pixel of each grid column for this class = ground contact point
            rows = np.where(mc & (h > p.lowest_h_min) & (h < p.class_h_max), np.arange(mc.shape[0])[:, None], -1)
            low = rows.max(axis=0)
            # ground contact only for columns that hold a real standing object (paint marks classed "man-made" are flat)
            tall = np.nanmax(np.where(mc, np.nan_to_num(h, nan=-9), -9), axis=0) > p.tall_h
            cols = np.nonzero((low >= 0) & tall)[0]
            lowmask = np.zeros_like(mc)
            lowmask[low[cols], cols] = True
            out.clouds[name] = f32(keep | lowmask)
    # --- contour
    if contour is not None:
        r, c, clear_only = G.contour_endpoints(contour)
        eu, ev = G.model_to_image_coords(c, r, Wimg, Himg, p.preprocess_mode, p.crop_top, p.crop_bottom)
        epc, eok = _lift(eu, ev, depth, K, p)
        eok &= ~G.above_horizon(ev, K, T_ground_cam, p.horizon_margin_px)
        out.clouds['contour_boundary'] = epc[eok & ~clear_only].astype(np.float32)
        out.clouds['contour_clear_end'] = epc[eok & clear_only].astype(np.float32)
        if 'clear' in out.clouds:
            out.clouds['clear'] = np.vstack([out.clouds['clear'], out.clouds['contour_clear_end']])
        if p.contour_marks_nondrivable and 'nondrivable' in out.clouds:
            out.clouds['nondrivable'] = np.vstack([out.clouds['nondrivable'], out.clouds['contour_boundary']])
    out.stats = {'valid_depth_frac': float(ok.mean()), 'plane_fitted': plane is not None,
                 **{k: len(v_) for k, v_ in out.clouds.items()}}
    return out
