import numpy as np

from offroad_perception import geometry as G
from offroad_perception import postprocess as P
from offroad_perception.tracking import CentroidTracker, TemporalFilter, cluster_bev

W, H = 1280, 720
K = np.array([[1171.5, 0, 640], [0, 1171.5, 360], [0, 0, 1.]])
T = G.body_to_optical_T((1.55, 0.1, 1.45))   # camera 1.45 m above the ground frame (footprint), level


def synth_scene():
    """Flat ground + a 0.9 m wide, 1.4 m tall box 10 m ahead. Returns depth (H,W), seg, objseg at 640x320."""
    depth = np.full((H, W), np.nan, np.float32)
    vv, uu = np.mgrid[0:H, 0:W]
    dirs = G.backproject(uu, vv, 1.0, K) @ T[:3, :3].T
    with np.errstate(all='ignore'):
        s = (0 - T[2, 3]) / dirs[..., 2]
    ground_z = np.where((s > 0) & (s < 60), s, np.nan)           # optical-axis depth == s (rays normalised in z_cam=1)
    depth = ground_z.astype(np.float32)
    # box: x in [11.0, 11.9] from base (camera at 1.55 -> depth 9.45..10.35), |y|<0.45, z in [0,1.4]
    p = G.transform(T, G.backproject(uu, vv, np.where(np.isfinite(depth), depth, 1.0), K))
    # analytic box front face: cast rays to plane x = 11.0
    with np.errstate(all='ignore'):
        sb = (11.0 - T[0, 3]) / dirs[..., 0]
    hit = G.transform(T, np.zeros(3))  # unused
    yb = T[1, 3] + sb * dirs[..., 1]
    zb = T[2, 3] + sb * dirs[..., 2]
    box = (sb > 0) & (np.abs(yb) < 0.45) & (zb > 0) & (zb < 1.4) & ((sb < np.nan_to_num(depth, nan=1e9)) | ~np.isfinite(depth))
    depth[box] = sb[box]
    # model-resolution labels
    ys, xs = np.mgrid[0:320, 0:640]
    uf, vf = G.model_to_image_coords(xs, ys, W, H)
    boxm = box[np.clip(np.rint(vf).astype(int), 0, H - 1), np.clip(np.rint(uf).astype(int), 0, W - 1)]
    gm = np.isfinite(depth[np.clip(np.rint(vf).astype(int), 0, H - 1), np.clip(np.rint(uf).astype(int), 0, W - 1)])
    seg = (gm & ~boxm).astype(np.uint8)
    obj = np.zeros((320, 640), np.uint8)
    obj[boxm] = 3
    return depth, seg, obj


def test_obstacle_ground_and_horizon():
    depth, seg, obj = synth_scene()
    out = P.build_clouds(seg, np.full(37, 10), obj, depth, K, T)
    geo = G.transform(T, out.clouds['geometric'])                  # footprint frame
    assert out.plane is not None and abs(out.plane[1]) < 0.02       # fitted ground ~ z=0
    assert len(geo) > 50
    assert (np.abs(geo[:, 0] - 11.0) < 0.1).mean() > 0.95            # box front face
    assert (np.abs(geo[:, 1]) < 0.5).all() and geo[:, 2].min() > 0.1 and geo[:, 2].max() < 1.5
    gnd = G.transform(T, out.clouds['clear'][:, :3])
    assert (np.abs(gnd[:, 2]) < 0.15).all()                          # clear cloud is ground only
    vc = G.transform(T, out.clouds['vehicle'])
    assert len(vc) > 20 and (np.abs(vc[:, 0] - 11.0) < 0.1).all()
    for c in out.clouds.values():                                    # nothing above the horizon
        if len(c):
            u, v = G.project(c.astype(float), K)
            assert not G.above_horizon(v, K, T, 0.0).any()


def test_contour_edge_never_obstacle():
    depth, seg, obj = synth_scene()
    out = P.build_clouds(seg, np.full(37, 45), obj, depth, K, T)
    assert len(out.clouds['contour_boundary']) == 0                  # bin 45 everywhere: clear only
    assert len(out.clouds['contour_clear_end']) >= 1


def test_temporal_filter_and_cluster():
    tf = TemporalFilter(0.2, 3, 2)
    a = np.array([[1.0, 1.0], [5.0, 5.0]])
    assert not tf.update(a).any()                                   # first frame: nothing confirmed
    m = tf.update(np.array([[1.05, 1.02], [9.0, 9.0]]))
    assert m[0] and not m[1]                                        # repeated cell confirmed, new cell not
    pts = np.vstack([np.random.default_rng(0).normal([10, 2], 0.2, (30, 2)), np.random.default_rng(1).normal([20, -3], 0.2, (30, 2))])
    cl = cluster_bev(pts)
    assert len(cl) == 2
    tr = CentroidTracker()
    ids1 = [i for i, _ in tr.update(cl)]
    ids2 = [i for i, _ in tr.update(cl)]
    assert ids1 == ids2


def test_ego_box_removes_own_body():
    depth, seg, obj = synth_scene()
    # a thin 'hood' strip 2.4 m ahead of the camera (x = 3.95 in the ground frame), 0.9 m high
    out0 = P.build_clouds(seg, None, obj, depth, K, T)
    p = P.Params(ego_box=(-1.08, 4.18, -1.065, 1.065))
    out1 = P.build_clouds(seg, None, obj, depth, K, T, p=p)
    g0 = G.transform(T, out0.clouds['geometric'])
    g1 = G.transform(T, out1.clouds['geometric'])
    assert len(g1) > 0 and (g1[:, 0] > 4.18).all()
    assert len(g1) <= len(g0)
