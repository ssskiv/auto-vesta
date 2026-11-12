import numpy as np

from offroad_perception import geometry as G

K = np.array([[640., 0, 320], [0, 640, 180], [0, 0, 1]])


def test_lift_depth_within_1mm():
    for (u, v, z) in [(320, 180, 5.0), (100, 50, 12.3), (600, 300, 3.1)]:
        p = G.backproject(u, v, z, K)
        u2, v2 = G.project(p, K)
        assert abs(u2 - u) < 1e-9 and abs(v2 - v) < 1e-9 and abs(p[2] - z) < 1e-3
        assert abs(p[0] - (u - 320) * z / 640) < 1e-3


def test_scale_intrinsics_and_coords():
    K0 = np.array([[1000., 0, 640], [0, 900, 360], [0, 0, 1]])
    Ks = G.scale_intrinsics(K0, 1280, 720)
    assert np.allclose([Ks[0, 0], Ks[1, 1], Ks[0, 2], Ks[1, 2]], [500, 400, 320, 160])
    # a pixel projected with K0 at original coords == same pixel projected with Ks at model coords
    p = np.array([2.0, -0.5, 10.0])
    u, v = G.project(p, K0)
    um, vm = G.project(p, Ks)
    ub, vb = G.model_to_image_coords(um, vm, 1280, 720)
    assert abs(ub - u) < 1e-9 and abs(vb - v) < 1e-9
    # crop mode: rows cropped, uniform scale
    Kc = G.scale_intrinsics(K0, 1280, 720, mode='crop', crop_top=40, crop_bottom=40)
    assert np.isclose(Kc[1, 1], 450) and np.isclose(Kc[1, 2], (360 - 40) * 0.5)


def cam_pose():
    return G.body_to_optical_T((1.6, 0, 1.3), pitch_down=np.deg2rad(5))


def test_ground_point_round_trip_under_1cm():
    T = cam_pose()
    for g in [(6, 0.5, 0), (10, -2, 0), (4, 1.2, 0), (20, 3, 0)]:
        g = np.array(g, float)
        pc = transform(np.linalg.inv(T), g)
        u, v = G.project(pc, K)
        back = G.pixel_to_ground(u, v, K, T)
        assert np.linalg.norm(back - g) < 0.01
        lifted = transform(T, G.backproject(u, v, pc[2], K))
        assert np.linalg.norm(lifted - g) < 0.01


def transform(T, p):
    return G.transform(T, p)


def test_left_of_centre_is_positive_y():
    T = cam_pose()
    p = G.transform(T, G.backproject(100, 180, 8.0, K))
    assert p[1] > 0 and p[0] > 1.6  # left of centre -> +y (left), in front of the camera
    p = G.transform(T, G.backproject(500, 180, 8.0, K))
    assert p[1] < 0


def test_optical_axis_points_forward_and_down():
    T = cam_pose()
    z_axis = T[:3, 2]
    assert z_axis[0] > 0.99 and z_axis[2] < 0  # forward, pitched down
    assert np.isclose(np.rad2deg(np.arcsin(-z_axis[2])), 5, atol=1e-6)


def test_bottom_centre_ray_ground_point():
    T = cam_pose()
    g = G.pixel_to_ground(320, 319, K, T)
    ang = np.arctan2(139, 640) + np.deg2rad(5)  # below optical axis + pitch
    assert abs(g[0] - (1.6 + 1.3 / np.tan(ang))) < 0.01 and abs(g[1]) < 1e-6 and abs(g[2]) < 1e-9


def test_above_horizon_rejected():
    T = G.body_to_optical_T((1.6, 0, 1.3), pitch_down=np.deg2rad(5))
    h = G.horizon_row(K, T)
    assert abs(h - (180 - 640 * np.tan(np.deg2rad(5)))) < 1e-6  # tilted down => horizon above centre
    assert G.above_horizon(h - 5, K, T) and not G.above_horizon(h + 20, K, T)
    assert np.isnan(G.pixel_to_ground(320, h - 5, K, T)).all()  # ray misses the ground


def test_ransac_recovers_plane():
    rng = np.random.default_rng(1)
    n_true = np.array([np.sin(np.deg2rad(4)), 0, np.cos(np.deg2rad(4))])
    n_true /= np.linalg.norm(n_true)
    xy = rng.uniform(-10, 10, (2000, 2))
    z = -(n_true[0] * xy[:, 0] + n_true[1] * xy[:, 1]) / n_true[2] + rng.normal(0, 0.01, 2000)
    pts = np.c_[xy, z]
    obst = rng.uniform(-3, 3, (400, 3)) + [0, 0, 1.0]  # obstacle outliers above ground
    n, d, inl = G.fit_ground_plane(np.r_[pts, obst], up_prior=[0, 0, 1])
    assert np.rad2deg(np.arccos(np.clip(n @ n_true, -1, 1))) < 1.0
    assert inl[:2000].mean() > 0.9


def test_contour_flags():
    bins = np.full(37, 10)
    bins[5] = 45
    bins[0] = 40   # 400 px left of col 320 -> col -80: leaves the image
    bins[18] = 20  # straight up: row 319-200=119, in image
    r, c, clear = G.contour_endpoints(bins)
    assert clear[5] and clear[0] and not clear[18]
    assert abs(r[18] - 119) < 1e-6 and abs(c[18] - 320) < 1e-6
    # bin 45 never an obstacle even if the endpoint is inside the image
    assert clear[5]
    assert (r >= 0).all() and (r <= 319).all() and (c >= 0).all() and (c <= 639).all()


def test_median_depth_invalid_and_window():
    d = np.full((20, 20), 5.0, np.float32)
    d[10, 10] = np.nan
    d[9, 9] = 0.0
    d[11, 11] = 99.0
    m = G.median_depth(d, np.array([10, 0, 50]), np.array([10, 0, 50]))
    assert m[0] == 5.0 and m[1] == 5.0 and np.isnan(m[2])
    d[:] = np.nan
    assert np.isnan(G.median_depth(d, np.array([5]), np.array([5]))).all()
