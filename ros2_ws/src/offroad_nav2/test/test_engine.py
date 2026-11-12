import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from offroad_perception import engine as E

MODELS = os.environ.get('OFFROAD_MODELS', '/offroad/Models')
if not os.path.isdir(MODELS):
    MODELS = '/home/skiv/Programming/autoware.off-road/Models'
SEG = os.environ.get('SEG_CKPT', '/models/last_freespaceseg.pth')
CON = os.environ.get('CONTOUR_CKPT', '/models/last_freespacecontour.pth')
OBJ = os.environ.get('OBJSEG_CKPT', '/models/last_objectseg.pth')
have_ckpt = all(os.path.exists(p) for p in (SEG, CON, OBJ))
need_ckpt = pytest.mark.skipif(not have_ckpt, reason='checkpoints not found')
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')


def frame(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (320, 640, 3), dtype=np.uint8)


def test_preprocess_shape_and_modes():
    for img in (frame(), np.zeros((720, 1280, 3), np.uint8)):
        x = E.preprocess(img, 'cpu', half=False)
        assert x.shape == (1, 3, 320, 640) and x.dtype == torch.float32
    x = E.preprocess(np.zeros((720, 1280, 3), np.uint8), 'cpu', half=False, mode='crop', crop_top=40, crop_bottom=40)
    assert x.shape == (1, 3, 320, 640)
    with pytest.raises(ValueError):
        E.preprocess(np.zeros((10, 10), np.uint8), 'cpu')


def test_builds_with_network_blocked():
    """pretrained=False must not touch the network: run a fresh interpreter with sockets disabled."""
    code = (
        "import socket,sys;sys.path.insert(0,%r);"
        "socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('network blocked'));"
        "from offroad_perception import engine as E;"
        "E._ensure_models_on_path(%r);"
        "from model_components.scene_seg_network import SceneSegNetwork;"
        "from model_components.object_seg_network import ObjectSegNetwork;"
        "from model_components.freespace_networks import FreespaceNetwork;"
        "FreespaceNetwork(SceneSegNetwork(pretrained=False));ObjectSegNetwork(SceneSegNetwork(pretrained=False));print('ok')"
    ) % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), MODELS)
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120,
                       env={**os.environ, 'TORCH_HOME': '/nonexistent_torch_home'})
    assert 'ok' in r.stdout, r.stderr[-800:]


@pytest.fixture(scope='module')
def eng():
    return E.Engine(MODELS, SEG, CON, OBJ)


@need_ckpt
def test_output_ranges(eng):
    out = eng.infer(frame())
    assert out['seg'].shape == (320, 640) and set(np.unique(out['seg'])) <= {0, 1}
    assert out['contour'].shape == (37,) and out['contour'].max() <= 45
    assert out['objseg'].shape == (320, 640) and out['objseg'].max() <= 4


@need_ckpt
def test_matches_old_wrappers():
    """Seg/contour/objseg results equal the existing Models/inference wrappers (fp32, same checkpoint)."""
    from PIL import Image
    E._ensure_models_on_path(MODELS)
    sys.path.insert(0, os.path.join(MODELS, 'inference'))
    from freespace_seg_infer import FreespaceSegNetworkInfer
    from freespace_contour_infer import FreespaceContourNetworkInfer
    from object_seg_infer import ObjectSegNetworkInfer
    img = frame(3)
    e = E.Engine(MODELS, SEG, CON, OBJ, device='cpu', half=False)
    # the old wrappers use PIL ToTensor on a uint8 image: same normalisation as preprocess()
    pil = Image.fromarray(img)
    x = E.preprocess(img, 'cpu', half=False)
    got = e.infer_tensor(x)
    assert (got['seg'] == FreespaceSegNetworkInfer(SEG).inference(pil)).mean() > 0.999
    assert (got['contour'] == FreespaceContourNetworkInfer(CON).inference(pil)).all()
    assert (got['objseg'] == ObjectSegNetworkInfer(OBJ).inference(pil)).mean() > 0.999


@need_ckpt
@cuda
def test_fp16_fp32_agree():
    e16 = E.Engine(MODELS, SEG, CON, OBJ, device='cuda', half=True)
    e32 = E.Engine(MODELS, SEG, CON, OBJ, device='cuda', half=False)
    # a smooth synthetic scene, not noise: noise gives near-ties that make argmax unstable
    yy, xx = np.mgrid[0:320, 0:640]
    img = np.stack([(xx / 640 * 255), (yy / 320 * 255), ((xx + yy) % 255)], -1).astype(np.uint8)
    a = e16.infer(img)
    b = e32.infer(img)
    for k in ('seg', 'objseg'):
        assert (a[k] == b[k]).mean() >= 0.99, k
