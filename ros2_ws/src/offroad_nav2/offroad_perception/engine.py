"""Inference engine: three separate networks, one preprocessed frame. Pure torch, no ROS imports.

Networks (no shared weights, see Modules/Nav2/INTEGRATION_GUIDE.md section 2):
  * FreespaceSeg     checkpoint -> only the seg path   (backbone, SceneContext, SceneNeck, FreespaceSegHead)
  * FreespaceContour checkpoint -> only the contour path (backbone, FreespaceContourHead on features[4])
  * ObjectSeg        checkpoint -> full network
Each network is built once, with pretrained=False (no download), loaded with strict=True, run under
torch.no_grad() in fp16 (when on CUDA).
"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

INPUT_W, INPUT_H = 640, 320
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CONTOUR_RAYS = 37
CONTOUR_BINS = 46  # bin 45 == "nothing found / left the image": never an obstacle


def _ensure_models_on_path(models_root):
    """Put the autoware.off-road `Models/` directory on sys.path once (no per-file hacks)."""
    models_root = str(models_root)
    if models_root not in sys.path:
        sys.path.insert(0, models_root)


def load_state_dict(path, device='cpu'):
    sd = torch.load(path, map_location=device, weights_only=True)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    return sd


class _SegPath(torch.nn.Module):
    """Seg-only path of a FreespaceNetwork. The contour head is never called and is dropped."""

    def __init__(self, net):
        super().__init__()
        self.backbone = net.PreTrainedBackbone
        self.context = net.SceneContext
        self.neck = net.SceneNeck
        self.head = net.FreespaceSegHead

    def forward(self, x):
        f = self.backbone(x)
        return self.head(self.neck(self.context(f[4]), f), f)


class _ContourPath(torch.nn.Module):
    """Contour-only path of a FreespaceNetwork: backbone + contour head, no neck, no seg head."""

    def __init__(self, net):
        super().__init__()
        self.backbone = net.PreTrainedBackbone
        self.head = net.FreespaceContourHead

    def forward(self, x):
        return self.head(self.backbone(x)[4])


def _build_freespace(models_root, ckpt, device):
    _ensure_models_on_path(models_root)
    from model_components.freespace_networks import FreespaceNetwork
    from model_components.scene_seg_network import SceneSegNetwork
    net = FreespaceNetwork(SceneSegNetwork(pretrained=False))
    net.load_state_dict(load_state_dict(ckpt, 'cpu'), strict=True)
    return net


def _build_objectseg(models_root, ckpt):
    _ensure_models_on_path(models_root)
    from model_components.object_seg_network import ObjectSegNetwork
    from model_components.scene_seg_network import SceneSegNetwork
    net = ObjectSegNetwork(SceneSegNetwork(pretrained=False))
    net.load_state_dict(load_state_dict(ckpt, 'cpu'), strict=True)
    return net


def preprocess(rgb, device, half=True, mode='resize', crop_top=0, crop_bottom=0):
    """rgb: uint8 HxWx3 numpy (or CUDA/CPU tensor). Returns (1,3,320,640) normalised tensor.

    mode 'resize': squash full frame (matches training, A.Resize(640, 320)).
    mode 'crop':   drop crop_top / crop_bottom rows first, then resize (TODO(user) decision 7.8).
    """
    t = torch.as_tensor(rgb).to(device, non_blocking=True)
    if t.ndim != 3 or t.shape[2] != 3:
        raise ValueError(f'expected HxWx3 image, got {tuple(t.shape)}')
    if mode == 'crop':
        t = t[crop_top:t.shape[0] - crop_bottom if crop_bottom else t.shape[0]]
    elif mode != 'resize':
        raise ValueError(f'unknown preprocess mode {mode}')
    t = t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    if t.shape[2] != INPUT_H or t.shape[3] != INPUT_W:
        t = F.interpolate(t, size=(INPUT_H, INPUT_W), mode='bilinear', align_corners=False)
    mean = torch.tensor(MEAN, device=t.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=t.device).view(1, 3, 1, 1)
    t = (t - mean) / std
    return t.half() if half else t


class Engine:
    def __init__(self, models_root, seg_ckpt=None, contour_ckpt=None, objseg_ckpt=None,
                 device=None, half=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.half = (self.device.type == 'cuda') if half is None else half
        self.seg = self.contour = self.objseg = None
        if seg_ckpt:
            self.seg = self._prep(_SegPath(_build_freespace(models_root, seg_ckpt, self.device)))
        if contour_ckpt:
            self.contour = self._prep(_ContourPath(_build_freespace(models_root, contour_ckpt, self.device)))
        if objseg_ckpt:
            self.objseg = self._prep(_build_objectseg(models_root, objseg_ckpt))

    def _prep(self, m):
        m = m.to(self.device).eval()
        return m.half() if self.half else m

    def infer_tensor(self, x, want=('seg', 'contour', 'objseg'), sync=False):
        """x: preprocessed (1,3,320,640). Returns dict of numpy arrays (+ 'timing_ms' when sync)."""
        out, timing = {}, {}
        with torch.no_grad():
            for name in ('seg', 'contour', 'objseg'):
                net = getattr(self, name)
                if net is None or name not in want:
                    continue
                t0 = time.perf_counter()
                y = net(x)
                if name == 'contour':
                    r = torch.argmax(y.squeeze(0).squeeze(-1), dim=1).to(torch.uint8)  # (37,)
                else:
                    r = torch.argmax(y.squeeze(0), dim=0).to(torch.uint8)  # (320,640)
                out[name] = r.cpu().numpy()
                if sync:
                    timing[name] = (time.perf_counter() - t0) * 1000
        if sync:
            out['timing_ms'] = timing
        return out

    def infer(self, rgb, want=('seg', 'contour', 'objseg'), mode='resize', crop_top=0, crop_bottom=0):
        x = preprocess(rgb, self.device, self.half, mode, crop_top, crop_bottom)
        return self.infer_tensor(x, want)
