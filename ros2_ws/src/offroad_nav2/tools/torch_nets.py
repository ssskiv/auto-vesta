"""PyTorch side of the networks, used ONLY by tools/export_onnx.py (the runtime is C++ / ONNX Runtime, see src/core/engine.cpp).

Three separate networks, no shared weights:

Networks (no shared weights, see Modules/Nav2/INTEGRATION_GUIDE.md section 2):
  * FreespaceSeg     checkpoint -> only the seg path   (backbone, SceneContext, SceneNeck, FreespaceSegHead)
  * FreespaceContour checkpoint -> only the contour path (backbone, FreespaceContourHead on features[4])
  * ObjectSeg        checkpoint -> full network
Each network is built once, with pretrained=False (no download), loaded with strict=True, run under
torch.no_grad() in fp16 (when on CUDA).
"""
import sys

import torch

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
