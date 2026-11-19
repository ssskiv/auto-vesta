#!/usr/bin/env python3
"""Export the three networks to ONNX for the C++ perception node (ONNX Runtime). Offline tool, uses PyTorch and local model_components/.

Each file takes a uint8 RGB image (1, 3, 320, 640) and returns the argmax class map as uint8:
  seg.onnx      -> (1, 320, 640)  {0 non-drivable, 1 drivable}       (FreespaceSeg checkpoint, seg path only)
  contour.onnx  -> (1, 37)        bins 0..45                          (FreespaceContour checkpoint, contour path only)
  objseg.onnx   -> (1, 320, 640)  {0 bg, 1 man-made, 2 vru, 3 vehicle, 4 natural}
ImageNet normalisation, fp16 weights/activations and the argmax are part of the graph.
Usage: export_onnx.py --ckpt-dir /models --out /onnx_out [--check]"""
import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch_nets as E  # noqa: E402


class Wrapper(torch.nn.Module):
    def __init__(self, net, kind):
        super().__init__()
        self.net, self.kind = net.half().eval(), kind
        # normalisation in fp32 (done in fp16 it differs between PyTorch and ONNX Runtime), the network itself in fp16
        self.register_buffer('mean', torch.tensor(E.MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer('std', torch.tensor(E.STD).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        y = self.net(((x.float() - self.mean) / self.std).half())
        if self.kind == 'contour':
            return torch.argmax(y.squeeze(-1), dim=2).to(torch.uint8)           # (1, 37)
        return torch.argmax(y, dim=1).to(torch.uint8)                           # (1, 320, 640)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default=Path('/models'), type=Path,
                    help='checkpoint root, with per-network subdirectories or flat last_*.pth files')
    ap.add_argument('--out', required=True)
    ap.add_argument('--check', action='store_true', help='compare with the PyTorch engine on a test image')
    ap.add_argument('--image', help='RGB image for --check (default: a smooth synthetic scene)')
    a = ap.parse_args()
    checkpoints: dict[str, Path] = {}
    for network in ('FreespaceSeg', 'FreespaceContour', 'ObjectSeg'):
        filename = f'last_{network.lower()}.pth'
        nested = a.ckpt_dir / network / filename
        flat = a.ckpt_dir / filename
        if not nested.is_file() and not flat.is_file():
            ap.error(f'checkpoint missing: {nested} or {flat}')
        checkpoints[network] = nested if nested.is_file() else flat
    nets = {
        'seg': E._SegPath(E._build_freespace(str(checkpoints['FreespaceSeg']))),
        'contour': E._ContourPath(E._build_freespace(str(checkpoints['FreespaceContour']))),
        'objseg': E._build_objectseg(str(checkpoints['ObjectSeg'])),
    }
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy = torch.randint(0, 255, (1, 3, 320, 640), dtype=torch.uint8, device=dev)
    os.makedirs(a.out, exist_ok=True)
    img = None
    if a.check:
        if a.image:
            import cv2
            img = cv2.cvtColor(cv2.resize(cv2.imread(a.image), (640, 320)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].copy()
        else:
            yy, xx = np.mgrid[0:320, 0:640]   # smooth synthetic scene (noise gives near-tie argmaxes)
            img = np.stack([xx / 640 * 255, yy / 320 * 255, (xx + yy) % 255], 0)[None].astype(np.uint8)
    for name, net in nets.items():
        w = Wrapper(net, name).to(dev)
        ref = w(torch.from_numpy(img).to(dev)).cpu().numpy() if a.check else None   # BEFORE the export: torch.onnx.export alters the module
        path = f'{a.out}/{name}.onnx'
        torch.onnx.export(w, dummy, path, input_names=['image'], output_names=['classes'], opset_version=17,
                          do_constant_folding=True, dynamo=False)
        print('exported', path, f'{os.path.getsize(path) / 1e6:.0f} MB')
        if a.check:
            import onnxruntime as ort
            s = ort.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            got = s.run(None, {'image': img})[0]
            print(f'  provider {s.get_providers()[0]}; agrees with torch on {(got == ref).mean() * 100:.2f} % of outputs')


if __name__ == '__main__':
    main()
