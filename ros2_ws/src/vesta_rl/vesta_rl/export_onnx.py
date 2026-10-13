"""Export the deterministic actor to ONNX and check it against torch.

    ros2 run vesta_rl export_onnx --model runs/sim/model.zip --out policy.onnx

Inputs: costmap [N, 1, PATCH_SIZE, PATCH_SIZE], vec [N, VEC_SIZE]. Output: action [N, 2].
"""
import argparse

import numpy as np
import torch
from stable_baselines3 import PPO

from . import config as C
from .policy import DeterministicActor


def export(model_path, out):
    """Exports and checks; returns the max |onnx - torch| difference."""
    model = PPO.load(model_path, device='cpu')
    actor = DeterministicActor(model.policy).eval()
    cm = torch.rand(1, 1, C.PATCH_SIZE, C.PATCH_SIZE)
    vec = torch.randn(1, C.VEC_SIZE) * 0.5
    torch.onnx.export(
        actor, (cm, vec), out,
        input_names=['costmap', 'vec'], output_names=['action'],
        dynamic_axes={'costmap': {0: 'n'}, 'vec': {0: 'n'}, 'action': {0: 'n'}},
        opset_version=17, dynamo=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(0)
    cm = rng.random((16, 1, C.PATCH_SIZE, C.PATCH_SIZE), dtype=np.float32)
    vec = (rng.standard_normal((16, C.VEC_SIZE)) * 0.5).astype(np.float32)
    onnx_out = sess.run(['action'], {'costmap': cm, 'vec': vec})[0]
    with torch.no_grad():
        torch_out = actor(torch.as_tensor(cm), torch.as_tensor(vec)).numpy()
    sb3_out, _ = model.predict({'costmap': cm, 'vec': vec}, deterministic=True)
    d_torch = np.abs(onnx_out - torch_out).max()
    d_sb3 = np.abs(onnx_out - np.clip(sb3_out, -1, 1)).max()
    if d_torch > 1e-5 or d_sb3 > 1e-5:
        raise RuntimeError(f'ONNX output does not match torch: {d_torch:.2e}, {d_sb3:.2e}')
    return max(d_torch, d_sb3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    d = export(args.model, args.out)
    print(f'exported {args.out}: max |onnx - torch / sb3 predict| = {d:.2e}')


if __name__ == '__main__':
    main()
