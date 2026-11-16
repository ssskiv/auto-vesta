# Model definitions for ONNX export

This directory contains the 11 architecture files required by `../torch_nets.py`
to construct FreespaceSeg, FreespaceContour, and ObjectSeg networks. It is the
complete local import dependency set for those networks, including SceneSeg
and its EfficientNet backbone.

Source: `autowarefoundation/autoware.off-road`, `Models/model_components/`,
commit `be583909939a5defa80eb8fa0c06ae47118c2153`.
The source repository is `https://github.com/autowarefoundation/autoware.off-road`.
Copied on November 15, 2026. Architecture code is preserved; missing final
newlines are normalized. The upstream Apache 2.0 license is included as `LICENSE`.
The package initializer and this README were added in auto-vesta.

These definitions depend on Torch and Torchvision, already included in the
simulation image. Training, data preparation, and unrelated model families are
not included.

Place completed checkpoint copies in the checkout's `ml_models/` directory:

```text
ml_models/last_freespaceseg.pth
ml_models/last_freespacecontour.pth
ml_models/last_objectseg.pth
```

Run `make onnx` from the checkout root. The exporter imports this package beside
itself, both in the source tree and after ROS package installation. It constructs
backbones with `pretrained=False`, loads checkpoint weights strictly, and writes
`ml_models/onnx/{seg,contour,objseg}.onnx`. No external model repository is needed.
