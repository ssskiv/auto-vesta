"""The vendored architectures must satisfy the ONNX export tensor contracts."""

import sys
from pathlib import Path
from typing import Literal

import pytest
import torch
from typing_extensions import assert_never

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

from export_onnx import Wrapper
from model_components.freespace_networks import FreespaceNetwork
from model_components.object_seg_network import ObjectSegNetwork
from model_components.scene_seg_network import SceneSegNetwork
from torch_nets import _ContourPath, _SegPath


@pytest.mark.parametrize('kind, expected_shape', [
    ('seg', (1, 320, 640)),
    ('contour', (1, 37)),
    ('objseg', (1, 320, 640)),
])
def test_export_shapes_without_external_model_sources(
    kind: Literal['seg', 'contour', 'objseg'], expected_shape: tuple[int, ...],
) -> None:
    with torch.device('meta'):
        scene = SceneSegNetwork(pretrained=False)
        match kind:
            case 'seg':
                network = _SegPath(FreespaceNetwork(scene))
            case 'contour':
                network = _ContourPath(FreespaceNetwork(scene))
            case 'objseg':
                network = ObjectSegNetwork(scene)
            case unreachable:
                assert_never(unreachable)
        wrapper = Wrapper(network, kind)
        image = torch.zeros((1, 3, 320, 640), dtype=torch.uint8)

        output = wrapper(image)

    assert output.shape == expected_shape
    assert output.dtype == torch.uint8
