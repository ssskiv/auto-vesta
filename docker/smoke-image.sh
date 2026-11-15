#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || -z "$1" ]]; then
  printf 'Usage: %s IMAGE_REF\n' "${0##*/}" >&2
  exit 2
fi

image_ref=$1
container_script=$(cat <<'CONTAINER_SCRIPT'
set -eo pipefail
source /opt/ros/humble/setup.bash
ros2 pkg prefix nav2_bringup
ros2 pkg prefix webots_ros2_driver
test -x /usr/local/webots/webots
test -s /opt/onnxruntime/include/onnxruntime_cxx_api.h

python3 - <<'PY'
import ctypes
import pathlib
import subprocess

import cv_bridge
import numpy
import onnxruntime
import torch

assert int(numpy.__version__.split(".")[0]) < 2, numpy.__version__
providers = onnxruntime.get_available_providers()
assert "CUDAExecutionProvider" in providers, providers
print("VERSIONS", numpy.__version__, torch.__version__, onnxruntime.__version__, flush=True)
print("ADVERTISED_PROVIDERS", providers, flush=True)

library_dir = pathlib.Path("/opt/onnxruntime/lib")
required = [
    library_dir / "libonnxruntime.so",
    library_dir / "libonnxruntime_providers_shared.so",
    library_dir / "libonnxruntime_providers_cuda.so",
]
for library in required:
    assert library.is_file(), f"required library missing: {library}"
    ctypes.CDLL(str(library))

allowed_host_driver = {"libcuda.so.1", "libnvidia-ml.so.1"}
for library in required:
    result = subprocess.run(["ldd", str(library)], capture_output=True, text=True, check=True)
    missing = [line.strip() for line in result.stdout.splitlines() if "not found" in line]
    print("REQUIRED_LIBRARY", str(library), "MISSING", missing, flush=True)
    unexpected = [line for line in missing if line.split()[0] not in allowed_host_driver]
    assert not unexpected, f"required library dependency missing: {unexpected}"

tensorrt_provider = library_dir / "libonnxruntime_providers_tensorrt.so"
if tensorrt_provider.is_file():
    result = subprocess.run(
        ["ldd", str(tensorrt_provider)], capture_output=True, text=True, check=True
    )
    missing = [line.strip() for line in result.stdout.splitlines() if "not found" in line]
    print("OPTIONAL_TENSORRT_PROVIDER present MISSING_DEPENDENCIES", missing, flush=True)
else:
    print("OPTIONAL_TENSORRT_PROVIDER absent", flush=True)

print(
    "CPU_DEPENDENCY_SMOKE_PASS; CUDA advertised and native core/CUDA dependencies checked; "
    "no CUDA initialization, GPU execution, or rendering claimed",
    flush=True,
)
PY
CONTAINER_SCRIPT
)

docker run --rm --pull=never --network=none \
  --entrypoint /bin/bash \
  -- \
  "$image_ref" \
  -lc "$container_script"
