#!/usr/bin/env bash
set -euo pipefail
for dataset in rellis3d goose yamahacmu orfd offsed; do
    python3 "$(dirname "$0")/../mask_editing.py" "label_config=freespaceseg_$dataset" color_map=freespaceseg "$@"
done
