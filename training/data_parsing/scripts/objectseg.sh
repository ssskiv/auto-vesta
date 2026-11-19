#!/usr/bin/env bash
set -euo pipefail
for dataset in rellis3d goose offsed; do
    python3 "$(dirname "$0")/../mask_editing.py" "label_config=objectseg_$dataset" color_map=objectseg "$@"
done
