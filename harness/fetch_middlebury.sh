#!/usr/bin/env bash
# Fetches Middlebury 2014 stereo scenes (full resolution, "-perfect" i.e.
# radiometrically ideal calibration) directly as individual files -- not the
# ~100MB zip bundles, which include auxiliary imagery (ambient lighting
# variants, etc.) this project doesn't use.
#
# Usage: harness/fetch_middlebury.sh [DATA_DIR] [SCENE...]
#   DATA_DIR defaults to ./data
#   SCENE defaults to Motorcycle-perfect; pass one or more scene names
#   (see https://vision.middlebury.edu/stereo/data/scenes2014/ for the list)
# to fetch a larger validation set.
set -euo pipefail

DATA_DIR="${1:-data}"
shift || true
SCENES=("$@")
if [ ${#SCENES[@]} -eq 0 ]; then
    SCENES=("Motorcycle-perfect")
fi

BASE_URL="https://vision.middlebury.edu/stereo/data/scenes2014/datasets"

for scene in "${SCENES[@]}"; do
    dir="$DATA_DIR/$scene"
    mkdir -p "$dir"
    for f in calib.txt disp0.pfm im0.png im1.png; do
        dest="$dir/$f"
        if [ -s "$dest" ]; then
            echo "skip (already present): $scene/$f"
            continue
        fi
        echo "fetching: $scene/$f"
        curl -fsSL --max-time 120 -o "$dest" "$BASE_URL/$scene/$f"
    done
done

echo "done: ${SCENES[*]} in $DATA_DIR"
