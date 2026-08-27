#!/usr/bin/env bash
# End-to-end weather synthesis. Edit the values below, then:  bash run_pipeline.sh
#
# Stages (see README for what each one does):
#   preprocess   extract frames + estimate depth/camera geometry (Depth Anything 3)
#   anchoring    edit the first frame to the target weather (GPT-4.1 + FLUX Kontext)
#   simulation   simulate + project the particle field, package the VACE inputs
#   synthesis    drive the VACE video editor on those inputs
#
# Prerequisites:
#   - conda activate weathercrafter  &&  pip install -e ".[synthesis]"
#   - an input video at data/<dataset>/<dataset>.mp4
#   - API keys exported (e.g. `source .env`): OPENAI_API_KEY, BFL_API_KEY
set -euo pipefail

dataset_name="mycity"          # data/<dataset_name>/<dataset_name>.mp4
target_weather="snowy"         # rainy | snowy
appearance_stage="medium"      # short | medium | long   (how long the weather has been going)
particle_severity="moderate"   # light | moderate | heavy (particle density / dynamics)
tea_cache="0.2"                # VACE TeaCache threshold (larger = faster, lower quality; 0 disables)

# Stages to run, in order. Drop any you don't need -- e.g. omit `synthesis` to stop
# at the VACE-ready inputs, or omit `preprocess` if the geometry already exists.
stages="preprocess,anchoring,simulation,synthesis"

python -m weathercrafter pipeline \
    --dataset_name "$dataset_name" \
    --target_weather "$target_weather" \
    --appearance_stage "$appearance_stage" \
    --particle_severity "$particle_severity" \
    --tea_cache_l1_thresh "$tea_cache" \
    --stages "$stages"
