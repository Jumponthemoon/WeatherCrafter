# Input data

Each clip lives in its own folder, named after the clip:

```
data/<dataset>/<dataset>.mp4
```

That is all `preprocess` needs — it extracts frames to `data/<dataset>/images/` and
resizes them so the shortest side is 504 px.

If you already have frames, drop them in `data/<dataset>/images/` as zero-padded
PNG or JPG (`0000.png`, `0001.png`, …). Extraction is skipped when that folder is
already populated; the frames are only resized.

```
data/
    mycity/
        mycity.mp4          <- your input clip
        images/             <- created by `preprocess`
            0000.png
            ...
```

## What makes a good input clip

- **5 seconds at 15 fps (75 frames).** This is what the defaults assume
  (`--fps 15 --sec 5`). Longer clips work, but VACE generates on a 4n+1 schedule
  and synthesis cost scales with length.
- **Visible ground.** Gravity is recovered by fitting the dominant ground plane in
  the estimated point map. Clips with no clear ground plane — tight indoor shots,
  extreme close-ups, pure sky — produce unreliable particle direction.
- **Moderate camera motion.** The appearance anchor conditions the whole clip from
  the first frame, so it drifts under large viewpoint change.

## Example clips

Two clips are used throughout the README and the paper figures:

| Clip | Scene | Resolution | Direct link |
|------|-------|-----------|-------------|
| `drone` | aerial shot of a large landmark | 896x504 | [download](https://drive.google.com/file/d/1yGe_Egw5qp_CiCglDivyohrdltchKw8P/view?usp=sharing) |
| `driving` | ego-view of a city intersection | 756x504 | [download](https://drive.google.com/file/d/1OY70USlbAXP2hbmeHNBSHhBTzhv8PCrq/view?usp=sharing) |

Both are 75 frames at 15 fps, ready for `preprocess`. They are hosted on Google
Drive rather than in this repository — fetch them with:

```bash
bash scripts/download_examples.sh          # both
bash scripts/download_examples.sh drone    # just one
```

The script verifies SHA-256 checksums and skips clips you already have.

If Google Drive refuses the automated request — it throttles them, and answers
with an HTML page rather than the file — download the clips by hand from the
links below and save them as `data/drone/drone.mp4` and
`data/driving/driving.mp4`. Or mirror them somewhere and point the script at it:

```bash
export WEATHERCRAFTER_EXAMPLES_URL="https://example.com/clips"
```

Then:

```bash
python -m weathercrafter pipeline --dataset_name drone \
    --target_weather snowy --appearance_stage medium --particle_severity moderate
```

### Preparing them yourself

If the download is unavailable, or you would rather source the data directly:

**`driving`** — the clip we use comes from the
[Waymo Open Dataset](https://waymo.com/open/). Register, accept the license, and
download a `segment-*.tfrecord` from the Perception set. Export 75 consecutive
front-camera frames, then either drop them into `data/driving/images/` or encode
them:

```bash
ffmpeg -framerate 15 -i frames/%03d.jpg -frames:v 75 \
       -c:v libx264 -crf 20 -pix_fmt yuv420p data/driving/driving.mp4
```

Any forward-facing dashcam or ego-view clip works as a substitute.

**`drone`** — a 5-second aerial shot of a large landmark. Any aerial clip with
visible ground works; the scene is only there to show that the method is not
driving-specific.

Any clip of your own works just as well — nothing in the pipeline is tied to these
two.

## A note on licensing

If you publish results derived from a third-party dataset, check its terms first.
Waymo Open Dataset, nuScenes, PandaSet, and DL3DV-10k each restrict redistribution
and commercial use in different ways, and those terms extend to frames and to
videos synthesised from them.
