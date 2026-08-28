#!/usr/bin/env bash
# Fetch the example clips used in the README into data/.
#
#   bash scripts/download_examples.sh          # both clips
#   bash scripts/download_examples.sh drone    # just one
#
# Clips are hosted on Google Drive. To mirror them somewhere else, point the
# script at a directory containing drone.mp4 / driving.mp4:
#
#   export WEATHERCRAFTER_EXAMPLES_URL="https://example.com/clips"
set -euo pipefail

# name  sha256  google-drive-file-id
CLIPS=(
  "drone   16aa1075bf66c9d05fdfe70108ba531a32156a7813983f8aebf5453311d23acd 1yGe_Egw5qp_CiCglDivyohrdltchKw8P"
  "driving 08e89503d52f2796303fa30c8d0e6db6535ddab7ddb4447965ea9b4171da1e17 1OY70USlbAXP2hbmeHNBSHhBTzhv8PCrq"
)

BASE_URL="${WEATHERCRAFTER_EXAMPLES_URL:-}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

want="${1:-all}"
fetched=0

for entry in "${CLIPS[@]}"; do
    read -r name sha gid <<<"$entry"
    [ "$want" != "all" ] && [ "$want" != "$name" ] && continue

    dest="data/$name/$name.mp4"
    if [ -s "$dest" ]; then
        echo "[skip]  $dest already exists"
        fetched=$((fetched + 1))
        continue
    fi

    if [ -n "$BASE_URL" ]; then
        url="$BASE_URL/$name.mp4"
    elif [[ "$gid" == REPLACE_WITH_* ]]; then
        echo "[error] no download source configured for '$name'." >&2
        echo "        Either set WEATHERCRAFTER_EXAMPLES_URL to a mirror, or fill in" >&2
        echo "        the Google Drive file IDs at the top of this script." >&2
        echo "        See data/README.md to prepare the clip from its source instead." >&2
        exit 1
    else
        url="https://drive.google.com/uc?export=download&id=$gid"
    fi

    mkdir -p "data/$name"
    echo "[get]   $name.mp4"
    if ! curl -fL --progress-bar "$url" -o "$dest.part"; then
        rm -f "$dest.part"
        echo "[error] download failed for $name" >&2
        echo "        See data/README.md for how to prepare this clip yourself." >&2
        exit 1
    fi

    # Google Drive answers with an HTML interstitial when a file is rate-limited
    # or needs a virus-scan confirmation, which curl happily saves as a "success".
    if head -c 512 "$dest.part" | grep -qi "<!doctype html\|<html"; then
        rm -f "$dest.part"
        echo "[error] got an HTML page instead of $name.mp4." >&2
        echo "        Google Drive throttles automated downloads; try again later," >&2
        echo "        download it manually to $dest, or use a mirror via" >&2
        echo "        WEATHERCRAFTER_EXAMPLES_URL." >&2
        exit 1
    fi

    if command -v sha256sum >/dev/null 2>&1; then
        got="$(sha256sum "$dest.part" | cut -d' ' -f1)"
        if [ "$got" != "$sha" ]; then
            rm -f "$dest.part"
            echo "[error] checksum mismatch for $name.mp4" >&2
            echo "        expected $sha" >&2
            echo "        got      $got" >&2
            exit 1
        fi
    fi

    mv "$dest.part" "$dest"
    echo "[ok]    $dest"
    fetched=$((fetched + 1))
done

if [ "$fetched" -eq 0 ]; then
    echo "Unknown clip '$want'. Available: drone, driving" >&2
    exit 1
fi

cat <<'EOF'

Done. Next:

    python -m weathercrafter pipeline --dataset_name drone \
        --target_weather snowy --appearance_stage medium --particle_severity moderate

Note: these clips come from third-party sources with their own license terms --
see data/README.md before redistributing them or results derived from them.
EOF
