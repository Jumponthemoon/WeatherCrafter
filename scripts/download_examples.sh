#!/usr/bin/env bash
# Fetch the example clips used in the README into data/.
#
#   bash scripts/download_examples.sh              # both clips
#   bash scripts/download_examples.sh colosseum    # just one
#
# Point it somewhere else with:
#   export WEATHERCRAFTER_EXAMPLES_URL="https://example.com/path"
set -euo pipefail

BASE_URL="${WEATHERCRAFTER_EXAMPLES_URL:-https://huggingface.co/datasets/Jumponthemoon/weathercrafter-examples/resolve/main}"

# name  sha256
CLIPS=(
  "colosseum 16aa1075bf66c9d05fdfe70108ba531a32156a7813983f8aebf5453311d23acd"
  "waymo_172 08e89503d52f2796303fa30c8d0e6db6535ddab7ddb4447965ea9b4171da1e17"
)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

want="${1:-all}"
fetched=0

for entry in "${CLIPS[@]}"; do
    read -r name sha <<<"$entry"
    [ "$want" != "all" ] && [ "$want" != "$name" ] && continue

    dest="data/$name/$name.mp4"
    if [ -s "$dest" ]; then
        echo "[skip]  $dest already exists"
        fetched=$((fetched + 1))
        continue
    fi

    mkdir -p "data/$name"
    echo "[get]   $name.mp4"
    if ! curl -fL --progress-bar "$BASE_URL/$name.mp4" -o "$dest.part"; then
        rm -f "$dest.part"
        echo "[error] download failed for $name" >&2
        echo "        Check WEATHERCRAFTER_EXAMPLES_URL, or see data/README.md" >&2
        echo "        for how to prepare this clip yourself." >&2
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
    echo "Unknown clip '$want'. Available: colosseum, waymo_172" >&2
    exit 1
fi

cat <<'EOF'

Done. Next:

    python -m weathercrafter pipeline --dataset_name colosseum \
        --target_weather snowy --appearance_stage medium --particle_severity moderate

Note: these clips come from third-party sources with their own license terms --
see data/README.md before redistributing them or results derived from them.
EOF
