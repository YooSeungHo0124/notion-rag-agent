#!/usr/bin/env bash
# Convert the WebM clips from scripts/record_demo.py into README GIFs (real-time, 8 fps).
set -euo pipefail
for f in docs/assets/raw/*.webm; do
  n=$(basename "$f" .webm)
  ffmpeg -v error -y -i "$f" -vf "fps=8,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" "docs/assets/demo_$n.gif"
  echo "docs/assets/demo_$n.gif $(du -h "docs/assets/demo_$n.gif" | cut -f1) $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")s"
done
