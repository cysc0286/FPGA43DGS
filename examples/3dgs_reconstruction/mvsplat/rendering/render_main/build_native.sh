#!/bin/sh
set -eu
here="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p "$here/bin"
sh "$here/cpu/mvsplat/rendering/build.sh" "$here/cpu" a53 "$here/bin/live_exact"
sha256sum "$here/bin/live_exact"
printf '%s\n' 'Rebuilt program: repeat board correctness validation before reusing performance claims.'
