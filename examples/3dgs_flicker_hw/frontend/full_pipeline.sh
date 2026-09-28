#!/bin/sh
set -eu
# Input paths are supplied by the hash-checking host runner.
model=$1
camera=$2
attributes=$3
group=$4
renderer=$5
work=$6
backend=$7
mkdir -p "$work"
start=$(date +%s%N)
"$attributes" "$model" "$camera" "$work/attributes.bin"
after_attributes=$(date +%s%N)
"$group" "$work/attributes.bin" "$work/scene.bin"
after_group=$(date +%s%N)
(
  cd "$work"
  case "$backend" in
    fpga) "$renderer" mode 2 scene.bin frame pipeline 8 1 0 ;;
    cpu_dense) "$renderer" cpu scene.bin frame dense 4 1 0 0 ;;
    cpu_base) "$renderer" cpu scene.bin frame base 4 1 0 0 ;;
    *) echo "unknown backend" >&2; exit 2 ;;
  esac
)
after_render=$(date +%s%N)
printf 'TIMING_NS start=%s after_attributes=%s after_group=%s after_render=%s\n' \
  "$start" "$after_attributes" "$after_group" "$after_render"
