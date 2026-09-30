#!/bin/sh
set -eu
renderer="$1"
candidate="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
sdk="${ICRAFT_SDK_ROOT:-/root/heterogs_npu/sdk_3.36.1/usr}"
lib="$sdk/lib/aarch64-linux-gnu"
g++ -O3 -std=gnu++17 -ffp-contract=off -fopenmp -Wall -Wextra \
    "$candidate/live_renderer.cpp" -o "$candidate/live_renderer" \
    -I"$renderer/src/cpu" -I"$sdk/include" -L"$lib" -Wl,-rpath,"$lib" \
    -licraft_zg330backend -licraft_hostbackend -licraft_xrt -licraft_xir \
    -licraft_utils -ldw -ldl -pthread
