#!/bin/sh
set -eu
renderer="$1"
candidate="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
sdk="${ICRAFT_SDK_ROOT:-/root/heterogs_npu/sdk_3.36.1/usr}"
lib="$sdk/lib/aarch64-linux-gnu"
profile="${2:-a53}"
output="${3:-$candidate/live_renderer}"
# Keep finite-value checks in every profile. FMA changes rounding explicitly;
# never use -ffast-math, which also removes those runtime checks.
case "$profile" in
    portable) flags="-ffp-contract=off" ;;
    a53) flags="-mcpu=cortex-a53 -ffp-contract=off -fno-math-errno -fno-trapping-math" ;;
    a53-fma) flags="-mcpu=cortex-a53 -ffp-contract=fast -fno-math-errno -fno-trapping-math" ;;
    *) echo "Unknown compiler profile: $profile" >&2; exit 2 ;;
esac
printf 'compiler_profile=%s flags=%s\n' "$profile" "$flags"
g++ -O3 -std=gnu++17 $flags -fopenmp -Wall -Wextra \
    "$candidate/live_renderer.cpp" -o "$output" \
    -I"$renderer/src/cpu" -I"$sdk/include" -L"$lib" -Wl,-rpath,"$lib" \
    -licraft_zg330backend -licraft_hostbackend -licraft_xrt -licraft_xir \
    -licraft_utils -ldw -ldl -pthread
