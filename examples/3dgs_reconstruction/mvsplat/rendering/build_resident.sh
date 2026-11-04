#!/bin/sh
set -eu
renderer="$1"
candidate="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
sdk="${ICRAFT_SDK_ROOT:-/root/heterogs_npu/sdk_3.36.1/usr}"
lib="$sdk/lib/aarch64-linux-gnu"
g++ -O3 -std=gnu++17 -ffp-contract=off -Wall -Wextra \
    "$candidate/attributes_resident.cpp" -o "$candidate/attributes_resident" \
    -I"$renderer/src/cpu"
g++ -O3 -std=gnu++17 -ffp-contract=off -DPACKED_SORT -Wall -Wextra \
    "$candidate/group_sort_resident.cpp" -o "$candidate/group_sort_resident" \
    -I"$renderer/src/cpu"
g++ -O3 -std=gnu++17 -ffp-contract=off -fopenmp -Wall -Wextra \
    "$candidate/render_resident.cpp" -o "$candidate/render_resident" \
    -I"$renderer/src/cpu" -I"$sdk/include" -L"$lib" -Wl,-rpath,"$lib" \
    -licraft_zg330backend -licraft_hostbackend -licraft_xrt -licraft_xir \
    -licraft_utils -ldw -ldl -pthread
