#!/bin/sh
set -eu
GS_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
GS_SDK=${ICRAFT_SDK_ROOT:-/root/heterogs_npu/sdk_3.36.1/usr}
GS_LIB="$GS_SDK/lib/aarch64-linux-gnu"
mkdir "$GS_ROOT/rebuilt"
g++ -O2 -std=gnu++17 -ffp-contract=off -Wall -Wextra "$GS_ROOT/src/cpu/attributes.cpp" -o "$GS_ROOT/rebuilt/attributes"
g++ -O2 -std=gnu++17 -ffp-contract=off -DPACKED_SORT -Wall -Wextra "$GS_ROOT/src/cpu/group_sort.cpp" -o "$GS_ROOT/rebuilt/group_sort"
g++ -O3 -std=gnu++17 -ffp-contract=off -fopenmp -Wall -Wextra "$GS_ROOT/src/cpu/render.cpp" -o "$GS_ROOT/rebuilt/render" -I"$GS_SDK/include" -L"$GS_LIB" -Wl,-rpath,"$GS_LIB" -licraft_zg330backend -licraft_hostbackend -licraft_xrt -licraft_xir -licraft_utils -ldw -ldl -pthread
printf '%s\n' 'Rebuilt CPU binaries are in rebuilt/. Frozen bin/ was preserved. Validate new binaries before use.'
