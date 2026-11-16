#!/bin/sh
set -eu
here="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
sdk="${ICRAFT_SDK_ROOT:-/root/heterogs_npu/sdk_3.36.1/usr}"
lib="$sdk/lib/aarch64-linux-gnu"
output="${1:-$here/camera_servo_native}"
g++ -O2 -std=gnu++17 -Wall -Wextra -mcpu=cortex-a53 \
    "$here/camera_servo_native.cpp" -o "$output" \
    -I"$sdk/include" -L"$lib" -Wl,-rpath,"$lib" \
    -licraft_zg330backend -licraft_hostbackend -licraft_xrt -licraft_xir \
    -licraft_utils -ldw -ldl -pthread
