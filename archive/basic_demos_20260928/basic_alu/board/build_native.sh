#!/bin/sh
set -eu
cd "$(dirname "$0")"
[ "$(uname -m)" = aarch64 ] || { echo 'Requires the ARM64 board'; exit 2; }
g++ -std=gnu++17 -O0 -Wall -Wextra test_basic_alu.cpp -o test_basic_alu \
  -licraft_xrt -licraft_xir -licraft_utils -ldw -pthread -ldl
ldd ./test_basic_alu > dependencies.txt
if grep -q 'not found' dependencies.txt; then cat dependencies.txt; exit 3; fi
echo 'BUILD_NATIVE=PASS'
