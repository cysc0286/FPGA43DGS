# Source this only for the isolated CPU reconstruction environment.
# Usage: source bounded/arm_environment.sh [/root/fpga43dgs_reconstruction]
_hgs_cpu_base="${1:-/root/fpga43dgs_reconstruction}"
export PATH="$_hgs_cpu_base/arm_env/bin:$PATH"
export LD_LIBRARY_PATH="$_hgs_cpu_base/arm_env/lib"
# On the board's glibc 2.31, OpenCV's late libgomp dlopen exhausts static TLS.
# Load the same environment's OpenMP runtime before Python imports extensions.
export LD_PRELOAD="$_hgs_cpu_base/arm_env/lib/libgomp.so.1"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset _hgs_cpu_base
