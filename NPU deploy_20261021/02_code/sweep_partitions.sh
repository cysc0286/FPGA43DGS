#!/bin/bash
B=/root/fpga43dgs_reconstruction/npu_board_v2_20260929
PY=/root/fpga43dgs_reconstruction/arm_env/bin/python3
SDKLIB=/root/heterogs_npu/sdk_3.36.1/usr/lib/aarch64-linux-gnu:/root/fpga43dgs_reconstruction/arm_env/lib
PRE=/root/fpga43dgs_reconstruction/arm_env/lib/libgomp.so.1:/root/fpga43dgs_reconstruction/arm_env/lib/python3.11/site-packages/torch.libs/libgomp-58a43326.so.1.0.0
cd $B/mvsplat || exit 1

for p in backbone_cnn regressor_residual upsampler proj_feature to_gaussians to_disparity; do
  out=$B/sweep_$p
  rm -rf $out
  echo "########## $p start $(date +%H:%M:%S)" >> /tmp/sweep.log
  env PYTHONPATH=$B:$B/deps LD_LIBRARY_PATH=$SDKLIB OMP_NUM_THREADS=2 \
      $PY -m npu.verify_diagnostic --backend npu \
      --graphs $B/candidate/graphs --bundle $B/candidate/compiled \
      --input $B/candidate/input --weights $B/weights/re10k.ckpt \
      --vendor $B/vendor/MVSplat_reference --out $out \
      --library $B/arm_bridge/libmgs_npu.so --worker-python $PY \
      --partitions $p >> /tmp/sweep.log 2>&1
  if [ -f $out/hybrid/gaussians.npz ]; then
    env LD_PRELOAD=$PRE LD_LIBRARY_PATH=/root/fpga43dgs_reconstruction/arm_env/lib \
        PYTHONPATH=$B:$B/deps OMP_NUM_THREADS=2 \
        $PY -m npu.image_probe --input $B/candidate/input \
        --cpu $out/cpu/gaussians.npz --npu $out/hybrid/gaussians.npz \
        --out $out/image >> /tmp/sweep.log 2>&1
    echo "########## $p image done $(date +%H:%M:%S)" >> /tmp/sweep.log
  else
    echo "########## $p FAILED no gaussians $(date +%H:%M:%S)" >> /tmp/sweep.log
  fi
done
echo "########## SWEEP COMPLETE $(date +%H:%M:%S)" >> /tmp/sweep.log
