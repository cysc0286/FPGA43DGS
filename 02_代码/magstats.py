import sys
import numpy as np

with np.load(sys.argv[1]) as d:
    e = d["expected"].astype(np.float64)
a = np.load(sys.argv[2]).astype(np.float64)

print("shape expected", e.shape, "actual", a.shape)
ae = np.abs(e)
err = np.abs(a - e)
print("|expected| mean %.5g max %.5g p99 %.5g" % (ae.mean(), ae.max(), np.percentile(ae, 99)))
print("|err|      mean %.5g max %.5g" % (err.mean(), err.max()))
rel = err / np.maximum(ae, 1e-12)
p50, p95, p999 = np.percentile(rel, [50, 95, 99.9])
print("rel err    mean %.3g p50 %.3g p95 %.3g p99.9 %.3g" % (rel.mean(), p50, p95, p999))
print("corr(|expected|, |err|) %.4f" % np.corrcoef(ae.ravel(), err.ravel())[0, 1])

a32 = np.ascontiguousarray(a, dtype=np.float32)
u = a32.view(np.uint32)
tf32 = (u & np.uint32(0xFFFFE000)).view(np.float32)
fp16 = a32.astype(np.float16).astype(np.float32)
print("tf32-exact fraction %.6f" % float(np.mean(tf32 == a32)))
print("fp16-exact fraction %.6f" % float(np.mean(fp16 == a32)))
print("zero fraction %.6f" % float(np.mean(a32 == 0)))
