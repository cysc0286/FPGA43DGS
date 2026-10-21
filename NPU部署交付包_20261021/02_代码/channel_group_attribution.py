"""按输出通道组归因首层卷积误差：把"38.10% 越界"落到具体的 16 通道组。

为什么单独做这一个
------------------
ICraft 把卷积权重按 `HWIO + 输出通道每 16 个一组` 打包
（见 `first_conv_weight_audit.py:15-21`，解码形状 `(7,7,3,4,16) -> (64,3,7,7)`）。
但现有诊断（`precision_probe.py` / `first_conv_layout_probe.py` / `verify.py`）
比较的都是**全局**指标：一个组的权重坏掉，全局 max 与越界比例一起变差，
却定位不到是哪 16 个通道。

板端记录的 38.10% 既不是 0%（纯精度损失）也不是 100%（整体布局错乱），
最像"部分输出通道正确、部分错误"。按组拆开就能区分：

  - 误差集中在某几组、其余组正常  → 该组权重上传/对齐问题（可修）
  - 各组误差均匀                  → 精度问题（换 qdtype，含未试过的 bf16）
  - 全部组严重越界                → 整体布局/权重大面积错误

运行位置
--------
纯 numpy，**不导入 torch / onnxruntime / SDK**，所以：

  * 能在板子上直接跑（工作目录 `mvsplat/`）：
        python -m gaussian_generation.npu_branch.channel_group_attribution \\
            --oracle  graphs/first_conv/real_weight/oracle.npz \\
            --actual  probe/real_weight_0.npy --out CHANNEL_GROUPS
  * 也能把 `.npy` 与 `oracle.npz` 拷到 PC 后在本机跑（推荐：板上内存紧张）

数据从哪来
----------
* `oracle_probe.py` 每轮把 NPU 实际输出存成 `<name>_<repeat>.npy`
* `known_conv.py` / `first_conv.py` 同样保存 `.npy` 与 `oracle.npz`
* `oracle.npz` 内的 `expected` 是 ONNX/CPU 参考

约定
----
* 输入数组按 **NCHW** 解释，通道轴 = 1；若拿到的是 NHWC 线上布局，用 `--layout NHWC`
  （注意 `artifacts.from_wire` 已经转回 NCHW，通常不需要）。
* 容差带与仓库一致：`atol + rtol*|expected|`，默认 `2e-4 / 2e-3`。
"""
import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_RTOL = 2e-3
DEFAULT_ATOL = 2e-4


def metrics(actual, expected, rtol, atol):
    """与 verify.errors 同语义（max/mean/rmse/allclose），另加越界比例。

    这里不 import verify.errors：那条路径会拉起 gaussian_generation.runtime
    与 SDK 相关模块，而本工具需要在没有 torch / 没有板子的机器上跑。
    """
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch {actual.shape} vs {expected.shape}")
    if not np.isfinite(actual).all():
        return dict(passed=False, reason="non-finite",
                    non_finite_ratio=float((~np.isfinite(actual)).mean()),
                    max_abs=float("inf"), mean_abs=float("inf"), rmse=float("inf"),
                    outside_ratio=1.0, count=int(actual.size))
    a = actual.astype(np.float64)
    e = expected.astype(np.float64)
    diff = np.abs(a - e)
    band = atol + rtol * np.abs(e)
    return dict(
        passed=bool(np.allclose(actual, expected, rtol=rtol, atol=atol)),
        max_abs=float(diff.max()),
        mean_abs=float(diff.mean()),
        rmse=float(np.sqrt(np.mean(diff * diff))),
        outside_ratio=float(np.mean(diff > band)),
        count=int(actual.size),
    )


def to_nchw(array, layout):
    if layout == "NCHW":
        return array
    if layout == "NHWC":
        return np.transpose(array, (0, 3, 1, 2))
    raise ValueError("layout must be NCHW or NHWC")


def attribute(actual, expected, group_size=16, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
    """按输出通道分组归因，并给出"是否局部化"的判定。"""
    if actual.ndim != 4:
        raise ValueError(f"expected a 4-D NCHW array, got shape {actual.shape}")
    channels = actual.shape[1]
    if group_size <= 0 or channels % group_size:
        # 通道数不是 16 的整数倍时，最后一组按余数处理，但要在报告里标出来
        pass

    groups = []
    for start in range(0, channels, group_size):
        stop = min(start + group_size, channels)
        groups.append(dict(
            group=start // group_size,
            channels=[start, stop - 1],
            **metrics(actual[:, start:stop], expected[:, start:stop], rtol, atol)))

    overall = metrics(actual, expected, rtol, atol)
    failing = [g["group"] for g in groups if not g["passed"]]
    passing = [g["group"] for g in groups if g["passed"]]

    ratios = [g["outside_ratio"] for g in groups]
    if overall["passed"]:
        verdict, reason = "all_pass", "全局通过，无需归因"
    elif not passing:
        # 全部组都失败：比较组间离散度，判断是"均匀"还是"有轻重"
        verdict = "uniform" if (max(ratios) - min(ratios)) < 0.15 else "all_fail_uneven"
        reason = ("所有组都越界且比例接近 → 更像精度/整体布局问题，而不是某一组权重坏了"
                  if verdict == "uniform" else
                  "所有组都越界但比例差异明显 → 先看比例最高的组")
    else:
        verdict = "localized"
        reason = (f"仅第 {failing} 组失败、第 {passing} 组正常 → 误差局部化在特定通道组，"
                  "优先查该组的权重上传/对齐")

    return dict(
        tolerance=dict(rtol=rtol, atol=atol), group_size=group_size,
        channels=channels, groups_in_total=len(groups),
        overall=overall, groups=groups,
        failing_groups=failing, passing_groups=passing,
        verdict=verdict, reason=reason,
    )


def to_markdown(report):
    lines = [
        f"| 组 | 通道范围 | 最大绝对误差 | RMSE | 越界比例 | 判定 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for g in report["groups"]:
        lines.append(
            f"| {g['group']} | {g['channels'][0]}–{g['channels'][1]} | "
            f"{g['max_abs']:.6g} | {g['rmse']:.6g} | {g['outside_ratio']:.2%} | "
            f"{'通过' if g['passed'] else '失败'} |")
    return "\n".join(lines)


def load_pair(oracle_path, actual_path, expected_key, actual_key):
    oracle_path, actual_path = Path(oracle_path), Path(actual_path)
    if actual_path.suffix == ".npy":
        actual = np.load(actual_path, allow_pickle=False)
    else:
        with np.load(actual_path, allow_pickle=False) as data:
            actual = data[actual_key]
    with np.load(oracle_path, allow_pickle=False) as data:
        expected = data[expected_key]
    return np.asarray(actual), np.asarray(expected)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oracle", type=Path, required=True, help="含参考输出的 npz")
    p.add_argument("--actual", type=Path, required=True, help="NPU 实际输出 .npy 或 .npz")
    p.add_argument("--expected-key", default="expected", help="oracle npz 内的键名")
    p.add_argument("--actual-key", default="actual", help="actual 为 npz 时的键名")
    p.add_argument("--group-size", type=int, default=16,
                   help="输出通道分组大小，ICraft 打包为 16")
    p.add_argument("--layout", choices=("NCHW", "NHWC"), default="NCHW")
    p.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    p.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    p.add_argument("--out", type=Path, help="报告 JSON 路径")
    p.add_argument("--markdown", action="store_true")
    a = p.parse_args(argv)

    actual, expected = load_pair(a.oracle, a.actual, a.expected_key, a.actual_key)
    if actual.shape != expected.shape:
        raise SystemExit(f"shape mismatch: actual {actual.shape} vs expected {expected.shape};"
                         " 若数组来自线上布局，请加 --layout NHWC")
    actual = to_nchw(actual, a.layout)
    expected = to_nchw(expected, a.layout)

    report = attribute(actual, expected, a.group_size, a.rtol, a.atol)
    report["oracle"], report["actual"] = str(a.oracle), str(a.actual)

    print(f"通道数 {report['channels']}，分 {report['groups_in_total']} 组（每组 {a.group_size}）")
    print(f"全局：max={report['overall']['max_abs']:.6g} "
          f"越界={report['overall']['outside_ratio']:.2%} "
          f"{'通过' if report['overall']['passed'] else '失败'}")
    print(to_markdown(report) if a.markdown else
          "\n".join(f"  组{g['group']:>2} [{g['channels'][0]:>3}-{g['channels'][1]:>3}] "
                    f"max={g['max_abs']:.6g} 越界={g['outside_ratio']:.2%} "
                    f"{'PASS' if g['passed'] else 'FAIL'}" for g in report["groups"]))
    print(f"\n判定：{report['verdict']}")
    print(f"理由：{report['reason']}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告：{a.out}")
    return 0 if report["verdict"] == "all_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
