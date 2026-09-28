"""Join completed FLK1 ablations without modifying their frozen measurements."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(folder, schedule):
    result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    physical = json.loads((folder / 'physical_provenance.json').read_text(encoding='utf-8'))
    if not result['complete'] or result['schedule'] != schedule:
        raise ValueError('Incomplete measurement or wrong schedule')
    if physical['boot_id'] not in (folder / 'environment.txt').read_text(encoding='utf-8'):
        raise ValueError('Physical boot provenance mismatch')
    if len(result['runs']) != len(summary['runs']):
        raise ValueError('Summary count mismatch')
    rows = {}
    for row in summary['runs']:
        timing = folder / 'received' / (row['stem'] + '_timing.csv')
        if digest(timing) != row['timing_sha256']:
            raise ValueError('Timing evidence drift')
        if not row['backend'].startswith('cpu') and not row['fp16_hls']['bitwise_equal']:
            raise ValueError('Hardware output failed its numerical contract')
        key = (row['case'], row['backend'])
        if key in rows:
            raise ValueError('Duplicate measurement key')
        rows[key] = row
    return result, physical, rows


def table(headers, rows):
    return '\n'.join(['|' + '|'.join(headers) + '|',
                      '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['|' + '|'.join(map(str, row)) + '|' for row in rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', required=True, type=Path)
    parser.add_argument('--serial', required=True, type=Path)
    parser.add_argument('--stability', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    _, physical, matrix = load_run(args.matrix, 'pipeline')
    _, serial_physical, serial = load_run(args.serial, 'serial')
    _, stable_physical, stable = load_run(args.stability, 'pipeline')
    for other in (serial_physical, stable_physical):
        for key in ('boot_sha256', 'boot_id', 'manifest_sha256', 'stage_result_sha256'):
            if other[key] != physical[key]:
                raise ValueError('Cannot pair different hardware/software provenance')
    analysis = json.loads((args.matrix / 'analysis.json').read_text(encoding='utf-8'))
    if analysis['input_summary_sha256'] != digest(args.matrix / 'summary.json'):
        raise ValueError('Quality analysis has stale timing summary')
    if analysis['input_result_sha256'] != digest(args.matrix / 'result.json'):
        raise ValueError('Quality analysis has stale measurement result')
    for row in analysis['runs']:
        if row['readback_sha256'] != digest(args.matrix / 'received' / (row['stem'] + '.bin')):
            raise ValueError('Image readback drift')
    cases = sorted({case for case, _ in matrix})
    paired, stages, warmups = [], [], []
    for case in cases:
        for mode in ('mode0', 'mode1', 'mode2'):
            p, s = matrix[case, mode], serial[case, mode]
            if p['golden_sha256'] != s['golden_sha256']:
                raise ValueError('Serial/pipeline golden mismatch')
            paired.append({'case': case, 'backend': mode,
                           'serial_ms': s['mean_ms'], 'pipeline_ms': p['mean_ms'],
                           'speedup': s['mean_ms'] / p['mean_ms'],
                           'latency_reduction_percent': 100 * (1 - p['mean_ms'] / s['mean_ms']),
                           'pipeline_upload_overlap_count': p['uploads_entirely_during_active_job'],
                           'serial_upload_overlap_count': s['uploads_entirely_during_active_job']})
        dense = matrix[case, 'mode2']
        if stable[case, 'mode2']['golden_sha256'] != dense['golden_sha256']:
            raise ValueError('Stability golden mismatch')
        trace = np.atleast_1d(np.genfromtxt(args.matrix / 'received' /
                             (dense['stem'] + '_trace.csv'), names=True, delimiter=','))
        means = dense['means']
        stages.append({'case': case, 'host_total_ms': dense['mean_ms'],
                       **{key + '_ms': means[key + '_us'] / 1000
                          for key in ('prepare', 'upload', 'wait', 'read')},
                       'other_host_ms': dense['mean_ms'] - sum(means[k + '_us'] / 1000
                                           for k in ('prepare', 'upload', 'wait', 'read')),
                       'hardware_cycles': means['hardware_cycles'],
                       'hardware_duration_ms': means['hardware_cycles'] / 200000,
                       'input_bytes': means['input_bytes'], 'output_bytes': means['output_bytes'],
                       'jobs': means['jobs'], 'cpu_percent': dense['cpu_percent'],
                       'unserved_explicit_read_requests_per_frame':
                           float(trace['unserved_kernel_read_requests'].sum() / dense['samples']),
                       'prefetch_overlap_cycles': means['prefetch_overlap_cycles']})
        log = (args.matrix / (dense['stem'] + '.txt')).read_text(encoding='utf-8')
        first = re.search(r'sample=-1 total_us=([\d.e+]+)', log)
        warmups.append({'case': case, 'first_prepared_frame_ms_rounded':
                        float(first.group(1)) / 1000 if first else None,
                        'scope': 'First warmup only; excludes process startup, scene load and initial allocation'})
    sources = {}
    for name in ('matrix', 'serial', 'stability'):
        folder = getattr(args, name)
        sources[name] = {'path': str(folder.resolve()), **{
            f + '_sha256': digest(folder / f)
            for f in ('result.json', 'summary.json', 'physical_provenance.json')}}
    output = {'sources': sources, 'physical': physical,
              'scope': analysis['scope'], 'pipeline_runs': analysis['runs'],
              'serial_pipeline_pairs': paired, 'dense_stages': stages,
              'stability_runs': list(stable.values()), 'first_warmup': warmups,
              'precision_note': analysis['precision_note'],
              'quality_note': analysis['quality_note'],
              'counter_limits': 'No CTU mask/VRU occupancy counters. Zero unserved explicit kernel reads does not prove zero HLS input stalls. Prefetch counter counts pushes while compute is in flight and not reading, not all overlapped cycles.',
              'tail_note': analysis['tail_note'], 'power': 'not measured', 'lpips': 'not measured'}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'comparison.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    lines = ['# FLK1 实板对照数据', '',
             '2026-09-27，悟净 30TAI Lite；559,263 点官方 train/iteration_7000 模型，'
             'v0/v10 两视角，320×178，240 个 Tile。常规组每组预热 1 次、计时 3 次；'
             '稳定性组每视角预热 1 次、计时 10 次。下表均由冻结数据生成。', '',
             '范围为已准备参数到完整帧回读，包含 CPU 打包、上传、提交、同步、回读和像素重组；'
             '不含离线 GPU 投影/SH/排序、首次分配及文件读写。CPU FP32 与 FPGA FP16 不同；'
             'CPU Dense 还缺少 FLK1 的 sub-tile AABB，不能解读为完全同算法、同精度的硬件加速。', '',
             '## 全部模式与 CPU 对照', '',
             table(['视角', '后端', '均值 ms', '中位 ms', '最小/最大 ms', 'P95/P99 ms', 'FPS', 'CPU占用 %'],
                   [[r['case'], r['backend'], f"{r['mean_ms']:.3f}", f"{r['median_ms']:.3f}",
                     f"{r['min_ms']:.3f}/{r['max_ms']:.3f}", f"{r['p95_ms']:.3f}/{r['p99_ms']:.3f}",
                     f"{r['fps']:.4f}", f"{r['cpu_percent']:.2f}"] for r in analysis['runs']]), '',
             'CPU 100% 约为一个核，400% 约为四核。所有 P95/P99 仅是小样本描述性分位数，'
             '不能声称可靠尾延迟；固定视角重复也不增加独立场景数。', '',
             '## 官方输出一致性', '',
             table(['视角', '后端', 'PSNR dB', 'SSIM', 'RGB最大/均值差', 'T最大差', 'last差异像素', '官方严格门槛'],
                   [[r['case'], r['backend'], f"{r['display_quality_vs_official']['psnr_clamped_rgb_db']:.4f}",
                     f"{r['display_quality_vs_official']['ssim_gaussian11_valid']:.7f}",
                     f"{r['official']['rgb_max']:.8f}/{r['official']['rgb_mean']:.8f}",
                     f"{r['official']['transmittance_max']:.8f}", r['official']['last_mismatches'],
                     '通过' if r['official']['passed'] else '失败'] for r in analysis['runs']]), '',
             '所有 FPGA 模式均逐位匹配自己的 FP16 HLS 参考（RGB/T/last 零差异），'
             '这与官方 FP32 严格门槛失败是两种不同结论。PSNR/SSIM 的对象是同视角官方渲染，'
             '不是测试照片；SSIM 使用 Gaussian 11×11、σ=1.5、有效内部窗口，RGB 裁剪至 [0,1]。'
             'LPIPS、实测功耗和能量/帧未测。', '',
             '## 串行与流水', '',
             table(['视角', '模式', '串行 ms', '流水 ms', '加速比', '延迟下降 %', '流水/串行完全重叠上传数'],
                   [[r['case'], r['backend'], f"{r['serial_ms']:.3f}", f"{r['pipeline_ms']:.3f}",
                     f"{r['speedup']:.4f}", f"{r['latency_reduction_percent']:.2f}",
                     f"{r['pipeline_upload_overlap_count']}/{r['serial_upload_overlap_count']}"] for r in paired]), '',
             '每组共 90 次上传。完全重叠只统计上传前后均观察到前一任务 busy 且未完成的区间，'
             '排除部分重叠；不是推算出来的理论流水收益。', '',
             '## Dense 时间分解', '',
             table(['视角', '总计 ms', '准备 ms', '上传 ms', '等待 ms', '回读 ms', '其他主机 ms', '硬件周期', '硬件单列 ms'],
                   [[r['case'], *[f"{r[k]:.3f}" for k in ('host_total_ms', 'prepare_ms', 'upload_ms',
                      'wait_ms', 'read_ms', 'other_host_ms')], f"{r['hardware_cycles']:.1f}",
                     f"{r['hardware_duration_ms']:.3f}"] for r in stages]), '',
             '硬件时长按 200 MHz 换算，是与 CPU 准备/上传重叠的设备区间，不能再加进总耗时。'
             '其他主机时间包含提交、状态读取、ACK 和像素重组，当前未再细分。', '',
             table(['视角', '上传 B/帧', '回读 B/帧', '任务/帧', '显式未满足读取请求/帧', '预取重叠事件/帧'],
                   [[r['case'], int(r['input_bytes']), int(r['output_bytes']), int(r['jobs']),
                     r['unserved_explicit_read_requests_per_frame'], f"{r['prefetch_overlap_cycles']:.1f}"] for r in stages]), '',
             '读取计数只捕获 kernel_read 为真但 empty_n 为假的拍；HLS 可在空 FIFO 时抑制 read，'
             '故零计数不能证明全部输入等待为零。预取计数为计算未结束且不读输入时的数据入队拍。'
             '没有 CTU 实际淘汰数、各 VRU 占用率或完整停顿归因计数，本轮不补造这些指标。', '',
             '## 连续 10 帧稳定性', '',
             table(['视角', '均值 ms', '中位 ms', '最小/最大 ms', 'P95/P99 ms', 'FPS'],
                   [[r['case'], f"{r['mean_ms']:.3f}", f"{r['median_ms']:.3f}",
                     f"{r['min_ms']:.3f}/{r['max_ms']:.3f}", f"{r['p95_ms']:.3f}/{r['p99_ms']:.3f}",
                     f"{r['fps']:.4f}"] for r in stable.values()]), '',
             '两视角所有重复帧 RGB/T/last 均精确一致；最终输出再与各自 HLS Golden 比较通过。'
             '这是短时连续运行检查，不是长时间可靠性试验。主矩阵首次预热帧约为 ' +
             ' / '.join(f"{r['first_prepared_frame_ms_rounded']:.3f} ms" for r in warmups) +
             '，日志只保留六位有效数字；不包含冷启动和首次模型载入。', '',
             '## 原始证据', '',
             *[f'- {name}: `{value["path"]}`' for name, value in sources.items()], '',
             '本目录 comparison.json 保存完整逐组统计、来源 SHA256 和 BOOT/程序版本；'
             '物理实现、论文对应、优劣与未完成项见 ../../pipeline/VALIDATION.md。', '']
    (args.output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
