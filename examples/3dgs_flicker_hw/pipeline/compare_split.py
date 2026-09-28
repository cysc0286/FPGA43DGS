"""Compare immutable pre/post split runs with the same host and input ABI."""
import argparse,hashlib,json
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(folder):
 r=json.loads((folder/'result.json').read_text());s=json.loads((folder/'analysis.json').read_text());p=json.loads((folder/'physical_provenance.json').read_text())
 assert r['complete'] and s['input_result_sha256']==sha(folder/'result.json')
 assert s['input_summary_sha256']==sha(folder/'summary.json')
 assert p['boot_id'] in (folder/'environment.txt').read_text()
 for x in s['runs']:
  assert x['timing_sha256']==sha(folder/'received'/(x['stem']+'_timing.csv'))
  assert x['readback_sha256']==sha(folder/'received'/(x['stem']+'.bin'))
 return r,s,p,{(x['case'],x['backend']):x for x in s['runs']}
def table(h,rows):return '\n'.join(['|'+'|'.join(h)+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def main():
 p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 br,bs,bp,b=load(a.before);nr,ns,np,n=load(a.after)
 assert br['stage_record']==nr['stage_record'] and br['schedule']==nr['schedule']=='pipeline'
 assert bp['stage_result_sha256']==np['stage_result_sha256'] and bp['fpga_compute_clock_mhz']==np['fpga_compute_clock_mhz']==200
 assert bp['boot_sha256']!=np['boot_sha256']
 a.output.mkdir(exist_ok=False)
 pairs=[];performance=[];quality=[];allmodes=[];stages=[]
 cases=sorted({k[0] for k in b})
 for case in cases:
  old,new=b[case,'mode2'],n[case,'mode2']
  assert old['readback_sha256']==new['readback_sha256']
  speed=old['mean_ms']/new['mean_ms'];reduction=100*(1-1/speed)
  pairs.append(dict(case=case,old_dense_ms=old['mean_ms'],new_dense_ms=new['mean_ms'],speedup=speed,latency_reduction_percent=reduction,output_identical=True))
  performance.append([case,f"{n[case,'cpu4']['mean_ms']:.3f}",f"{n[case,'cpu_dense4']['mean_ms']:.3f}",f"{old['mean_ms']:.3f}",f"{new['mean_ms']:.3f}",f'{speed:.3f}x',f'{reduction:.2f}%',f"{new['fps']:.3f}"])
  q=new['display_quality_vs_official'];o=new['official']
  quality.append([case,f"{q['psnr_clamped_rgb_db']:.5f}",f"{q['ssim_gaussian11_valid']:.8f}",f"{q['raw_rgb_max']:.6f}",f"{q['raw_rgb_mean']:.8f}",f"{o['transmittance_max']:.6f}",o['last_mismatches'],'逐位一致'])
  means=new['means']
  stages.append([case,f"{means['prepare_us']/1000:.3f}",f"{means['upload_us']/1000:.3f}",f"{means['wait_us']/1000:.3f}",f"{means['read_us']/1000:.3f}",f"{means['hardware_cycles']/200000:.3f}",int(means['input_bytes']),int(means['output_bytes']),f"{new['cpu_percent']:.2f}%"])
  for mode in range(6):
   row=n[case,'mode'+str(mode)];q=row['display_quality_vs_official'];prior=b.get((case,'mode'+str(mode)))
   if prior:assert row['readback_sha256']==prior['readback_sha256']
   assert row['fp16_hls']['bitwise_equal']
   allmodes.append([case,mode,row['samples'],f"{row['mean_ms']:.3f}",f"{row['median_ms']:.3f}",f"{row['p95_ms']:.3f}",f"{row['p99_ms']:.3f}",f"{q['psnr_clamped_rgb_db']:.3f}",f"{q['ssim_gaussian11_valid']:.6f}"])
 report={'before':str(a.before.resolve()),'after':str(a.after.resolve()),'source_sha256':{str(x.resolve()):sha(x) for d in [a.before,a.after] for x in [d/'result.json',d/'summary.json',d/'analysis.json',d/'physical_provenance.json']},'pairs':pairs,'physical_before':bp,'physical_after':np,'scope':nr['scope'],'power':'not measured','lpips':'not measured'}
 (a.output/'comparison.json').write_text(json.dumps(report,indent=2))
 lines=['# FPGA split 优化：同板完整场景对照','',
 '同一官方 train/iteration_7000、559,263 Gaussian、两个固定视角、320×178；相同 ARM 可执行文件、输入及 200 MHz。时间为已准备帧的 CPU 打包、搬运、同步、回读及重组，不包含离线 GPU 投影/SH/分组/排序。每组预热1次、测量5次。','',
 table(['视角','CPU4 ms','CPU Dense4 ms','旧 FPGA Dense ms','新 FPGA Dense ms','前后加速','延迟下降','新 FPS'],performance),'',
 'CPU 使用 FP32，FPGA 使用 FP16；CPU Dense 还缺新硬件的 sub-tile AABB。因此 CPU 比值是当前实现的实测对照，不能解释为同精度、同算法的纯硬件收益。旧/新 FPGA Dense 则输出逐位一致，能隔离此次调度修改。','',
 '## 正确性与画质','',table(['视角','PSNR dB','SSIM','RGB最大误差','RGB平均误差','T最大误差','last差异像素','旧版/Golden'],quality),'',
 'PSNR/SSIM 对同视角官方渲染图，非实拍测试照片。硬件与各模式 FP16 Golden 逐位相等；官方 FP32 严格检查仍失败。LPIPS、功耗未测。','',
 '## 六种模式及延迟分布','',table(['视角','模式','样本数','平均ms','中位ms','P95 ms','P99 ms','PSNR dB','SSIM'],allmodes),'',
 'P95/P99 仅为5个重复固定视角样本的描述性分位数，不能证明实际应用尾延迟。模式0/1/2/3/4/5依次为无筛选/AABB/Dense/Sparse/Smooth-Focused/Spiky-Focused。','',
 '## 数据搬运和主机开销','',table(['视角','准备ms','上传ms','等待ms','回读ms','硬件区间ms','输入B','输出B','CPU占一个核比例'],stages),'',
 '硬件区间与主机工作重叠，不能把它再次加进总时间。CPU比例100%相当于占满一个核。缓冲复用、输入/输出字节和30个任务的数量保持相同。','',
 '## 来源','',f'- 更新前：`{a.before.resolve()}`',f'- 更新后：`{a.after.resolve()}`',f"- 新物理报告：`{np['physical_report']}`",f"- 新 BOOT：`{np['boot_sha256']}`",'','资源/时序和连续运行补充见同目录 `VALIDATION.md`。原始结果及来源文件哈希保存在 `comparison.json`。']
 (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(pairs,indent=2))
if __name__=='__main__':main()
