"""Evaluate board readbacks on the cloud; never send references to the board."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--frames',type=Path,required=True)
    ap.add_argument('--directory',type=Path,required=True)
    args=ap.parse_args()
    sys.path.insert(0,str(args.source.resolve()))
    from utils.loss_utils import ssim
    from lpipsPyTorch import LPIPS
    from check_output import pixels,sha
    plan=json.loads((args.directory/'plan.json').read_text())
    criterion=LPIPS('vgg','0.1').cuda().eval()
    def tensor(path):
        w,h,p=pixels(path)
        a=np.clip(p['rgba'][:,:3].reshape(h,w,3),0,1).copy()
        return torch.from_numpy(a).permute(2,0,1).unsqueeze(0).cuda()
    result=dict(reference='Matching official CUDA image, same Gaussian subset and camera; NOT ground-truth photos',
        rgb_range='Raw float readback clamped to [0,1], no 8-bit quantization',
        convention='Pinned GraphDECO metrics.py SSIM and LPIPS(VGG 0.1), inputs [0,1]',
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),runs=[])
    result['metric_source_sha256']={str(p.relative_to(args.source)):sha(p) for p in
        [args.source/'utils/loss_utils.py',*sorted((args.source/'lpipsPyTorch').rglob('*.py'))]}
    with torch.no_grad():
        for item in plan:
            actual=args.directory/item['file'];expected=args.frames/item['case']/'official.bin'
            if sha(actual)!=item['actual_sha256']:raise ValueError('Board readback hash drift')
            a,b=tensor(actual),tensor(expected)
            mse=torch.mean((a.double()-b.double())**2).item()
            row=dict(**item,official_sha256=sha(expected),psnr_db=None if mse==0 else -10*math.log10(mse),
                ssim=ssim(a,b).item(),lpips_vgg=criterion(a,b).item())
            result['runs'].append(row)
            print(json.dumps(row),flush=True)
    (args.directory/'quality.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
