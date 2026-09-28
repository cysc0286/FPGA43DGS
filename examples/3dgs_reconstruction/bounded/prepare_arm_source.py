"""Create a separately versioned ARM source using only required LibTorch headers.

No tensor operations, optimization flags, resource limits or model equations change.
The original bounded_v1 tree and patch remain immutable.
"""
import difflib
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    src = ROOT / 'vendor/OpenSplat_bounded_v1'
    dst = ROOT / 'vendor/OpenSplat_bounded_arm_v2'
    if dst.exists():
        raise FileExistsError(dst)
    manifest = json.loads((src / 'heterogs_source.json').read_text(encoding='utf-8'))
    for name, expected in manifest['modified_files'].items():
        if sha(src / name) != expected:
            raise ValueError('Original source changed: ' + name)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('build*', '.git', '__pycache__'))
    changes = {}
    def edit(name, old, new):
        p = dst / name
        before = p.read_text(encoding='utf-8')
        if old not in before:
            raise ValueError('Missing patch anchor: ' + name)
        p.write_text(before.replace(old, new, 1), encoding='utf-8', newline='\n')
        changes[name] = None
    simple = ['cv_utils.hpp', 'image_pipeline.hpp', 'input_data.hpp', 'kdtree_tensor.hpp',
              'nerfstudio.hpp', 'openmvg.hpp', 'opensfm.hpp', 'point_io.hpp', 'tensor_math.hpp']
    for name in simple:
        edit(name, '#include <torch/torch.h>', '#include <torch/types.h>')
    edit('model.hpp', '#include <torch/torch.h>',
         '#include <torch/types.h>\n#include <torch/utils.h>\n#include <torch/optim/adam.h>')
    for name in ['project_gaussians.hpp', 'rasterize_gaussians.hpp', 'spherical_harmonics.hpp', 'ssim.hpp']:
        edit(name, '#include <torch/torch.h>',
             '#include <torch/types.h>\n#include <torch/csrc/autograd/custom_function.h>')
    edit('rasterizer/gsplat-cpu/bindings.h', '#include <torch/all.h>', '#include <torch/types.h>')
    edit('rasterizer/gsplat-cpu/gsplat_cpu.cpp', '#include "bindings.h"',
         '#include "bindings.h"\n#include <torch/utils.h>\n#include <torch/nn/functional/normalization.h>\n#include <torch/nn/functional/padding.h>')
    edit('tensor_math.cpp', '#include "tensor_math.hpp"',
         '#include "tensor_math.hpp"\n#include <torch/nn/functional/normalization.h>')
    diff = []
    for name in sorted(changes):
        old = (src / name).read_text(encoding='utf-8').splitlines(keepends=True)
        new = (dst / name).read_text(encoding='utf-8').splitlines(keepends=True)
        # Assert the patch only changes preprocessor include lines and whitespace.
        body = lambda lines: [s.strip() for s in lines if s.strip() and not s.lstrip().startswith('#include')]
        if body(old) != body(new):
            raise AssertionError('Non-header change: ' + name)
        diff.extend(difflib.unified_diff(old, new, fromfile='bounded_v1/' + name, tofile='arm_v2/' + name))
        manifest['modified_files'][name] = sha(dst / name)
    patch = ROOT / 'bounded/arm_lean_headers_v2.patch'
    patch.write_text(''.join(diff), encoding='utf-8', newline='\n')
    manifest.update(variant='bounded_arm_v2', parent_manifest_sha256=sha(src / 'heterogs_source.json'),
                    arm_header_patch_sha256=sha(patch),
                    arm_change_scope='include dependencies only; CPU math, -O3 and resource ABI unchanged')
    (dst / 'heterogs_source.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'source': str(dst), 'header_only_files': len(changes), 'patch_sha256': sha(patch)}, indent=2))

if __name__ == '__main__':
    main()
