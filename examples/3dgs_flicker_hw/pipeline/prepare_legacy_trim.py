"""Prepare, but do not install, a smaller legacy wrapper for congested FLK1 builds.

The vendor adder/DMA/register/reset path is retained. Historical GSC1/GSB1
compute units are removed from this candidate only; their archived platforms
remain the way to run those old experiments. Removed capabilities are not faked.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def trim(source):
    start = source.index('    wire [31:0] gs_alpha,')
    stop = source.index('  reg_ctrl   U_reg_ctrl(', start)
    result = source[:start] + '    // Historical GSC1/GSB1 units omitted in this candidate.\n\n' + source[stop:]
    for old, new in {
        'itf_ra_awvalid&&!batch_wsel': 'itf_ra_awvalid',
        'legacy_wready': 'itf_ra_awready',
        'itf_ra_arvalid&&!batch_rsel&&read_free': 'itf_ra_arvalid',
        'legacy_rready': 'itf_ra_arready',
        'legacy_rdata': 'itf_ra_rdata',
        'legacy_rvalid': 'itf_ra_rvalid',
        'itf_ra_rready&&itf_ra_rvalid': 'itf_ra_rready',
    }.items():
        result = result.replace(old, new)
    result, reads = re.subn(r'(\.rfu_rreg\d\s*\()\s*[^)]*\)', r"\g<1>32'd0)", result)
    result, writes = re.subn(r'(\.rfu_wreg\d\s*\()\s*[^)]*\)', r'\g<1>)', result)
    if reads != 10 or writes != 10 or re.search(r'\b(gs_\w+|batch_\w+|legacy_[rw]\w+)\b', result):
        raise ValueError('Unexpected source structure; do not apply trim')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=REPO/'platform/flicker_cat_compact_fpga/rtl/adder_op/legacy_adder_top.v')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = a.source.read_text()
    (a.output/'legacy_adder_top.v').write_text(trim(source), newline='\n')
    (a.output/'legacy_reference.v').write_text(source.replace('module legacy_adder_top(', 'module legacy_reference('), newline='\n')
    record = {'source': str(a.source.resolve()), 'source_sha256': hashlib.sha256(a.source.read_bytes()).hexdigest(),
              'candidate_sha256': hashlib.sha256((a.output/'legacy_adder_top.v').read_bytes()).hexdigest(),
              'state': 'prepared only, not platform-integrated',
              'removed': ['GSC1 register compositor', 'GSB1 register batch executor'],
              'retained': ['vendor adder', 'vendor DMA', 'base register map', 'reset/CDC'],
              'capabilities': {'0x9c': '0 (GSC1 absent)', '0x100': '0xffffffff (unmapped GSB1)', '0xc0': '0x20230628 (vendor adder)'}}
    (a.output/'derivation.json').write_text(json.dumps(record, indent=2))
    print(a.output)


if __name__ == '__main__':
    main()
