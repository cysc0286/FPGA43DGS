"""Clone the local licensed reference project and connect the original ALU RTL.

Vendor files remain under ignored platform/. Never edits the baseline project.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil

EXAMPLE = Path(__file__).resolve().parents[1]
REPO = next(p for p in EXAMPLE.parents if (p / "AGENTS.md").is_file() and (p / "platform").is_dir())
SOURCE = REPO / 'platform/fpai_demo_fpga'
TARGET = REPO / 'platform/basic_alu_fpga'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if TARGET.exists():
        raise SystemExit(f'Refusing to overwrite existing project: {TARGET}')
    top_relative = Path('rtl/adder_op/adder_top.v')
    originals = {str(p): digest(SOURCE / p) for p in
                 (top_relative, Path('fpai_demo_vivado.xpr'))}
    shutil.copytree(SOURCE, TARGET, ignore=shutil.ignore_patterns('.Xil', '*.log', '*.jou'))
    xpr = TARGET / 'fpai_demo_vivado.xpr'
    text = xpr.read_text(encoding='utf-8')
    text = text.replace(SOURCE.as_posix(), TARGET.as_posix())
    xpr.write_text(text, encoding='utf-8', newline='\n')
    top = TARGET / top_relative
    text = top.read_text(encoding='utf-8')
    marker = '  reg_ctrl   U_reg_ctrl('
    if text.count(marker) != 1:
        raise RuntimeError('Unexpected reference top: reg_ctrl instance not unique')
    addition = '''    wire [31:0] alu_a, alu_b, alu_opcode, alu_request;
    wire [31:0] alu_completed, alu_status;
    wire [63:0] alu_result;
    basic_alu U_basic_alu (
        .clk(ra_clk), .rst_n(ra_rst_n_1),
        .operand_a(alu_a), .operand_b(alu_b), .opcode(alu_opcode),
        .request_seq(alu_request), .completed_seq(alu_completed),
        .result(alu_result), .status(alu_status)
    );

'''
    text = text.replace(marker, addition + marker)
    ports = {'rfu_wreg0': 'alu_a', 'rfu_wreg1': 'alu_b',
             'rfu_wreg2': 'alu_opcode', 'rfu_wreg5': 'alu_request',
             'rfu_rreg0': 'alu_result[31:0]', 'rfu_rreg1': 'alu_result[63:32]',
             'rfu_rreg2': 'alu_completed', 'rfu_rreg3': 'alu_status',
             'rfu_rreg4': "32'h414c5531"}
    for port, signal in ports.items():
        text, count = re.subn(r'(\.' + port + r'\s*\()([^)]*)(\))',
                             lambda m: m[1] + ' ' + signal + ' ' + m[3], text)
        if count != 1:
            raise RuntimeError(f'Expected exactly one {port}, got {count}')
    top.write_text(text, encoding='utf-8', newline='\n')
    shutil.copy2(EXAMPLE / 'rtl/basic_alu.v', TARGET / 'rtl/adder_op/basic_alu.v')
    if originals != {p: digest(SOURCE / p) for p in originals}:
        raise RuntimeError('Baseline changed unexpectedly')
    record = {'source': str(SOURCE), 'target': str(TARGET),
              'baseline_sha256': originals, 'alu_rtl_sha256': digest(EXAMPLE / 'rtl/basic_alu.v'),
              'capability': '0x414c5531', 'vendor_sources_modified_in_clone': [str(top_relative)]}
    (TARGET / 'basic_alu_manifest.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
