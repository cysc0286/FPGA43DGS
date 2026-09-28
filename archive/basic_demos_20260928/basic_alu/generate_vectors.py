"""Independent Python arithmetic oracle shared by RTL and real-board checks."""
from pathlib import Path
import hashlib
import json
import random

ROOT = Path(__file__).resolve().parent
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
OPERATIONS = ['ADD32', 'SUB32', 'MUL_U32', 'MUL_S32', 'AND', 'OR', 'XOR',
              'SHL32', 'SHR32', 'SAR32', 'MIN_S32', 'MAX_S32']


def signed(value):
    return value if value < (1 << 31) else value - (1 << 32)


def expected(op, a, b):
    if op == 0: value = (a + b) & MASK32
    elif op == 1: value = (a - b) & MASK32
    elif op == 2: value = a * b
    elif op == 3: value = signed(a) * signed(b)
    elif op == 4: value = a & b
    elif op == 5: value = a | b
    elif op == 6: value = a ^ b
    elif op == 7: value = (a << (b & 31)) & MASK32
    elif op == 8: value = a >> (b & 31)
    elif op == 9: value = (signed(a) >> (b & 31)) & MASK32
    elif op == 10: value = min(signed(a), signed(b)) & MASK32
    elif op == 11: value = max(signed(a), signed(b)) & MASK32
    else: return 0, 1
    return value & MASK64, 0


def main():
    rng = random.Random(0x3D650001)
    edges = [0, 1, 2, 15, 31, 32, 63, 0x7fffffff, 0x80000000,
             0xffffffff, 0xfffffffe, 0x55555555, 0xaaaaaaaa]
    pairs = [(a, b) for a in edges for b in edges]
    pairs += [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(64)]
    vectors = [(op, a, b, *expected(op, a, b))
               for op in range(len(OPERATIONS)) for a, b in pairs]
    vectors += [(op, 0x80000000, 0xffffffff, 0, 1) for op in (12, 255, 0xffffffff)]
    # Include a valid operation after an error, proving error is cleared.
    vectors.append((0, 0xffffffff, 1, 0, 0))
    folder = ROOT / 'data'; folder.mkdir(exist_ok=True)
    payload = ''.join('{:08x} {:08x} {:08x} {:016x} {:x}\n'.format(*v) for v in vectors)
    (folder / 'vectors.txt').write_text(payload, encoding='ascii', newline='\n')
    manifest = {'schema': 'basic-alu-golden-v1',
                'encoding': 'ASCII hexadecimal fields, LF newline, no header',
                'fields': ['opcode:u32', 'a:u32_bits', 'b:u32_bits', 'expected:u64_bits', 'error:u1'],
                'endianness': 'text fields are most significant digit first; register low word at 0x84, high word at 0x88',
                'comparison': 'exact bits, zero tolerance',
                'signed_representation': 'twos complement; signed operations interpret input bits as int32',
                'seed': '0x3d650001', 'vectors': len(vectors),
                'per_operation': len(pairs), 'operations': OPERATIONS,
                'sha256': hashlib.sha256(payload.encode('ascii')).hexdigest(),
                'semantics': '32-bit wrap for add/sub; full 64-bit products; shifts mask count to 5 bits'}
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
