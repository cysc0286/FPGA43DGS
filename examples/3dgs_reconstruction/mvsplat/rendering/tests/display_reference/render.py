"""Independent scalar framebuffer oracle retained from the frozen v1 renderer.

Test-only: the original write_ppm function is unchanged. No CLI, device access,
old renderer launch or manifest loader is retained here.
"""
import math
import struct


def write_ppm(source, output):
    raw = source.read_bytes()
    if raw[:8] != b'GSSOUT01':
        raise ValueError('Framebuffer ABI')
    w, h = struct.unpack_from('<II', raw, 8)
    if len(raw) != 16 + w * h * 20:
        raise ValueError('Framebuffer size')
    rgb = bytearray(w * h * 3)
    for i, values in enumerate(struct.iter_unpack('<4fI', raw[16:])):
        if not all(math.isfinite(v) for v in values[:4]):
            raise ValueError('Nonfinite framebuffer')
        for c in range(3):
            rgb[i * 3 + c] = round(max(0, min(1, values[c])) * 255)
    output.write_bytes(('P6\n%d %d\n255\n' % (w, h)).encode() + rgb)
