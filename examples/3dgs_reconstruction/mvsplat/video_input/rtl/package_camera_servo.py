"""Replace only the fixed-size PL payload in the accepted, unsigned 30TAI BOOT.

Procise must first generate an image containing only the patched candidate bit.
This script checks that image against the bit file before using its PL bytes.
No boot header, partition metadata, PS bytes, or board storage is modified.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import struct

ACCEPTED_BOOT = "b4e6d4d238d8a33bbf77c4d18df43143fa4304f7c713f09b479c6d5ccd39f9c6"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def partitions(data: bytes) -> list[dict]:
    require(len(data) >= 0x900, "Truncated FMSH image")
    require(struct.unpack_from("<II", data, 0x20) == (0xAA995566, 0x464D5348),
            "Unexpected FMSH boot header")
    require(struct.unpack_from("<I", data, 0x28)[0] == 0, "Encrypted image unsupported")
    table, first = struct.unpack_from("<II", data, 0x98)
    require(table % 4 == 0 and 0xB8 <= table <= len(data)-64, "Invalid image table")
    iht = struct.unpack_from("<16I", data, table)
    require(sum(iht) & 0xFFFFFFFF == 0xFFFFFFFF, "Image-table checksum mismatch")
    count = iht[1]
    require(1 <= count <= 4 and iht[2]*4 == first, "Unexpected partition count or pointer")
    result, seen = [], set()
    for index in range(count):
        require(first % 4 == 0 and table+64 <= first <= len(data)-64
                and first not in seen, "Invalid partition chain")
        seen.add(first)
        h = struct.unpack_from("<16I", data, first)
        require(sum(h) & 0xFFFFFFFF == 0xFFFFFFFF, "Partition-header checksum mismatch")
        require(h[0] == h[1] == h[2] and h[0] > 0 and h[10] == 1
                and h[11] == h[13] == h[14] == 0, "Signed/checksummed/expanded partition unsupported")
        offset, length = h[8]*4, h[0]*4
        require(first+64 <= offset <= len(data)-length, "Partition data outside image")
        require(h[9] in (0x10, 0x20, 0x110), "Unexpected partition attributes")
        result.append(dict(index=index, header=first, offset=offset, length=length,
                           entry=h[4] | h[5] << 32, load=h[6] | h[7] << 32,
                           attributes=h[9], sha256=sha(data[offset:offset+length])))
        first = h[3]*4
    require(first == 0, "Partition chain is longer than the image table")
    ranges = sorted((p["offset"], p["offset"]+p["length"]) for p in result)
    require(all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])), "Overlapping partitions")
    require(max(p["header"]+64 for p in result) <= ranges[0][0], "Headers overlap payloads")
    return result


def bit_payload(data: bytes) -> bytes:
    require(data[:13] == bytes.fromhex("00090ff00ff00ff00ff0000001"), "Unsupported bit header")
    pos = 13
    for tag in b"abcd":
        require(pos+3 <= len(data) and data[pos] == tag, "Malformed bit metadata")
        size = int.from_bytes(data[pos+1:pos+3], "big")
        pos += 3+size
    require(pos+5 <= len(data) and data[pos] == ord("e"), "Missing bit payload length")
    length = int.from_bytes(data[pos+1:pos+5], "big")
    raw = data[pos+5:]
    require(length == len(raw) and length > 0 and length % 4 == 0, "Invalid bit payload size")
    words = array("I", raw)
    require(words.itemsize == 4, "32-bit words required")
    words.byteswap()
    return words.tobytes()


def assemble(baseline: bytes, vendor_image: bytes, bit: bytes) -> tuple[bytes, dict]:
    require(sha(baseline) == ACCEPTED_BOOT, "Use the exact accepted rendering BOOT as baseline")
    old, vendor = partitions(baseline), partitions(vendor_image)
    require([p["attributes"] for p in old] == [0x10, 0x20, 0x110, 0x110], "Unexpected accepted layout")
    require(len(vendor) == 1 and vendor[0]["attributes"] == 0x20, "Procise input must contain only PL")
    target, candidate = old[1], vendor[0]
    require(candidate["length"] == target["length"], "PL length changed; a full boot rebuild is required")
    pl = vendor_image[candidate["offset"]:candidate["offset"]+candidate["length"]]
    raw = bit_payload(bit)
    padding = len(pl)-len(raw)
    require(0 <= padding <= 16 and padding % 4 == 0
            and pl == raw + bytes.fromhex("00000020")*(padding//4),
            "Vendor PL image differs from the candidate bitstream or NOP padding")
    start, end = target["offset"], target["offset"]+target["length"]
    output = baseline[:start] + pl + baseline[end:]
    after = partitions(output)
    require(output[:start] == baseline[:start] and output[end:] == baseline[end:],
            "Bytes outside the PL payload changed")
    require(all(old[i] == after[i] for i in (0, 2, 3)), "PS partition changed")
    return output, dict(baseline_sha256=sha(baseline), boot_sha256=sha(output),
                        bit_sha256=sha(bit), vendor_image_sha256=sha(vendor_image),
                        partitions=after, all_bytes_outside_pl_unchanged=True,
                        ps_partitions_unchanged=True, board_programmed=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "vendor_pl_image", "bit", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    report = args.output.with_suffix(args.output.suffix+".json")
    if args.output.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite a boot image or evidence")
    output, evidence = assemble(args.baseline.read_bytes(), args.vendor_pl_image.read_bytes(), args.bit.read_bytes())
    with args.output.open("xb") as stream:
        stream.write(output)
        stream.flush()
        os.fsync(stream.fileno())
    require(args.output.read_bytes() == output, "Output readback mismatch")
    with report.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2)
        stream.write("\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
