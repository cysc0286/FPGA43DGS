"""Read-only verifier for this vendor's FMQL four-partition boot images."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
VENDOR=REPO/'2026FPGAInfo/20260916/BIT/BIT/30tai_lite_detpost/BOOT.bin'
VENDOR_SHA='ff350477e624c50d2f8180fb4b9130ec7688fbc7ca553412ed7c3dd2a68b31ef'

def sha(b):return hashlib.sha256(b).hexdigest()
def payload(bit):
    if bit[:2]!=b'\x00\x09':raise ValueError('not a supported bit header')
    pos=2+int.from_bytes(bit[:2],'big')+2
    for _ in range(5):
        tag=chr(bit[pos]);pos+=1;nlen=4 if tag=='e' else 2
        n=int.from_bytes(bit[pos:pos+nlen],'big');pos+=nlen
        if tag=='e':
            if pos+n!=len(bit) or n%4:raise ValueError('bad bit payload length')
            raw=bit[pos:pos+n]
            return b''.join(raw[i:i+4][::-1] for i in range(0,len(raw),4))
        if tag not in 'abcd':raise ValueError('bad bit header tag')
        pos+=n
    raise ValueError('missing bit payload')

def partitions(blob):
    if blob[0x20:0x28]!=bytes.fromhex('665599aa48534d46'):raise ValueError('not FMQL BOOT')
    off=struct.unpack_from('<I',blob,0x9c)[0];result=[]
    for i in range(4):
        words=struct.unpack_from('<16I',blob,off)
        if sum(words)&0xffffffff!=0xffffffff:raise ValueError('partition header checksum')
        if words[0]!=words[1] or words[1]!=words[2]:raise ValueError('encrypted/compressed partition unsupported')
        size=words[2]*4;start=words[8]*4
        if start+size>len(blob):raise ValueError('partition out of range')
        result.append({'index':i,'offset':start,'bytes':size,'flags':words[9],
                       'sha256':sha(blob[start:start+size]),'header':list(words)})
        off=words[3]*4
    if off!=0:raise ValueError('expected exactly four partitions')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--boot',type=Path,required=True)
    p.add_argument('--bit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();old=VENDOR.read_bytes();new=a.boot.read_bytes();bit=a.bit.read_bytes()
    assert sha(old)==VENDOR_SHA
    before=partitions(old);after=partitions(new);expected=payload(bit)
    assert [p['flags'] for p in after]==[0x10,0x20,0x110,0x110]
    same_ps=all(before[i]['sha256']==after[i]['sha256'] and before[i]['header']==after[i]['header'] for i in [0,2,3])
    pl=after[1];data=new[pl['offset']:pl['offset']+pl['bytes']]
    # The Procise packager pads this part with a complete little-endian NOOP.
    # Derive permitted trailing bytes from the frozen, boot-tested image.
    old_pl=before[1];old_data=old[old_pl['offset']:old_pl['offset']+old_pl['bytes']]
    match=(data[:len(expected)]==expected and len(data)==len(old_data) and
           0<=len(data)-len(expected)<64 and data[len(expected):]==old_data[len(expected):])
    assert same_ps and match,'PS changed or PL does not match bitstream'
    result={'boot_sha256':sha(new),'bit_sha256':sha(bit),'known_boot_sha256':sha(old),
      'boot_bytes':len(new),'non_pl_partitions_exact':same_ps,'pl_payload_exact':match,
      'partitions':after,'passed':True,'hardware_validation':'NOT_RUN'}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
