"""Exercise framing across fragmented reads and the complete-RGB timing contract."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from rendering.runtime import LiveRenderer


class FramedRuntime(LiveRenderer):
    def __init__(self, records):
        self.pending=bytearray()
        self.fragments=iter(records)
        self.scene={"gaussians":1}
        self.revision=1
        self.configuration={}
        self.sent=[]

    def _receive(self, deadline):
        self.pending.extend(next(self.fragments))

    def _send(self, value):
        self.sent.append(bytes(value))


def camera():
    return b"FLCAM001"+struct.pack("<4I",2,1,2,1)+struct.pack("<14d",*([1.0]*14))


def wire(pixels):
    header=dict(width=2,height=1,rgb_bytes=6,raw_bytes=0)
    return b"FRAME "+json.dumps(header).encode()+b"\n"+pixels


class LiveProtocolTests(unittest.TestCase):
    def test_fragmented_frames_have_separate_immutable_pixels(self):
        first=b"\0\1\2\3\4\5";second=b"\6\7\10\11\12\13"
        all_bytes=wire(first)+wire(second)
        runtime=FramedRuntime([all_bytes[i:i+7] for i in range(0,len(all_bytes),7)])
        a=runtime.render_camera(camera())
        b=runtime.render_camera(camera())
        self.assertEqual(a.rgb.tobytes(),first)
        self.assertEqual(b.rgb.tobytes(),second)
        self.assertFalse(a.rgb.flags.writeable)
        self.assertTrue(a.metadata["complete"])
        self.assertEqual(runtime.sent,[b"RENDER\n"+camera()]*2)

    def test_incomplete_payload_cannot_publish_frame(self):
        runtime=FramedRuntime([wire(b"\0")])
        with self.assertRaises(StopIteration):
            runtime.render_camera(camera())

    def test_archive_is_explicit_and_retains_frame(self):
        runtime=FramedRuntime([wire(b"\0\1\2\3\4\5")])
        f=runtime.render_camera(camera())
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"frame"
            self.assertFalse(out.exists())
            r=f.archive(out)
            self.assertEqual(r["rgb_sha256"],hashlib.sha256(f.rgb.tobytes()).hexdigest())
            self.assertEqual((out/"frame.ppm").read_bytes(),b"P6\n2 1\n255\n"+f.rgb.tobytes())
            self.assertFalse((out/"frame.bin").exists())

    def test_wrong_raw_contract_is_rejected(self):
        runtime=FramedRuntime([wire(b"\0\1\2\3\4\5")])
        with self.assertRaisesRegex(ValueError,"Raw frame shape"):
            runtime.render_camera(camera(),include_raw=True)

    def test_nonfinite_camera_is_rejected_before_submit(self):
        runtime=FramedRuntime([])
        bad=bytearray(camera());struct.pack_into("<d",bad,24,float("nan"))
        with self.assertRaisesRegex(ValueError,"Nonfinite"):
            runtime.render_camera(bad)
        self.assertEqual(runtime.sent,[])


if __name__=="__main__":
    unittest.main()
