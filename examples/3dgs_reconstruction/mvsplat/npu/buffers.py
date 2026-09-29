"""One input/output arena for serialized partitions, with an explicit old-policy control.

This is file-backed IPC, not DMA allocation. The caller holds the runtime lock
until the returned output has been copied into independently owned memory.
"""
from pathlib import Path
import math
import numpy as np


def buffer_plan(parts, policy="shared"):
    if policy not in ("shared", "per_partition") or not parts:
        raise ValueError("Expected nonempty partitions and a known buffer policy")
    files, views = {}, {}
    for name, part in parts.items():
        views[name] = {}
        for kind in ("input", "output"):
            shape = part[kind]["shape"]
            if len(shape) != 4 or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError("Buffer shape must be positive static 4D")
            size = math.prod(shape)*4
            filename = (kind if policy == "shared" else name+"_"+kind)+".bin"
            files[filename] = max(files.get(filename, 0), size)
            views[name][kind] = dict(file=filename, shape=shape)
    return dict(schema="mvsplat_buffers_v2", policy=policy, files=files, views=views,
                allocated_bytes=sum(files.values()),
                logical_io_bytes=sum(math.prod(p[k]["shape"])*4
                                     for p in parts.values() for k in ("input", "output")))


class MappedBuffers:
    def __init__(self, root, parts, plan, *, create=False, limit_bytes=128*1024**2):
        # Reconstruct the entire descriptor before touching a file. This prevents
        # stale shapes, overlapping input/output files and path substitutions.
        if plan != buffer_plan(parts, plan.get("policy")):
            raise ValueError("Buffer descriptor differs from validated partition ABI")
        if limit_bytes <= 0 or plan["allocated_bytes"] > limit_bytes:
            raise ValueError("IPC buffer allocation exceeds configured byte budget")
        self.maps, self.views = {}, {}
        self.plan = plan
        root = Path(root).resolve()
        try:
            for filename, size in plan["files"].items():
                path = root/filename
                if path.is_symlink() or path.resolve().parent != root:
                    raise ValueError("Buffer path escapes worker directory")
                if create:
                    with path.open("xb") as stream:
                        stream.truncate(size)
                elif path.stat().st_size != size:
                    raise ValueError("Buffer file size mismatch: "+filename)
                self.maps[filename] = np.memmap(path, mode="r+", dtype=np.float32, shape=(size//4,))
            for name, specs in plan["views"].items():
                self.views[name] = [self.maps[specs[k]["file"]][:math.prod(specs[k]["shape"])].reshape(
                    specs[k]["shape"]) for k in ("input", "output")]
        except BaseException:
            self.close()
            raise

    def close(self):
        self.views.clear()
        for array in self.maps.values():
            array._mmap.close()
        self.maps.clear()
