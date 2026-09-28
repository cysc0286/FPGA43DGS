"""Read real SIFT descriptors from an existing COLMAP DB without modifying it."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--database", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    with sqlite3.connect(a.database.resolve().as_uri()+"?mode=ro", uri=True) as db:
        images = db.execute("SELECT image_id,name FROM images ORDER BY name").fetchall()
        if len(images) < 2:
            raise ValueError("Need at least two images")
        ids = sorted(set([0, max(0, len(images)//2-1), len(images)-2]))
        cases = []
        for i in ids:
            entries = images[i:i+2]
            arrays = []
            for image_id, _ in entries:
                dtype, rows, cols, data = db.execute("SELECT type,rows,cols,data FROM descriptors WHERE image_id=?", (image_id,)).fetchone()
                if dtype != 0 or cols != 128 or len(data) != rows*128:
                    raise ValueError("Expected COLMAP 4.2 uint8 SIFT descriptor table")
                arrays.append(np.frombuffer(data, np.uint8).reshape(rows, cols).copy())
            name = "pair_%03d_%03d" % (i, i+1)
            np.savez(a.output/(name+".npz"), a=arrays[0], b=arrays[1])
            cases.append({"name": name, "image_ids": [x[0] for x in entries], "images": [x[1] for x in entries],
                          "rows": [len(x) for x in arrays], "descriptor_pairs": len(arrays[0])*len(arrays[1]),
                          "npz_sha256": hashlib.sha256((a.output/(name+".npz")).read_bytes()).hexdigest()})
    manifest = {"database": str(a.database.resolve()), "database_sha256": hashlib.sha256(a.database.read_bytes()).hexdigest(),
                "schema": "COLMAP 4.2 SIFT uint8", "cases": cases, "source_modified": False}
    (a.output/"manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
