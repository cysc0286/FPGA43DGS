"""Pinhole calibration and rectification into the centered frozen-camera ABI."""
import argparse
import json
from pathlib import Path

import numpy as np

from common import save, sha


class Calibration:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.record = json.loads(self.path.read_text(encoding="utf-8"))
        if self.record.get("schema") != "uvc_pinhole_v1":
            raise ValueError("Expected uvc_pinhole_v1 calibration")
        self.size = tuple(self.record["image_size"])
        self.k = np.asarray(self.record["K"], dtype=np.float64)
        self.d = np.asarray(self.record["D"], dtype=np.float64).reshape(-1)
        if (len(self.size) != 2 or any(type(v) is not int or v < 64 for v in self.size)
                or self.k.shape != (3, 3) or not np.isfinite(self.k).all()
                or not np.isfinite(self.d).all() or self.d.size not in (4, 5, 8, 12, 14)
                or min(self.k[0, 0], self.k[1, 1]) <= 0
                or not np.allclose(self.k[2], [0, 0, 1])
                or abs(self.k[0, 1]) > 1e-10 or abs(self.k[1, 0]) > 1e-10):
            raise ValueError("Invalid pinhole calibration")
        self.digest = sha(self.path)
        self.maps = None
        self.intrinsic = None

    def rectify(self, frame, width):
        import cv2  # Board startup must load Torch before OpenCV/OpenMP.
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            raise ValueError(f"Calibration mode {self.size} differs from capture {(w, h)}; recalibrate")
        scale = min(1., width / w)
        size = (round(w * scale), round(h * scale))
        if self.maps is None:
            # cv.resize uses half-pixel centers, not a simple principal-point scale.
            sx, sy = size[0]/w, size[1]/h
            affine = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]])
            scaled = affine @ self.k
            self.intrinsic, _ = cv2.getOptimalNewCameraMatrix(
                scaled, self.d, size, 0, size, centerPrincipalPoint=True)
            self.maps = cv2.initUndistortRectifyMap(scaled, self.d, None,
                self.intrinsic, size, cv2.CV_32FC1)
            self.output_size = size
        if size != self.output_size:
            raise ValueError("Rectification dimensions changed")
        resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        return cv2.remap(resized, *self.maps, interpolation=cv2.INTER_LINEAR)

    def metadata(self):
        return dict(kind="calibrated_centered_rectification", schema="uvc_pinhole_v1",
                    source=str(self.path), sha256=self.digest, source_image_size=list(self.size),
                    source_intrinsic=self.k.tolist(), source_distortion=self.d.tolist(),
                    intrinsic=self.intrinsic.tolist(), distortion="rectified to zero",
                    image_size=list(self.output_size))


def calibrate(images, pattern, square_mm, out):
    """Calibrate from a directory of fixed-mode images; never open a camera."""
    import cv2
    if min(pattern) < 3 or not np.isfinite(square_mm) or square_mm <= 0:
        raise ValueError("Invalid inner-corner grid or square size")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    object_grid = np.zeros((pattern[0]*pattern[1], 3), np.float32)
    object_grid[:, :2] = np.mgrid[:pattern[0], :pattern[1]].T.reshape(-1, 2)*square_mm
    objects, points, names = [], [], []
    size = None
    for path in sorted(Path(images).iterdir()):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot decode calibration image: {path}")
        current = image.shape[::-1]
        if size is not None and current != size:
            raise ValueError("Calibration images mix capture modes")
        size = current
        found, corners = cv2.findChessboardCornersSB(image, pattern)
        if found:
            objects.append(object_grid.copy())
            points.append(corners)
            names.append(dict(path=str(path.resolve()), sha256=sha(path)))
    if len(points) < 10:
        raise ValueError("Need at least 10 detected boards; recommend 20–30 varied views")
    rms, k, d, rotations, translations = cv2.calibrateCamera(objects, points, size, None, None)
    errors = []
    for obj, pixel, r, t in zip(objects, points, rotations, translations):
        projected, _ = cv2.projectPoints(obj, r, t, k, d)
        errors.append(float(np.sqrt(np.mean(np.sum((projected-pixel)**2, axis=2)))))
    record = dict(schema="uvc_pinhole_v1", image_size=list(size), K=k.tolist(), D=d.ravel().tolist(),
                  rms_px=float(rms), per_view_rms_px=errors, pattern_inner_corners=list(pattern),
                  square_mm=square_mm, images=names,
                  validation_scope="fit images only; inspect independent images and edge residuals")
    # Do not publish an unusable fit; quality still needs independent validation.
    if not np.isfinite(rms) or not np.isfinite(k).all() or not np.isfinite(d).all():
        raise ValueError("Nonfinite calibration fit")
    out.parent.mkdir(parents=True, exist_ok=True)
    save(out, record)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pattern", type=int, nargs=2, default=[9, 6], help="INNER corners")
    p.add_argument("--square-mm", type=float, required=True)
    a = p.parse_args(argv)
    result = calibrate(a.images, tuple(a.pattern), a.square_mm, a.out)
    print(json.dumps(dict(images=len(result["images"]), rms_px=result["rms_px"], out=str(a.out))))


if __name__ == "__main__":
    main()
