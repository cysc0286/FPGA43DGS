"""CPU SfM, undistortion and renderer-compatible intrinsics."""
import math
import time
import cv2
import numpy as np
from ..common import save
from ..contracts import FrameSet, PoseSet

def sfm(a):
    import pycolmap as p
    root = a.run
    FrameSet.load(root)
    db, imgs = root / "database.db", root / "frames"
    timings = {}
    start = time.perf_counter()
    ext = p.FeatureExtractionOptions()
    ext.use_gpu = False
    ext.num_threads = a.threads
    ext.max_image_size = a.width
    ext.sift.max_num_features = a.max_features
    p.extract_features(db, imgs, camera_mode=p.CameraMode.SINGLE,
                      reader_options=p.ImageReaderOptions(camera_model="SIMPLE_RADIAL"),
                      extraction_options=ext, device=p.Device.cpu)
    timings["features_s"] = time.perf_counter() - start
    start = time.perf_counter()
    matching = p.FeatureMatchingOptions()
    matching.use_gpu = False
    matching.num_threads = a.threads
    pairing = p.SequentialPairingOptions()
    pairing.overlap = a.overlap
    pairing.loop_detection = False
    p.match_sequential(db, matching_options=matching, pairing_options=pairing, device=p.Device.cpu)
    timings["matching_s"] = time.perf_counter() - start
    start = time.perf_counter()
    opts = p.IncrementalPipelineOptions()
    opts.num_threads = a.threads
    opts.random_seed = 42
    opts.ba_use_gpu = False
    opts.min_model_size = 5
    # Short video has a narrow baseline. This is a geometry setting, not supplied poses.
    opts.mapper.init_min_tri_angle = 4.0
    models = p.incremental_mapping(db, imgs, root / "sparse", options=opts)
    timings["mapping_s"] = time.perf_counter() - start
    if not models:
        raise RuntimeError("COLMAP could not recover a model; inspect video overlap/parallax")
    model_id, rec = max(models.items(), key=lambda item: item[1].num_reg_images())
    counts = {str(k): v.num_reg_images() for k, v in models.items()}
    registered = rec.num_reg_images()
    expected = len(list(imgs.glob("*.png")))
    save(root / "sfm.json", {"version": p.__version__, "has_cuda": p.has_cuda,
         "device": "cpu", "registered": registered, "input_images": expected,
         "registered_fraction": registered / expected, "points3D": rec.num_points3D(),
         "mean_reprojection_error_px": rec.compute_mean_reprojection_error(),
         "mean_track_length": rec.compute_mean_track_length(), "models": counts,
         "selected_model": int(model_id), "timings": timings})
    if registered < max(5, math.ceil(.8 * expected)):
        raise RuntimeError("Less than 80% of selected frames registered; do not call the chain validated")
    start = time.perf_counter()
    p.undistort_images(root / "undistorted", root / "sparse" / str(model_id), imgs,
                      undistort_options=p.UndistortCameraOptions(max_image_size=a.width), num_threads=a.threads)
    timings["undistort_s"] = time.perf_counter() - start
    # Train at the EXACT dimensions/intrinsics consumed by frozen FLCAM001.
    # Its principal point is (W-1)/2,(H-1)/2; do not silently discard cx,cy.
    und = p.Reconstruction(root / "undistorted" / "sparse")
    project = root / "project"
    (project / "images").mkdir(parents=True)
    (project / "sparse" / "0").mkdir(parents=True)
    transforms = {}
    for cid, cam in und.cameras.items():
        w = a.render_width
        h = round(cam.height * w / cam.width)
        scale = w / cam.width
        fx, fy = cam.focal_length_x * scale, cam.focal_length_y * scale
        cx, cy = (w - 1) / 2, (h - 1) / 2
        tx, ty = cx - scale * cam.principal_point_x, cy - scale * cam.principal_point_y
        mat = np.array([[scale, 0., tx], [0., scale, ty]], dtype=np.float64)
        transforms[cid] = {"affine": mat, "w": w, "h": h}
        cam.width, cam.height = w, h
        cam.params = np.array([fx, fy, cx, cy])
    for im in und.images.values():
        tr = transforms[im.camera_id]
        rgb = cv2.imread(str(root / "undistorted" / "images" / im.name))
        out = cv2.warpAffine(rgb, tr["affine"], (tr["w"], tr["h"]), flags=cv2.INTER_LINEAR)
        if not cv2.imwrite(str(project / "images" / im.name), out):
            raise RuntimeError("Failed to save adapted image: " + im.name)
        for point in im.points2D:
            point.xy = tr["affine"][:, :2] @ point.xy + tr["affine"][:, 2]
    und.write(project / "sparse" / "0")
    names = sorted(im.name for im in und.images.values())
    heldout = names[len(names) // 2]
    save(root / "project.json", {"heldout_image": heldout,
         "heldout_scope": "excluded from Gaussian optimization, included in SfM pose/initial-point estimation",
         "cameras": {str(k): {"width": v.width, "height": v.height, "params": v.params.tolist()} for k, v in und.cameras.items()},
         "image_count": len(names), "pixel_convention": "FLCAM001: principal point (W-1)/2,(H-1)/2",
         "timings": timings})
    return PoseSet.load(root)
