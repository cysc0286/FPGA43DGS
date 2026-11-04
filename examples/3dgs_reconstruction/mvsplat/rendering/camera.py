"""Camera transforms shared by native rendering and web controls."""
import numpy as np


def camera_view(base, controls, focus_distance, *, capture_center=None, default_mode='orbit',
                allowed_modes=None):
    """Turn at a declared capture centre, or orbit the original scene pivot.

    The capture centre comes from the input manifest, never from Gaussian means.
    Angles are relative to the reference camera axes, not a geographic bearing.
    """
    allowed = {"mode", "yaw", "pitch", "distance", "pan_x", "pan_y", "request_id"}
    if not isinstance(controls, dict) or set(controls)-allowed:
        raise ValueError("Unknown camera controls")
    mode = controls.get('mode', default_mode)
    if mode not in ('look', 'orbit'):
        raise ValueError('Invalid camera mode')
    if allowed_modes is not None and mode not in allowed_modes:
        raise ValueError('Camera mode is unavailable for this capture')
    if mode == 'look' and {'distance', 'pan_x', 'pan_y'} & set(controls):
        raise ValueError('Fixed-centre view does not allow translation or zoom controls')
    limits = dict(yaw=(-180, 180, 0), pitch=(-85, 85, 0), distance=(.6, 1.5, 1),
                  pan_x=(-.25, .25, 0), pan_y=(-.25, .25, 0))
    values = {}
    for key, (low, high, default) in limits.items():
        value = controls.get(key, default)
        if type(value) not in (int, float) or not np.isfinite(value) or not low <= value <= high:
            raise ValueError("Invalid camera control: " + key)
        values[key] = value
    request_id = controls.get("request_id", 0)
    if type(request_id) is not int or not 0 <= request_id <= 2**53-1:
        raise ValueError("Invalid request_id")
    if mode == 'look':
        center = np.asarray(capture_center, dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('Fixed-centre view requires an explicit capture_center')
    original = np.asarray(base["c2w"], dtype=np.float64)
    if (original.shape != (4, 4) or not np.isfinite(original).all()
            or not np.allclose(original[:3, :3].T @ original[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(original[:3, :3]), 1, atol=1e-5)):
        raise ValueError('Invalid reference camera')
    right, down, forward = original[:3, :3].T
    pivot = original[:3, 3] + forward*focus_distance
    pivot += focus_distance*(right*values["pan_x"] + down*values["pan_y"])
    def rotation(axis, degrees):
        angle = np.radians(degrees)
        x, y, z = axis
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)
    turn = rotation(down, values["yaw"]) @ rotation(right, values["pitch"])
    axes = turn @ original[:3, :3]
    position = center if mode == 'look' else pivot - axes[:, 2]*focus_distance*values['distance']
    c2w = np.eye(4)
    c2w[:3, :3] = axes
    c2w[:3, 3] = position
    return dict(base, c2w=c2w.tolist())


def orbit_view(base, controls, focus_distance):
    """Compatibility entry point for the scene-exchange orbit controller."""
    return camera_view(base, controls, focus_distance)
