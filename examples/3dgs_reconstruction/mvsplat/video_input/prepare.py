"""Compatibility import for the scene-preparation coordinator.

New code imports scene_preparation; decoding and pose algorithms have separate owners.
"""
from scene_preparation import main
from video_input.decode import frame_indices, decode_selected
from pose_estimation.geometry import mutual_ratio_matches, estimate_pair, estimate_target

if __name__ == "__main__":
    main()
