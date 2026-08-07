# NymeriaPlus Dataset

This document describes the NymeriaPlus dataset, the curation process and
exploratory data analysis (EDA) statistics of the filtered version used for
this research.

## 1. Original NymeriaPlus Dataset

NymeriaPlus is a large-scale multimodal dataset capturing 3D human motion in real-world environments. It includes egocentric multi-camera video, IMU data, high-fidelity 3D human body tracking (SMPL), 3D scene meshes, and text annotations describing atomic actions. The dataset features diverse environments and complex object interactions.

## 2. Curation and Filtering

To ensure high-quality training sequences, we apply strict filters in two stages:

### A. Download Stage (`download.py`)

When downloading the dataset, we prune sequences that are missing essential modalities:
*   **Artifact Integrity Check:** We only retain sequences that possess complete 3D SMPL pose data, object bounding boxes, 3D object meshes, and atomic action narrations.
*   **Downsampling:** To normalize motion speed across the dataset, SMPL joint streams are downsampled to a fixed target framerate (e.g., 10 FPS).

### B. Preprocessing Stage (`preprocess.py`)

The downloaded raw data is transformed into training tensors with the following logic:
1.  **Duration Filtering:** We discard text annotations shorter than a required threshold (e.g., < 10 frames or 1 second) or excessively long summary narrations (e.g., > 60 frames), leaving only precise, atomic actions.
2.  **Text Normalization:** Narrator tags (like "C") are replaced with natural language ("the person"), and text is lowercased and stripped of whitespace artifacts.
3.  **Kinematic Stabilization:** We extract a single, time-invariant body shape (beta parameters) per participant by averaging across all their frames. We then run forward kinematics via SMPL once to generate the 3D joint positions (`joints.npy`). This avoids bone-length jitter caused by frame-to-frame shape estimation noise.
4.  **Scene Voxelization:** 3D environment models (ShapeR meshes and Boxy primitives) are rasterized into a discrete 3D boolean occupancy grid (`scene_occupancy.npz`) at 5cm resolution, bounding only the area where the person actually moves.

## 3. Resulting Dataset (EDA)

After applying the filtering and preprocessing steps above, the usable dataset size is:

*   **Total Sequences:** 707
*   **Total Frames:** 6,550,431
*   **Total Hours of Motion (at 10 FPS):** 181.96 hours
*   **Total Atomic Annotations:** 144,356

## 4. Train / Validation / Test Split

The 707 filtered sequences are split at the **sequence level**, not per-window or per-annotation: a fixed assignment in `splits.json` puts an entire recording — the participant, the room, and every one of its annotations — into exactly one split. This keeps test-split rooms and motions genuinely unseen during training, rather than just held-out sub-windows of an otherwise-seen sequence.

*   **Train:** 548 sequences
*   **Validation:** 83 sequences
*   **Test:** 76 sequences

Only the train split feeds `normalization.npz`'s per-channel statistics (`compute_normalization` in `preprocess.py`), so validation and test motion never leaks into the statistics the model is normalized against.

## 5. Dataset Directory Structure

The preprocessed dataset must follow this layout:

```text
data/nymeriaplus/
├── normalization.npz          # Dataset feature mean/std statistics
└── sequences/
    ├── <sequence_name_01>/
    │   ├── joints.npy          # [Frames, 24, 3] World-space 3D joint locations
    │   ├── yaw.npy             # [Frames] Root facing yaw angles (radians)
    │   ├── scene_occupancy.npz # Compressed occupancy grid
    │   ├── scene_bounds.npy    # Global 3D world bounding box [[Xmin...],[Xmax...]]
    │   └── annotations.jsonl   # Text annotations
    └── ...
```

## 6. Expected Data Layout

*   **`joints.npy`**: A float32 or float64 numpy array of shape `[Num_Frames, 24, 3]`. It contains the metric 3D position (in meters) of 24 SMPL joints for every frame in the recording. No rotations are stored.
*   **`yaw.npy`**: A float32 numpy array of shape `[Num_Frames]`. It represents the forward-facing angle (heading) of the person in radians on the horizontal plane.
*   **`scene_occupancy.npz`**: A boolean numpy array saved as a compressed archive, representing a 3D voxel grid of the environment's static geometry. True means occupied, False means empty.
*   **`scene_bounds.npy`**: A float32 numpy array of shape `[2, 3]`. The first row is the minimum `[X, Y, Z]` of the scene occupancy grid, and the second row is the maximum. The grid resolution is inferred from these bounds and the occupancy array shape.
*   **`annotations.jsonl`**: A JSON Lines file where each line is a JSON object with at least `start_frame` (int), `end_frame` (int), and `text` (string).
*   **`normalization.npz`**: Contains `mean` and `std` for the 72-D motion features. The goal has no statistics of its own — it is the window's last pelvis position, so it reuses channels `[0:3]` of these.
