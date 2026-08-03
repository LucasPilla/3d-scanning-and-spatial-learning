# Dataset Specification & Preprocessing Guide

This document describes the dataset directory structure, file formats, benchmark splits, and preprocessing workflow for `nymeria_plus_motion`.

---

## 1. Processed Dataset Directory Structure

```
data/nymeriaplus/
├── normalization.npz          # Dataset feature & goal mean/std statistics
└── sequences/
    ├── <sequence_name_01>/
    │   ├── features.npy       # [Frames, 148] 148-D normalized motion features
    │   ├── joints.npy         # [Frames, 24, 3] 3D joint locations in meters
    │   ├── yaw.npy            # [Frames] Root facing yaw angles (radians)
    │   ├── bone_offsets.npy   # [24, 3] Participant rest bone offsets
    │   ├── scene_occupancy.npz# Compressed 3D boolean occupancy grid
    │   ├── scene_bounds.npy   # [2, 3] Global 3D world bounding box [[Xmin,Ymin,Zmin],[Xmax,Ymax,Zmax]]
    │   └── annotations.jsonl  # Text prompts & frame windows
```

---

## 2. File Specifications

| Filename | Shape / Format | Description |
| :--- | :--- | :--- |
| `features.npy` | `[T, 148]` float32 | Unnormalized 148-D motion features (root deltas, yaw deltas, 24x6D rotations). |
| `joints.npy` | `[T, 24, 3]` float32 | 3D joint positions in meters relative to initial sequence origin. |
| `yaw.npy` | `[T]` float32 | Facing yaw angles (radians) derived from hip orientation. |
| `bone_offsets.npy` | `[24, 3]` float32 | Participant rest joint distances relative to parent joints. |
| `scene_occupancy.npz` | `[D, H, W]` bool (compressed) | Compressed 3D occupancy grid ($22.8\times$ file size reduction). |
| `scene_bounds.npy` | `[2, 3]` float32 | World coordinates mapping $(X,Y,Z) \to$ grid indices $[i,j,k]$. |
| `annotations.jsonl` | JSON Lines | Text prompts, start/end frame indices, and optional spatial goals. |

---

## 3. Benchmark Splits (`src/datasets/nymeriaplus/splits.json`)

To prevent data leakage, dataset splits are **location-disjoint**:

* **Training Set:** 31 locations (548 sequences, $77.5\%$ of dataset).
* **Validation Set:** 4 locations (`Loc_20`, `Loc_36`, `Loc_14`, `Loc_41`, 83 sequences, $11.7\%$).
* **Testing Set:** 4 locations (`Loc_16`, `Loc_25`, `Loc_42`, `Loc_30`, 76 sequences, $10.7\%$).

---

## 4. Normalization Statistics (`normalization.npz`)

Created automatically by `preprocess.py` using **only training sequences**:
* `mean`: `[148]` float32 mean vector.
* `std`: `[148]` float32 standard deviation vector.
* `goal_mean`: `[3]` float32 mean goal displacement vector $[dx, dy, dz]$.
* `goal_std`: `[3]` float32 standard deviation goal displacement vector.

---

## 5. Running Full Preprocessing

To process raw Nymeria recordings into the processed format:

```bash
python -m src.datasets.nymeriaplus.preprocess \
  --input /path/to/nymeria_dataset/download \
  --output data/nymeriaplus \
  --smpl-male-model /path/to/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl \
  --smpl-female-model /path/to/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl \
  --overwrite
```
