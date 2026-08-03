# NymeriaPlus Dataset & Skeleton Guide

This document describes the NymeriaPlus dataset, skeleton topology, frame transformations, egocentric annotations, and dataset curation filters.

---

## 1. Dataset Overview

Nymeria / NymeriaPlus is a large-scale multimodal dataset of 3D human motion captured in diverse real-world environments.
* **Total Recordings:** 707 full-length motion captures across 39 distinct locations.
* **Annotations:** Over 149,000 atomic activity narrations (e.g., "walking towards table", "sitting down on chair", "picking up object").
* **Sensors:** Egocentric multi-camera video, IMU, and high-fidelity 3D human body tracking.

---

## 2. Dataset Curation & Filtering Pipeline

To convert raw Nymeria recordings into high-quality training sequences, the preprocessing pipeline applies strict quality filters:

1. **Recording Integrity Validation:**
   * Validates existence of complete 3D SMPL pose data (`xdata_smpl_neutral.npz`), metadata (`metadata.json`), 3D object geometry (`objects/shaper` / `objects/boxy`), and narration annotations (`narration/atomic_action.csv`).
2. **Annotation Duration Filtering:**
   * Filters out short or corrupted text annotations where the duration is less than the required motion prediction window ($\Delta t < 16\text{ frames}$ at $10\text{ fps}$).
3. **Text Normalization & Cleansing:**
   * Normalizes narrator shorthand tags (replacing `C` with `"the person"`), lowercases text, and strips whitespace artifacts.
4. **Kinematic Skeleton Stabilization:**
   * Averages frame-wise SMPL shape parameters ($\beta$) per participant to yield a single, time-invariant skeleton topology per recording. This eliminates unphysical bone length jitter during forward kinematics.
5. **3D Scene Mesh Selection:**
   * Prioritizes high-confidence ShapeR 3D object reconstructions by sorting by variant score, falling back to Boxy 3D bounding box primitives for unmodeled objects.

---

## 3. Skeleton & Kinematic Hierarchy

The human body is modeled using the standard 24-joint SMPL skeleton topology:

```
[0] Pelvis (Root)
 ├── [1] L_Hip ── [4] L_Knee ── [7] L_Ankle ── [10] L_Foot
 ├── [2] R_Hip ── [5] R_Knee ── [8] R_Ankle ── [11] R_Foot
 └── [3] Spine1 ── [6] Spine2 ── [9] Spine3
      ├── [12] Neck ── [15] Head
      ├── [13] L_Collar ── [16] L_Shoulder ── [18] L_Elbow ── [20] L_Wrist ── [22] L_Hand
      └── [14] R_Collar ── [17] R_Shoulder ── [19] R_Elbow ── [21] R_Wrist ── [23] R_Hand
```

### Parent Indices (`SMPL_PARENTS`)
$$(-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)$$

---

## 4. Canonical Anchor Frame & Facing Yaw

To ensure motion prediction is invariant to global position and world orientation:

1. **Anchor Frame Alignment:**
   The motion history (32 frames) ends at the **anchor frame** ($t=31$).
2. **Facing Yaw ($\psi$):**
   Derived from the 3D hip vector ($J_{\text{R\_Hip}} - J_{\text{L\_Hip}}$). It defines the character's forward heading direction in the horizontal plane ($X, Y$).
3. **Anchor Transformation:**
   All future motion (16 window frames) and 3D scene crops are rotated into the anchor's local canonical frame:
   $$R_{\text{canonical}} = \text{YawRotation}(-\psi_{\text{anchor}})$$

---

## 5. Participant Rest Bone Offsets

Different individuals have different body heights and limb proportions.
* Rest bone offsets $\mathbf{O} \in \mathbb{R}^{24 \times 3}$ are computed per participant during preprocessing by evaluating SMPL shape parameters ($\beta$).
* **Row 0** stores the rest root height ($\sim 0.92\text{m}$ for standing pelvis).
* **Rows 1..23** store relative parent-to-child 3D vector offsets in rest pose.
