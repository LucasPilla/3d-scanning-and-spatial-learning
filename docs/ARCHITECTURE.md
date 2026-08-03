# System Architecture & Model Design

This document details the neural network architecture, diffusion sampling pipeline, spatial scene encoding, motion representation, and canonical frame transformations in `nymeria_plus_motion`.

---

## 1. High-Level Pipeline

The system generates 3D human body motion conditioned on **text prompts**, **3D spatial scene crops**, and **3D target goals**.

```
                         ┌───────────────────────┐
                         │ Text Prompt (CLIP)    │
                         └───────────┬───────────┘
                                     │
┌───────────────────────┐            ▼            ┌───────────────────────┐
│ Noisy Motion [B,T,148]│──► Motion Transformer ◄─│ Spatial Scene Tokens  │
└───────────────────────┘            ▲            └───────────────────────┘
                                     │
                         ┌───────────┴───────────┐
                         │ 3D Goal Vector [B, 3] │
                         └───────────────────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Denoised 148-D Motion │
                         └───────────────────────┘
```

---

## 2. Motion Representation & Canonical Anchor Space

To achieve translation- and rotation-invariant motion generation:

### A. 148-Dimensional Motion Feature Vector ($\mathbf{x} \in \mathbb{R}^{148}$)
At each frame $t$, the motion pose is represented as:
* **Root Linear Velocity (`[0:3]`):** $\mathbf{v}_{\text{root}} = (\Delta x, \Delta y, \Delta z)$ in the local facing frame.
* **Facing Yaw Angular Velocity (`[3:5]`):** $(\cos \Delta\psi, \sin \Delta\psi)$ relative to the previous frame.
* **24-Joint Rotations (`[5:149]`):** 24 body joint rotations expressed as **6D continuous rotation matrices** ($24 \times 6 = 144$ dimensions), avoiding gimbal lock or quaternion discontinuities.

---

### B. Canonical Anchor Frame Transformation

All history frames (32 frames) and future prediction windows (16 frames) are transformed relative to the **anchor frame** ($t_{\text{anchor}} = 31$):

1. **Anchor Pelvis Origin (0, 0, 0):**
   The origin $(0, 0, 0)$ of the canonical frame is defined directly at the 3D pelvis position at the anchor frame ($t_{\text{anchor}} = 31$).
   * Relative to $(0, 0, 0)$, standing feet sit at $Z \approx -0.92\text{m}$.
   * The 3D scene crop `[-1.0, 1.0, -1.0, 1.0, -1.2, 0.8]` is centered at $(0, 0, 0)$ (reaching $-1.2\text{m}$ down to cover the floor and $+0.8\text{m}$ up to cover head height).
   * Target goals $[dx, dy, dz]$ are expressed as 3D spatial displacements relative to $(0, 0, 0)$.
2. **Facing Yaw Rotation ($\psi_{\text{anchor}}$):**
   Derived from the 3D hip vector ($J_{\text{R\_Hip}} - J_{\text{L\_Hip}}$). All joint positions, root deltas, spatial scene crops, and 3D target goals are rotated by:
   $$R_{\text{canonical}} = \text{YawRotation}(-\psi_{\text{anchor}})$$
3. **Rollout Stitching:**
   By normalizing every window to its local anchor frame, auto-regressive window rollouts stitch seamlessly across arbitrary world trajectories during long-sequence generation.

---

## 3. Core Network Modules

### A. Motion Transformer (`src/models/transformer.py`)

The backbone model is a sequence-to-sequence Transformer operating on 148-dimensional motion features. Each layer applies sequential cross-attention branches:

1. **Motion Self-Attention:** Temporal self-attention across 48 motion frames (32 history + 16 window).
2. **Scene Cross-Attention:** Attends to 64 spatial scene tokens extracted by the 3D CNN.
3. **Text Cross-Attention:** Attends to text tokens (CLIP/BERT embeddings).
4. **Goal Cross-Attention:** Attends to 3D local goal position vectors $[dx, dy, dz]$. Single-token cross-attention uses a learned `null_token` prepended to prevent Softmax collapsing.
5. **Feed-Forward Network (FFN):** GELU activation with residual connections.

---

### B. Motion Diffusion Engine (`src/models/diffusion.py`)

* **Noise Schedule:** 50-timestep cosine beta schedule (`cosine_beta_schedule`).
* **Prediction Target:** Direct clean motion prediction ($\mathbf{x}_0$-parameterization) rather than $\epsilon$-noise prediction.
* **Classifier-Free Guidance (CFG):** Supports joint unconditional dropout for text, scene, and goal.
* **Loss Function:** Combined weighted MSE loss:
  $$\mathcal{L}_{\text{total}} = w_{\text{feature}} \cdot \mathcal{L}_{\text{feature}} + w_{\text{position}} \cdot \mathcal{L}_{\text{position}}$$
  * $\mathcal{L}_{\text{feature}}$: MSE loss on normalized 148-D features.
  * $\mathcal{L}_{\text{position}}$: Differentiable Forward Kinematics joint position MSE loss in meters.

---

### C. 3D Scene Encoder (`src/models/scene_encoder.py`)

* **Backbone:** `CNN3DSceneEncoder`
* **Input:** $32 \times 32 \times 32$ 3D occupancy / TSDF grid ($6.25\text{ cm}$ resolution).
* **Layer-by-Layer Feature Grid Downsampling:**
  $$32 \times 32 \times 32 \longrightarrow 16 \times 16 \times 16 \longrightarrow 8 \times 8 \times 8 \longrightarrow 4 \times 4 \times 4$$
* **Layer-by-Layer 3D Receptive Field Expansion:**
  $$1 \times 1 \times 1 \longrightarrow 3 \times 3 \times 3 \longrightarrow 7 \times 7 \times 7 \longrightarrow 15 \times 15 \times 15 \text{ voxels}$$
  $$(6.25\text{ cm} \longrightarrow 18.75\text{ cm} \longrightarrow 43.75\text{ cm} \longrightarrow 93.75\text{ cm})$$

* **Output:** **64 spatial feature tokens** ($4 \times 4 \times 4$ volumetric grid, 256-D per token, $50\text{ cm}$ resolution per token).
* **3D Receptive Field:** Each of the 64 spatial tokens has an effective 3D receptive field of **$15 \times 15 \times 15$ input voxels** ($93.75\text{ cm} \approx 0.94\text{ meters}$ in physical space). This wide receptive field provides rich contextual overlap between adjacent 3D spatial tokens, allowing the model to reason about connected 3D geometry (e.g., chair seats connected to legs and table tops).

---

### D. Differentiable Kinematics (`src/utils/kinematics.py`)
* **Forward Kinematics (`features_to_joints`):**
  Reconstructs 3D joint positions $\mathbf{J} \in \mathbb{R}^{24 \times 3}$ in metric space from 6D rotations and participant bone offsets:
  $$J_{\text{child}} = J_{\text{parent}} + R_{\text{parent}} \cdot O_{\text{child}}$$
