# nymeria_plus_motion

Text-, Scene-, and Goal-Conditioned Human Motion Diffusion Model on Nymeria Plus.

---

## 1. Architecture Overview

`nymeria_plus_motion` implements a sequence-to-sequence Motion Transformer denoiser for 3D human body motion diffusion, conditioned on natural language text, 3D voxel scene geometry, and 3D pelvis target goals.

```
                          ┌───────────────────────────┐
                          │ Text Prompt (CLIP/BERT/T5)│
                          └─────────────┬─────────────┘
                                        │
 ┌────────────────────────┐             ▼             ┌───────────────────────┐
 │ Noisy Motion [B,T,148] │──► Motion Transformer ◄───│ 3D Voxel Scene Tokens │
 └────────────────────────┘             ▲             └───────────────────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          │ 3D Goal Vector [B, 3]     │
                          └───────────────────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │ Denoised 148-D Motion     │
                          └───────────────────────────┘
```

### Key Components

- **148-D Motion Representation:** Each frame is encoded invariant to world location:
  - `[0:3]` Root translation delta ($\Delta x, \Delta y, \Delta z$) in the previous frame's facing yaw frame.
  - `[3:4]` Root facing yaw delta ($\Delta \psi$).
  - `[4:10]` Root residual orientation (6D continuous rotation matrix).
  - `[10:148]` 23 SMPL body joint local rotations ($23 \times 6\text{D} = 138$ dimensions).
- **Conditioning Modalities:**
  - **Text:** Encoded via CLIP, DistilBERT, or T5 backbones.
  - **Scene Geometry:** Encoded from $32\times32\times32$ voxel occupancy grids using a 3D CNN (`CNN3DSceneEncoder`) or 2D ViT.
  - **Goal Target:** 3D pelvis displacement $[dx, dy, dz]$ relative to the anchor frame.
- **Motion Diffusion Engine:** 50-step cosine DDPM schedule predicting clean motion ($x_0$-parameterization) with Classifier-Free Guidance (CFG) across text, scene, and goal gates.

For in-depth mathematical details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 2. Installation Guide

### Prerequisites
- Python >= 3.9, < 3.12
- PyTorch >= 2.8

### Installation

Clone the repository and install in editable mode:

```bash
cd nymeria_plus_motion
pip install -e .
```

To install optional dependencies for data preprocessing or dataset downloading:

```bash
# For dataset download utilities
pip install -e ".[download]"

# For preprocessing (SciPy, SMPLX, Trimesh)
pip install -e ".[preprocess]"
```

---

## 3. Directory Structure Explanation

```
nymeria_plus_motion/
├── configs/                  # Yaml configuration files
│   ├── default.yaml          # Master configuration with default hyperparameters
│   └── overfit_test.yaml     # Fast single-batch overfit test configuration
├── demo/                     # Interactive demo tool
│   ├── demo.py               # Real-time interactive rollout visualizer
│   └── scenes.py             # Empty-scene occupancy grid and skeleton helpers
├── docs/                     # Detailed technical and dataset documentation
│   ├── ARCHITECTURE.md       # Neural network architecture & mathematical formulation
│   ├── DATASET.md            # Dataset specification and array layout
│   └── NYMERIAPLUS.md        # Nymeria Plus data split and processing notes
├── pyproject.toml            # Project dependencies and packaging setup
├── README.md                 # Project guide (this file)
└── src/                      # Source package
    ├── cache.py              # Text feature pre-caching script
    ├── config.py             # Configuration loader, defaults, and builder functions
    ├── inference.py          # Auto-regressive rollout state & generation engine
    ├── train.py              # Model training script
    ├── datasets/             # Dataset implementations
    │   └── nymeriaplus/      # Nymeria Plus loader, preprocessing & splits
    ├── models/               # Model architectures
    │   ├── diffusion.py      # DDPM diffusion wrapper & CFG sampling
    │   ├── scene_encoder.py  # 3D CNN and ViT scene occupancy encoders
    │   ├── text_encoder.py   # CLIP, DistilBERT, and T5 frozen text towers
    │   └── transformer.py    # Motion Transformer denoiser
    └── utils/                # Kinematic, geometry, scene, and statistics helpers
        ├── checkpoint.py     # Resumable model checkpoint saving & loading
        ├── geometry.py       # Local-to-world transforms and padding helpers
        ├── kinematics.py     # Forward kinematics, SMPL tree, and 6D rotation math
        ├── scene.py          # Local voxel grid crop & sampling utilities
        └── statistics.py     # Normalization statistics loader and standard-scorer
```

---

## 4. Dataset Caching & Training Workflows

### Step 1: Pre-cache Text Features (Recommended)
Pre-caching text embeddings speeds up training by avoiding redundant text tower evaluations during epochs:

```bash
for encoder in clip bert t5; do
  python -m src.cache --dataset data/nymeriaplus --text-encoder $encoder --split all --max-text-tokens 64
done
```

### Step 2: Train the Motion Diffusion Model

Train using a configuration file:

```bash
python -m src.train --config configs/default.yaml --name my_experiment --workers 4
```

Checkpoint files and TensorBoard logs will be written to `runs/my_experiment/`.

### Step 3: Inspect Generated Motion

Run the interactive browser demo against a trained checkpoint:

```bash
python -m demo.demo \
  --config runs/my_experiment/config.yaml \
  --checkpoint runs/my_experiment/checkpoints/my_experiment_epoch300.pth
```

---

## 5. Quick-Start Command Reference

| Task | Command |
| :--- | :--- |
| **Install Package** | `pip install -e .` |
| **Cache Text Features** | `python -m src.cache --dataset data/nymeriaplus --text-encoder clip --split all` |
| **Run Overfit Test** | `python -m src.train --config configs/overfit_test.yaml --name overfit_run` |
| **Start Full Training** | `python -m src.train --config configs/default.yaml --name main_run --workers 4` |
| **Resume Training** | `python -m src.train --config configs/default.yaml --name main_run --resume-checkpoint runs/main_run/checkpoints/main_run_epoch050.pth` |
| **Run Interactive Demo** | `python -m demo.demo --config runs/main_run/config.yaml --checkpoint runs/main_run/checkpoints/main_run_epoch300.pth` |
