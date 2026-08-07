# nymeria_plus_motion

Text-, Scene-, and Goal-Conditioned Human Motion Diffusion Model on Nymeria Plus.

## Setup

```bash
git clone <repository-url>
cd nymeria_plus_motion
pip install -e ".[download,preprocess]"
```

## Data Preparation

To download and preprocess the dataset (requires SMPL models):
```bash
# 1. Download
python -m src.datasets.nymeriaplus.download \
  --url-json /path/to/urls.json \
  --output data/nymeriaplus/download \
  --fps 10

# 2. Preprocess
python -m src.datasets.nymeriaplus.preprocess \
  --input data/nymeriaplus/download \
  --output data/nymeriaplus \
  --smpl-male-model /path/to/smpl/male.pkl \
  --smpl-female-model /path/to/smpl/female.pkl
```

## Training

```bash
# Pre-cache text features
for encoder in clip bert t5; do
  python -m src.cache --dataset data/nymeriaplus --text-encoder $encoder --split all --max-text-tokens 64
done

# Train
python -m src.train --config configs/default.yaml --name main_run --workers 4
```

## Tests

```bash
# Overfit on a single batch to test the pipeline
python -m src.train --config configs/overfit_test.yaml --name overfit_run
```

## Demo

```bash
python -m demo.demo \
  --config runs/main_run/config.yaml \
  --checkpoint runs/main_run/checkpoints/main_run_epoch300.pth
```

## Evaluation

Generate a window for every held-out test-split annotation, conditioned on
that annotation's own history, scene, text, and goal, and save it alongside
its ground truth:

```bash
python -m src.test \
  --config runs/main_run/config.yaml \
  --checkpoint runs/main_run/checkpoints/main_run_epoch300.pth \
  --output runs/main_run/test_results.npz
```

Then browse the generated windows against ground truth in an interactive
local viewer, with a search bar over the annotation text. Results are saved
in world space, so the viewer can overlay each segment's real scene geometry
(the per-object meshes/boxes, not the coarse occupancy grid the model
conditions on) by pointing `--raw-scenes` at the raw Nymeria download, same
as `demo.demo`:

```bash
python -m src.visualize \
  --results runs/main_run/test_results.npz \
  --raw-scenes /path/to/nymeria_dataset/download
```

For a pre-token-redesign checkpoint (e.g. `runs/v1`), use `src.test_legacy`
instead; it writes the same result format, so `src.visualize` works
unchanged:

```bash
python -m src.test_legacy \
  --config runs/v1/config.yaml \
  --checkpoint runs/v1/checkpoints/v1_epoch300.pth \
  --output runs/v1/test_results.npz
```

## Directory Structure

```text
nymeria_plus_motion/
├── configs/                  # Yaml configuration files
├── demo/                     # Interactive demo tool & visualizer
├── docs/                     # Detailed technical and dataset documentation
├── pyproject.toml            # Project dependencies and packaging setup
├── README.md                 # Project guide (this file)
└── src/                      # Source package
    ├── cache.py              # Text feature pre-caching script
    ├── config.py             # Configuration loader
    ├── inference.py          # Auto-regressive rollout state & generation engine
    ├── train.py              # Model training script
    ├── test.py               # Test-split generation script
    ├── test_legacy.py        # Test-split generation script for v1-era checkpoints
    ├── visualize.py          # Generated-vs-ground-truth browser viewer
    ├── assets/                # Browser viewer page for visualize.py
    ├── datasets/             # Dataset implementations (Nymeria Plus)
    ├── models/               # Model architectures (Transformer, Encoders)
    └── utils/                # Geometry, scene, and statistics helpers
```
