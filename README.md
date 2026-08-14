# Human Motion Generation from Text, Scene and Trajectory

Autoregressive rollouts on held-out test-split segments, conditioned on text, scene, and goal:

| | | |
|---|---|---|
| ![Cooking](./docs/assets/gifs/cooking.gif) | ![Fridge](./docs/assets/gifs/fridge.gif) | ![Laying Down](./docs/assets/gifs/laying_down.gif) |
| Cooking | Fridge | Laying Down |
| ![Sitting](./docs/assets/gifs/sitting.gif) | ![Stairs](./docs/assets/gifs/stairs.gif) | ![Walking](./docs/assets/gifs/walking.gif) |
| Sitting | Stairs | Walking |

Ablations on a synthetic scene, varying the classifier-free guidance scale of each condition from 0.0 to 3.0 in isolation:

| | |
|---|---|
| ![Text Ablation](./docs/assets/gifs/synthetic_text_ablation.gif) | ![Goal Ablation](./docs/assets/gifs/synthetic_goal_ablation.gif) |
| Text Ablation | Goal Ablation |
| ![Scene Ablation](./docs/assets/gifs/synthetic_scene_ablation.gif) | ![No-Obstacle Scene Ablation](./docs/assets/gifs/synthetic_noobstacle_scene_ablation.gif) |
| Scene Ablation | No-Obstacle Scene Ablation |

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

Roll out every held-out test-split segment autoregressively -- window 1
conditioned on that annotation's own history, scene, text, and goal, every
window after it conditioned on the model's own prior generation -- and save
it alongside ground truth:

```bash
python -m src.evaluation.test \
  --config runs/main_run/config.yaml \
  --checkpoint runs/main_run/checkpoints/main_run_epoch300.pth \
  --output runs/main_run/test_results.npz
```

Then browse the rolled-out segments against ground truth in a local viewer,
playing each segment continuously (not window by window), overlaid on real
per-object scene geometry near its anchor by pointing `--raw-scenes` at the
raw Nymeria download:

```bash
python -m src.evaluation.visualize_categories \
  --results runs/main_run/test_results.npz \
  --raw-scenes /path/to/nymeria_dataset/download
```

Prints the local URL to open (`--port`, default `8002`).

Score every test-split segment with a full autoregressive rollout (each
window after the first conditioned on the model's own prior generation, not
ground truth), reporting collision rate, floor-contact distance, and
goal-reaching error against a ground-truth baseline:

```bash
python -m src.evaluation.eval \
  --config runs/main_run/config.yaml \
  --checkpoint runs/main_run/checkpoints/main_run_epoch300.pth \
  --output runs/main_run/eval_results.npz
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
    ├── evaluation/            # Test-split generation & browser viewers
    │   ├── test.py            # Full-split autoregressive rollout script
    │   ├── test_categories.py # Hand-picked category-segment rollout script
    │   ├── eval.py             # Collision/floor-contact/goal-reaching rollout metrics
    │   ├── visualize_categories.py # Ragged, per-segment rollout browser viewer
    │   └── assets/            # Browser viewer pages for the scripts above
    ├── datasets/             # Dataset implementations (Nymeria Plus)
    ├── models/               # Model architectures (Transformer, Encoders)
    └── utils/                # Geometry, scene, and statistics helpers
```
