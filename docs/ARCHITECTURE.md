# Architecture

## 1. Diagram

![Architecture Diagram](/home/lucas/.gemini/antigravity-ide/brain/eec86bfe-a105-446c-ab42-ac998c2dd094/architecture_diagram_1785886849622.png)

## 2. Transformer Architecture

Our model relies on token concatenation with shared self-attention.

*   **Sequence**: The input to the transformer is a single sequence of tokens constructed by concatenating all condition tokens onto the motion tokens.

    *   **Motion Tokens**: `64 tokens` (16 history + 48 future). The 72-D joint positions per frame are linearly projected into the model's hidden dimension, and sinusoidal positional embeddings plus a learned token-type embedding are added.

    *   **Text Tokens**: `64 tokens`, direct projection of the frozen encoder's per-token hidden states.

    *   **Scene Tokens**: `64 tokens`, each carrying a real 3D positional encoding.

    *   **Goal Token**: `1 token`

*   **Self-Attention**: Every transformer block applies a standard Multi-Head Self-Attention over this combined sequence. The motion tokens can attend directly to the text, the scene layout, and the goal vector simultaneously, without separate cross-attention layers.

*   **Classifier-Free Guidance (CFG)**: Missing or dropped conditions (e.g., dropping text to allow unconditional sampling) are simply excluded from the sequence using a key-padding mask, meaning the network dynamically adjusts to whichever conditions are present.

*   **Token Type Embeddings**: Concatenating five kinds of token into one sequence leaves self-attention with no signal for which is which beyond content. `TokenTypeEmbedding` adds one learned vector per modality — `0 = history`, `1 = future`, `2 = text`, `3 = scene`, `4 = goal` — to every token in that span, the same idea as BERT's segment embedding.

*   **Diffusion Timestep Injection**: It modulates the activations of every block via an **AdaLN-Zero** mechanism, as per ablation from DiT paper.

## 4. Condition Encoders

### Text Encoder

Text tokens are generated using a frozen language model (e.g., CLIP, DistilBERT, or T5). The raw text string is tokenized, passed through the pre-trained model, and the resulting dense hidden states are projected into the transformer's dimension. 

### Scene Encoder

The environment geometry is represented as a 3D boolean voxel occupancy grid of shape $64 \times 64 \times 64$ (at 5cm resolution, bounding a $3.2\text{m}^3$ volume around the person).

*   `CNN3DSceneEncoder` downsamples the grid with stride-2 convolutions ($64 \rightarrow 32 \rightarrow 16 \rightarrow 8 \rightarrow 4$), doubling channels as resolution shrinks.

*   **The output resolution is fixed at $4 \times 4 \times 4$, not derived from the input.** The number of stride-2 layers is computed from `scene_voxels` at construction, so the scene contributes **64 tokens**.

*   Because of the convolutional receptive field, each token observes a $1.55\text{m}$ physical area, providing rich overlapping context of stairs, tables, or walls.

*   **3D positional encoding**: each token's anchor-local $(x, y, z)$ cell center runs through `Position3D`, a small shared MLP, and is added to the projected feature.

### Goal (Legacy)

The goal is the anchor-local pelvis position `(dx, dy, dz)` of **the window's own last real frame**, chosen deterministically rather than sampled from a random future horizon. Since windows are always exactly `window_size` real frames (see below), that frame is unambiguous, and the goal is by construction the same physical quantity as channels `[0:3]` of the motion feature — so it reuses the motion statistics' own pelvis channels for normalization instead of carrying separate `goal_mean`/`goal_std`.

*   **Fixed-length, randomly-placed windows**: a window is `window_size` frames whose start is drawn uniformly from within the annotation's span, falling back to the annotation's start when the span is too short to offer a choice. At the default `window_size: 48` this collapses to roughly one window per annotation; a smaller value turns repeated accesses of the same annotation into free sub-window augmentation, and lets history contain the earlier part of the very same described action. Either way the future side is never padded — `target_mask` is always all-`False`.

*   **Goal token**: `GoalTokenEmbedding` emits exactly one token at a fixed sequence position, always. It carries the encoded goal when one is present, and a learned null vector when the goal is absent or was dropped — the same null-conditioning idea classifier-free guidance uses. A fixed position means the rest of the network never has to locate the goal frame's index to attend to its value. Goal takes no part in AdaLN: the block conditioning vector is still exactly the timestep embedding.

*   **Hard position-only inpainting**: every denoising step overwrites channels `[0:3]` of frame `history_frames + window_size - 1` with the goal value, exactly as history frames are inpainted. This is what makes goal-following an architectural guarantee rather than something the loss merely encourages. Only the pelvis position is fixed; the rest of that frame's pose is still generated.

*   **The presence mask is computed once.** `MotionDiffusion.training_loss` owns the goal-dropout draw and passes the same `goal_present` tensor to both the model and the inpainting step. Deriving them separately would let a sample whose token was dropped still read its goal off the inpainted channels — a training-time leak that would teach the model to ignore the token.

*   **No CFG for goal**: a goal is either followed outright (token + inpaint) or absent (null token, no inpaint). There is nothing to extrapolate between, so there is no `goal_guidance_scale`.

## 5. Losses

`MotionDiffusion.training_loss` returns a single `total` key: an MSE directly on the normalized anchored joint positions.
