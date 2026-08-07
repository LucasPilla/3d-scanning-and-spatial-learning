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

### Goal

The goal is the anchor-local pelvis position `(dx, dy, dz)`. It is by construction the same physical quantity as channels `[0:3]` of the motion feature, so it reuses the motion statistics' own pelvis channels for normalization.

There are two natural ways to make a diffusion model reach a goal:

*   **Goal as a condition token (soft, CFG-guided)**: the goal enters the shared attention sequence just like text and scene, and the model is only *encouraged* toward it by the training loss. This supports classifier-free guidance (a `goal_guidance_scale`) to tune how strongly it's followed at sampling time, but reaching it is never guaranteed — the model can trade it off against the scene or the text. This is what the earlier, pre-redesign model (`v1`) does.
*   **Goal as hard inpainting (architectural guarantee)**: every denoising step overwrites the target frame's pelvis position with the goal value directly, the same way history frames are fixed. Reaching the goal stops being a preference the loss encourages and becomes a constraint the sampling process enforces, at the cost of the freedom a soft token has to negotiate a path with the rest of the scene.

The current model uses **hard inpainting**. `GoalTokenEmbedding` still emits one token (a learned null vector when the goal is absent or was dropped) so the rest of the network knows whether a goal was set, but the actual position is enforced by inpainting rather than left to CFG — so there is no `goal_guidance_scale` for it.

## 5. Losses

The forward process corrupts the ground-truth future window $x_0$ at a random timestep $t$ in the usual DDPM way, $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$, and the transformer denoiser predicts the clean window $\hat{x}_0$ directly from $x_t$ (not the noise $\epsilon$). `MotionDiffusion.training_loss` returns a single `total` key: a masked MSE between $\hat{x}_0$ and $x_0$ over the un-padded future frames, scaled by `position_weight` ($\lambda$):

$$\mathcal{L} = \lambda \cdot \frac{1}{|\text{valid}| \cdot D}\sum_{i \,\in\, \text{valid}} \left(\hat{x}_{0,i} - x_{0,i}\right)^2$$

where $D = 72$ is the per-frame feature dimension (`JOINT_COUNT * 3`) and "valid" excludes any padded target frames — in practice the future side is never padded, so this is a plain MSE on the normalized anchored joint-position features. There is only this one loss term; no forward kinematics, velocity, or foot-contact term is involved.
