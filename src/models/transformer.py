"""
Transformer denoiser for motion diffusion.

Conditioning tokens (scene, text, goal) are concatenated onto the motion
sequence and share one self-attention per block — there is no separate
cross-attention path. The diffusion timestep instead modulates every block
via AdaLN-Zero rather than entering the token sequence itself.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.scene_encoder import CNN3DSceneEncoder
from src.models.text_encoder import DEFAULT_MAX_TOKENS, build_text_encoder
from src.utils.scene import scene_token_coordinates


class SinusoidalPosition(nn.Module):
    """
    1D Sinusoidal Positional Encoding added to motion sequence tokens.
    """

    def __init__(self, model_dim: int, maximum: int = 512):
        super().__init__()
        position = torch.arange(maximum, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / model_dim)
        )
        encoding = torch.zeros(maximum, model_dim)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("encoding", encoding)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[: values.shape[1]]


class TokenTypeEmbedding(nn.Module):
    """
    Learned per-modality embedding so self-attention can tell
    history/future/text/scene/goal tokens apart beyond content/position.
    """

    def __init__(self, model_dim: int, num_types: int = 5):
        super().__init__()
        self.embedding = nn.Embedding(num_types, model_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, type_index: int) -> torch.Tensor:
        return tokens + self.embedding.weight[type_index]


class Position3D(nn.Module):
    """
    MLP positional encoding for real 3D coordinates: encodes each scene
    token's anchor-local cell center, so nearby cells get related embeddings
    tied to real geometry instead of a memorized per-slot index.
    """

    def __init__(self, model_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(3, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.projection(coordinates)


class TimestepEmbedding(nn.Module):
    """
    Diffusion timestep embedder — the conditioning vector each block's
    AdaLN-Zero modulation is computed from.
    """

    def __init__(self, model_dim: int, maximum: int = 10000):
        super().__init__()
        self.position = SinusoidalPosition(model_dim, maximum)
        self.projection = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.projection(self.position.encoding[timesteps])


class ProgressEmbedding(nn.Module):
    """
    Embeds how far a sampled window sits within its annotation's frame span
    (0 = earliest possible start, 1 = latest), bucketed and sinusoidally
    encoded like the diffusion timestep -- so a window near the start of a
    labeled action reads differently from one near its end, even though both
    carry the identical text.
    """

    def __init__(self, model_dim: int, buckets: int = 100):
        super().__init__()
        self.buckets = int(buckets)
        self.position = SinusoidalPosition(model_dim, self.buckets)
        self.projection = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, progress: torch.Tensor) -> torch.Tensor:
        bucket = (progress.clamp(0.0, 1.0) * (self.buckets - 1)).round().long()
        return self.projection(self.position.encoding[bucket])


class SceneCondition(nn.Module):
    """
    Encodes scene inputs into spatial condition tokens with 3D positional embeddings.
    """

    def __init__(self, model_dim: int, dropout: float, *, voxels: int, scene_bounds):
        super().__init__()
        self.dropout = float(dropout)
        self.encoder = CNN3DSceneEncoder(int(voxels))
        self.projection = nn.Linear(self.encoder.output_dim, model_dim)
        self.position_encoder = Position3D(model_dim)
        coordinates = scene_token_coordinates(
            scene_bounds, self.encoder.output_resolution
        )
        self.register_buffer("coordinates", torch.from_numpy(coordinates).float())
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, scene: torch.Tensor | None):
        if scene is None:
            return None
        features = self.encoder(scene.float())
        tokens = self.norm(
            self.projection(features) + self.position_encoder(self.coordinates)
        )
        return tokens, None


class TextCondition(nn.Module):
    """
    Encodes natural language text prompts (or pre-cached text features) into condition tokens
    with key padding masks for padding tokens.

    A per-sample window-progress value (where the sampled window sits within
    its annotation's span) is summed onto the valid token positions before
    the CFG dropout gate is applied in encode_conditions, so it rides the
    same gate as the text itself -- progress means nothing without knowing
    what action it's progress through, so it should never survive text being
    dropped.

    `progress_enabled=False` omits `progress_embedding` entirely (no
    trainable weights for it), so a checkpoint trained without window-progress
    conditioning still loads cleanly and `window_progress` is simply ignored
    if a caller passes one in anyway -- see `src.config.build_inference`,
    which detects this per checkpoint rather than trusting the config.
    """

    def __init__(
        self,
        model_dim: int,
        dropout: float,
        *,
        feature_dim: int,
        encoder=None,
        progress_enabled: bool = True,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.encoder = encoder
        self.projection = nn.Linear(int(feature_dim), model_dim)
        self.progress_embedding = ProgressEmbedding(model_dim) if progress_enabled else None
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        features: torch.Tensor | None,
        window_progress: torch.Tensor | None = None,
    ):
        if features is None:
            return None
        weight = self.projection.weight
        features = features.to(device=weight.device, dtype=weight.dtype)
        valid = (features != 0).any(dim=-1)
        tokens = self.projection(features)
        if window_progress is not None and self.progress_embedding is not None:
            progress_tokens = self.progress_embedding(
                window_progress.to(device=weight.device, dtype=weight.dtype)
            )
            tokens = tokens + valid.unsqueeze(-1).float() * progress_tokens.unsqueeze(1)
        tokens = self.norm(tokens)
        key_padding_mask = ~valid
        return tokens, key_padding_mask


class GoalTokenEmbedding(nn.Module):
    """
    Encodes the goal displacement into one condition token, handled like
    SceneCondition/TextCondition -- CFG dropout masks it out via the shared
    `encode_conditions` loop, no separate null-token needed. This token is
    the only way the model sees the goal; it's no longer hard-inpainted.
    """

    def __init__(self, model_dim: int, dropout: float):
        super().__init__()
        self.dropout = float(dropout)
        self.value_projection = nn.Sequential(
            nn.Linear(3, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, goal: torch.Tensor | None):
        if goal is None:
            return None
        weight = self.value_projection[0].weight
        value = self.value_projection(
            goal.to(device=weight.device, dtype=weight.dtype).reshape(-1, 3)
        )
        tokens = self.norm(value).unsqueeze(1)
        return tokens, None


class MotionBlock(nn.Module):
    """
    Transformer block: shared self-attention over motion + condition tokens,
    then a 2-layer GELU feedforward, each sublayer modulated by AdaLN-Zero.
    """

    def __init__(self, model_dim: int, heads: int, ff_size: int, dropout: float):
        super().__init__()
        self.self_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.self_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, model_dim),
        )
        self.feedforward_dropout = nn.Dropout(dropout)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_dim, 6 * model_dim)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self, tokens: torch.Tensor, padding_mask: torch.Tensor, timestep_embedding: torch.Tensor
    ) -> torch.Tensor:
        (
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp,
        ) = self.adaLN_modulation(timestep_embedding).chunk(6, dim=-1)

        # 1. Self-Attention over the full motion + condition token sequence
        normed = self.self_norm(tokens) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attended, _ = self.self_attention(
            normed, normed, normed, key_padding_mask=padding_mask, need_weights=False
        )
        tokens = tokens + gate_msa.unsqueeze(1) * self.self_dropout(attended)

        # 2. Feedforward network
        normed = self.feedforward_norm(tokens) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        tokens = tokens + gate_mlp.unsqueeze(1) * self.feedforward_dropout(self.feedforward(normed))
        return tokens


class MotionTransformer(nn.Module):
    """
    Motion diffusion denoiser transformer.
    """

    def __init__(
        self,
        input_dim: int = 72,
        output_dim: int = 72,
        model_dim: int = 512,
        ff_size: int = 1024,
        heads: int = 8,
        layers: int = 8,
        dropout: float = 0.1,
        *,
        text_enabled: bool = True,
        text_dropout: float = 0.1,
        text_encoder_type: str = "clip",
        text_max_tokens: int = DEFAULT_MAX_TOKENS,
        cached_text_features: bool = False,
        text_feature_dim: int | None = None,
        text_progress_enabled: bool = True,
        scene_enabled: bool = True,
        scene_dropout: float = 0.1,
        scene_voxels: int | None = 64,
        scene_bounds=(-1.5, 1.5, -1.5, 1.5, -0.3, 2.7),
        goal_enabled: bool = True,
        goal_dropout: float = 0.1,
        joint_dropout: float = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.joint_dropout = float(joint_dropout)
        self.text_encoder_type = text_encoder_type
        self.text_max_tokens = text_max_tokens

        self.encoders = nn.ModuleDict()
        if scene_enabled:
            self.encoders["scene"] = SceneCondition(
                model_dim, scene_dropout, voxels=scene_voxels, scene_bounds=scene_bounds
            )
        if text_enabled:
            encoder = (
                None if cached_text_features
                else build_text_encoder(text_encoder_type, text_max_tokens)
            )
            self.encoders["text"] = TextCondition(
                model_dim,
                text_dropout,
                feature_dim=(text_feature_dim if encoder is None else encoder.output_dim),
                encoder=encoder,
                progress_enabled=text_progress_enabled,
            )
        if goal_enabled:
            self.encoders["goal"] = GoalTokenEmbedding(model_dim, goal_dropout)

        self.frame_projection = nn.Linear(input_dim, model_dim)
        self.timestep_embedding = TimestepEmbedding(model_dim)
        self.position = SinusoidalPosition(model_dim)
        self.token_type = TokenTypeEmbedding(model_dim, num_types=5)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            MotionBlock(model_dim, heads, ff_size, dropout)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, output_dim)

    @property
    def scene_enabled(self) -> bool:
        return "scene" in self.encoders

    @property
    def text_enabled(self) -> bool:
        return "text" in self.encoders

    @property
    def text_progress_enabled(self) -> bool:
        return self.text_enabled and self.encoders["text"].progress_embedding is not None

    @property
    def goal_enabled(self) -> bool:
        return "goal" in self.encoders

    def encode_text_features(self, text, batch_size: int) -> torch.Tensor:
        """
        Encodes raw text string prompts to text tensor features.
        """
        if not self.text_enabled:
            return None
        encoder = self.encoders["text"].encoder
        if encoder is None:
            device = next(self.parameters()).device
            encoder = build_text_encoder(self.text_encoder_type, self.text_max_tokens).to(device).eval()
            self.encoders["text"].encoder = encoder

        prompts = [text] * batch_size if isinstance(text, str) else [str(x) for x in (text or [""] * batch_size)]
        return encoder(prompts)

    def encode_conditions(
        self,
        batch: int,
        device,
        *,
        text=None,
        text_features: torch.Tensor | None = None,
        scene: torch.Tensor | None = None,
        goal: torch.Tensor | None = None,
        window_progress: torch.Tensor | None = None,
        drop: str | None = None,
        joint_mask: torch.Tensor | None = None,
    ) -> dict:
        """
        Pre-computes and caches condition tokens (scene, text, goal) and CFG dropout gates [B, 1, 1].
        """
        if text_features is None and text is not None and self.text_enabled:
            text_features = self.encode_text_features(text, batch)
        raw = {"scene": scene, "text": text_features, "goal": goal}

        conditions = {}
        for name in ("scene", "text", "goal"):
            if name not in self.encoders or name == drop:
                continue
            input_val = raw[name]
            if input_val is None:
                continue
            encoder = self.encoders[name]
            built = (
                encoder(input_val, window_progress) if name == "text"
                else encoder(input_val)
            )
            if built is None:
                continue
            tokens, padding_mask = built

            # Compute per-sample dropout gate for training & CFG
            gate = torch.ones(batch, 1, 1, device=device)
            if self.training and encoder.dropout > 0:
                kept = (torch.rand(batch, device=device) >= encoder.dropout).float()
                gate = gate * kept[:, None, None]
            if joint_mask is not None:
                gate = gate * (~joint_mask)[:, None, None].float()

            conditions[name] = (tokens, padding_mask, gate)

        return conditions

    def forward(
        self,
        noisy_current: torch.Tensor,
        timesteps: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        *,
        text=None,
        text_features: torch.Tensor | None = None,
        scene: torch.Tensor | None = None,
        goal: torch.Tensor | None = None,
        window_progress: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        drop: str | None = None,
        conditions: dict | None = None,
    ) -> torch.Tensor:
        """
        Predicts clean motion tokens from history prefix, noisy future frames, timestep t, and conditions.
        """
        batch = noisy_current.shape[0]
        device = noisy_current.device
        if conditions is None:
            joint_mask = None
            if self.training and self.joint_dropout:
                joint_mask = torch.rand(batch, device=device) < self.joint_dropout
            conditions = self.encode_conditions(
                batch,
                device,
                text=text,
                text_features=text_features,
                scene=scene,
                goal=goal,
                window_progress=window_progress,
                drop=drop,
                joint_mask=joint_mask,
            )
        elif drop is not None:
            conditions = {
                name: value for name, value in conditions.items() if name != drop
            }

        # 1. Project & concatenate motion sequence (history + noisy current),
        # tagging each span with its learned token-type embedding
        history_tokens = self.token_type(self.frame_projection(history), 0)
        current_tokens = self.token_type(self.frame_projection(noisy_current), 1)
        motion = torch.cat((history_tokens, current_tokens), dim=1)

        # 2. Add 1D temporal positional encoding (timestep modulates each
        # block via AdaLN-Zero instead of entering the sequence here)
        motion = self.position(motion)
        motion = self.input_dropout(motion)

        # 3. Build the motion padding mask, then concatenate condition tokens after motion. 
        current_mask = (
            target_mask.to(torch.bool) if target_mask is not None
            else torch.zeros(batch, noisy_current.shape[1], dtype=torch.bool, device=device)
        )
        sequence = [motion]
        masks = [history_mask.to(torch.bool), current_mask]
        for name, (tokens, padding_mask, gate) in conditions.items():
            tokens = self.token_type(tokens, {"text": 2, "scene": 3, "goal": 4}[name])
            count = tokens.shape[1]
            dropped = (gate.reshape(batch, 1) == 0).expand(-1, count)
            segment_mask = dropped if padding_mask is None else (padding_mask | dropped)
            sequence.append(tokens)
            masks.append(segment_mask)

        sequence = torch.cat(sequence, dim=1)
        padding_mask = torch.cat(masks, dim=1)

        # 4. Pass through MotionBlocks, all modulated by the same timestep embedding
        timestep_embedding = self.timestep_embedding(timesteps)
        for block in self.blocks:
            sequence = block(sequence, padding_mask, timestep_embedding)

        # 5. Final layer norm and linear output projection
        return self.output_projection(self.norm(sequence))
