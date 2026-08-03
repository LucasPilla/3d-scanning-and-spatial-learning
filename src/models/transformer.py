"""
Transformer denoiser for motion diffusion.

Combines self-attention across temporal motion frames with sequential cross-attention 
into independently gated conditioning modalities (scene, text, and goal).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.scene_encoder import build_scene_encoder
from src.models.text_encoder import DEFAULT_MAX_TOKENS, build_text_encoder


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


class TimestepEmbedding(nn.Module):
    """
    Diffusion Timestep Embedder.

    Maps discrete timestep t via 1D sinusoidal encoding followed by a 2-layer MLP projection.
    Returns tensor of shape [B, 1, model_dim] for broadcasting across motion tokens.
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
        return self.projection(self.position.encoding[timesteps]).unsqueeze(1)


class SceneCondition(nn.Module):
    """
    Encodes scene inputs into spatial condition tokens with learned positional embeddings.
    """

    def __init__(self, model_dim: int, dropout: float, *, encoder_type: str, voxels: int | None):
        super().__init__()
        self.dropout = float(dropout)
        self.encoder = build_scene_encoder(encoder_type, int(voxels or 32))
        self.projection = nn.Linear(self.encoder.output_dim, model_dim)
        self.position = nn.Parameter(torch.empty(self.encoder.token_count, model_dim))
        nn.init.normal_(self.position, std=0.02)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, scene: torch.Tensor | None):
        if scene is None:
            return None
        features = self.encoder(scene.float())
        tokens = self.norm(self.projection(features) + self.position)
        return tokens, None


class TextCondition(nn.Module):
    """
    Encodes natural language text prompts (or pre-cached text features) into condition tokens
    with key padding masks for padding tokens.
    """

    def __init__(self, model_dim: int, dropout: float, *, feature_dim: int, encoder=None):
        super().__init__()
        self.dropout = float(dropout)
        self.encoder = encoder
        self.projection = nn.Linear(int(feature_dim), model_dim)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, features: torch.Tensor | None):
        if features is None:
            return None
        weight = self.projection.weight
        features = features.to(device=weight.device, dtype=weight.dtype)
        valid = (features != 0).any(dim=-1)
        tokens = self.norm(self.projection(features))
        key_padding_mask = ~valid
        return tokens, key_padding_mask


class GoalCondition(nn.Module):
    """
    Encodes goal displacement target into a single goal token [B, 1, model_dim] via MLP.
    """

    def __init__(self, model_dim: int, dropout: float):
        super().__init__()
        self.dropout = float(dropout)
        self.projection = nn.Sequential(
            nn.Linear(3, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, goal: torch.Tensor | None):
        if goal is None:
            return None
        weight = self.projection[0].weight
        goal = goal.to(device=weight.device, dtype=weight.dtype)
        tokens = self.norm(self.projection(goal.reshape(-1, 3))).unsqueeze(1)
        return tokens, None


class ConditionAttention(nn.Module):
    """
    Cross-attention branch for one conditioning modality (scene, text, or goal).

    Prepends a learned `null_token` to key/value tokens so softmax attention mass can be downweighted
    when a condition is uninformative or single-token (goal). Multiplies residual output by a per-sample
    dropout `gate` [B, 1, 1] for Classifier-Free Guidance (CFG).
    """

    def __init__(self, model_dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.null_token = nn.Parameter(torch.empty(1, 1, model_dim))
        nn.init.normal_(self.null_token, std=0.02)
        self.attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, motion: torch.Tensor, condition_tuple: tuple) -> torch.Tensor:
        tokens, padding_mask, gate = condition_tuple
        batch = tokens.shape[0]

        # Prepend null_token at index 0 of key/value tokens
        tokens = torch.cat(
            (self.null_token.expand(batch, -1, -1).to(tokens.dtype), tokens), dim=1
        )
        if padding_mask is not None:
            visible = torch.zeros(batch, 1, dtype=torch.bool, device=padding_mask.device)
            padding_mask = torch.cat((visible, padding_mask), dim=1)

        attended, _ = self.attention(
            self.norm(motion),
            tokens,
            tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return motion + self.dropout(attended) * gate


class MotionBlock(nn.Module):
    """
    Transformer Block consisting of:
    1. Temporal self-attention over motion frames (history + noisy current motion).
    2. Sequential cross-attention through enabled modalities (scene -> text -> goal).
    3. 2-layer GELU Feed-Forward Network with pre-norm residuals.
    """

    def __init__(
        self,
        model_dim: int,
        heads: int,
        ff_size: int,
        dropout: float,
        condition_names: tuple[str, ...],
    ):
        super().__init__()
        self.self_norm = nn.LayerNorm(model_dim)
        self.self_attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.self_dropout = nn.Dropout(dropout)
        self.conditions = nn.ModuleDict({
            name: ConditionAttention(model_dim, heads, dropout)
            for name in condition_names
        })
        self.feedforward_norm = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, model_dim),
        )
        self.feedforward_dropout = nn.Dropout(dropout)

    def forward(self, motion: torch.Tensor, padding_mask: torch.Tensor, conditions: dict) -> torch.Tensor:
        # 1. Motion Self-Attention
        normed = self.self_norm(motion)
        attended, _ = self.self_attention(
            normed, normed, normed, key_padding_mask=padding_mask, need_weights=False
        )
        motion = motion + self.self_dropout(attended)

        # 2. Sequential Condition Cross-Attentions
        for name, branch in self.conditions.items():
            if name in conditions:
                motion = branch(motion, conditions[name])

        # 3. Feedforward Network
        return motion + self.feedforward_dropout(
            self.feedforward(self.feedforward_norm(motion))
        )


class MotionTransformer(nn.Module):
    """
    Motion Diffusion Denoiser Transformer.

    Predicts clean fixed-history-plus-future motion sequences from noisy inputs,
    diffusion timesteps, and optional conditioning signals (scene, text, goal).
    """

    def __init__(
        self,
        input_dim: int = 148,
        output_dim: int = 148,
        model_dim: int = 512,
        ff_size: int = 1024,
        heads: int = 4,
        layers: int = 8,
        dropout: float = 0.1,
        *,
        text_enabled: bool = True,
        text_dropout: float = 0.1,
        text_encoder_type: str = "clip",
        text_max_tokens: int = DEFAULT_MAX_TOKENS,
        cached_text_features: bool = False,
        text_feature_dim: int | None = None,
        scene_enabled: bool = True,
        scene_dropout: float = 0.1,
        scene_voxels: int | None = 32,
        scene_encoder_type: str = "vit",
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
                model_dim, scene_dropout, encoder_type=scene_encoder_type, voxels=scene_voxels
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
            )
        if goal_enabled:
            self.encoders["goal"] = GoalCondition(model_dim, goal_dropout)

        condition_names = tuple(name for name in ("scene", "text", "goal") if name in self.encoders)

        self.frame_projection = nn.Linear(input_dim, model_dim)
        self.timestep_embedding = TimestepEmbedding(model_dim)
        self.position = SinusoidalPosition(model_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            MotionBlock(model_dim, heads, ff_size, dropout, condition_names)
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
            built = encoder(input_val)
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
                drop=drop,
                joint_mask=joint_mask,
            )
        elif drop is not None:
            conditions = {
                name: value for name, value in conditions.items() if name != drop
            }

        # 1. Project & concatenate motion sequence (history + noisy current)
        history_tokens = self.frame_projection(history)
        current_tokens = self.frame_projection(noisy_current)
        motion = torch.cat((history_tokens, current_tokens), dim=1)

        # 2. Add 1D temporal positional encoding and broadcasted diffusion timestep embedding
        motion = self.position(motion) + self.timestep_embedding(timesteps)
        motion = self.input_dropout(motion)

        # 3. Build padding mask for motion sequence
        current_mask = torch.zeros(
            batch, noisy_current.shape[1], dtype=torch.bool, device=device
        )
        padding_mask = torch.cat((history_mask.to(torch.bool), current_mask), dim=1)

        # 4. Pass through MotionBlocks
        for block in self.blocks:
            motion = block(motion, padding_mask, conditions)

        # 5. Final layer norm and linear output projection
        return self.output_projection(self.norm(motion))
