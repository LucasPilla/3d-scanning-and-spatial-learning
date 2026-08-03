"""
Gaussian diffusion loss and joint classifier-free window sampling.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional
from tqdm import tqdm

from src.utils.kinematics import features_to_joints


def cosine_beta_schedule(timesteps: int) -> torch.Tensor:
    """
    Return the cosine DDPM schedule used by MDM.
    """

    steps = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
    cumulative = torch.cos((steps + 0.008) / 1.008 * math.pi / 2).square()
    cumulative = cumulative / cumulative[0]
    return (1.0 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999).float()


def extract(values: torch.Tensor, timesteps: torch.Tensor, shape) -> torch.Tensor:
    selected = values.gather(0, timesteps)
    return selected.reshape(len(timesteps), *((1,) * (len(shape) - 1)))


class MotionDiffusion(nn.Module):
    """
    Train and sample windows with hard-fixed history frames.
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 50,
        window_size: int = 16,
        *,
        statistics=None,
        feature_weight: float = 1.0,
        position_weight: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.timesteps = int(timesteps)
        self.window_size = int(window_size)
        self.statistics = statistics
        self.feature_weight = float(feature_weight)
        self.position_weight = float(position_weight)
        if self.position_weight and statistics is None:
            raise ValueError(
                "The position loss needs motion statistics to denormalize features"
            )
        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        previous = functional.pad(cumulative[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_cumulative_alphas", torch.sqrt(cumulative))
        self.register_buffer("sqrt_one_minus_cumulative_alphas", torch.sqrt(1.0 - cumulative))
        self.register_buffer(
            "posterior_variance", betas * (1.0 - previous) / (1.0 - cumulative)
        )
        self.register_buffer(
            "posterior_clean_coefficient",
            betas * torch.sqrt(previous) / (1.0 - cumulative),
        )
        self.register_buffer(
            "posterior_noisy_coefficient",
            (1.0 - previous) * torch.sqrt(alphas) / (1.0 - cumulative),
        )

    def q_sample(self, clean_motion, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(clean_motion)
        return (
            extract(self.sqrt_cumulative_alphas, timesteps, clean_motion.shape)
            * clean_motion
            + extract(
                self.sqrt_one_minus_cumulative_alphas, timesteps, clean_motion.shape
            )
            * noise
        )

    @staticmethod
    def _inpaint(window, history, history_frames: int):
        """
        Fix clean history frames.
        """

        window[:, :history_frames] = history
        return window

    def _predict_window(
        self,
        window,
        steps,
        history_mask,
        history_frames: int,
        **arguments,
    ):
        return self.model(
            window[:, history_frames:],
            steps,
            window[:, :history_frames],
            history_mask,
            **arguments,
        )

    def training_loss(
        self,
        clean_motion,
        timesteps,
        history,
        history_mask,
        *,
        bone_offsets=None,
        text=None,
        text_features=None,
        scene=None,
        goal=None,
        noise=None,
    ):
        """
        Compute the weighted denoising loss for one batch.

        Returns a dict of unweighted scalars plus the weighted `total` under
        which backward is called, so every term can be logged separately.
        """

        history_frames = history.shape[1]
        clean_window = torch.cat((history, clean_motion), dim=1)
        if noise is None:
            noise = torch.randn_like(clean_window)
        noisy_window = self.q_sample(clean_window, timesteps, noise)
        self._inpaint(noisy_window, history, history_frames)
        predicted_window = self._predict_window(
            noisy_window,
            timesteps,
            history_mask,
            history_frames,
            text=text,
            text_features=text_features,
            scene=scene,
            goal=goal,
        )
        predicted_motion = predicted_window[:, history_frames:]

        losses = {
            "feature": functional.mse_loss(predicted_motion, clean_motion),
        }
        total = self.feature_weight * losses["feature"]
        if not self.position_weight:
            losses["total"] = total
            return losses

        if bone_offsets is None:
            raise ValueError("The position loss needs per-sample bone offsets")
        # The window's own root deltas integrate from the identity pose at the
        # anchor, so both skeletons land in the same canonical frame with no
        # world transform anywhere.
        predicted_joints = features_to_joints(
            self.statistics.denormalize(predicted_motion), bone_offsets
        )
        target_joints = features_to_joints(
            self.statistics.denormalize(clean_motion), bone_offsets
        )
        losses["position"] = functional.mse_loss(predicted_joints, target_joints)
        losses["total"] = total + self.position_weight * losses["position"]
        return losses

    @torch.no_grad()
    def sample_window(
        self,
        history,
        history_mask,
        *,
        text=None,
        text_features=None,
        scene=None,
        goal=None,
        text_guidance_scale: float = 1.0,
        goal_guidance_scale: float = 1.0,
        scene_guidance_scale: float = 1.0,
        generator=None,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Sample one window with fixed history, conditioned on a pelvis goal.
        """

        device = next(self.model.parameters()).device
        batch, history_frames = history.shape[:2]
        window_frames = history_frames + self.window_size
        motion = torch.randn(
            batch,
            window_frames,
            self.model.input_dim,
            device=device,
            generator=generator,
        )
        if goal is not None:
            goal = torch.as_tensor(
                goal, device=device, dtype=motion.dtype
            ).reshape(batch, 3)
        was_training = self.model.training
        self.model.eval()
        try:
            # The memories depend on the conditions alone, so they are built
            # once and reused by every step and every guidance pass.
            conditions = self.model.encode_conditions(
                batch,
                device,
                text=text,
                text_features=text_features,
                scene=scene,
                goal=goal,
            )
            # A modality absent from `conditions` contributes nothing whether
            # or not it is dropped, so guiding on it could only cancel out.
            scales = {
                "text": float(text_guidance_scale),
                "goal": float(goal_guidance_scale),
                "scene": float(scene_guidance_scale),
            }
            guided = tuple(
                name for name, scale in scales.items()
                if name in conditions and scale != 1.0
            )
            for index in tqdm(
                reversed(range(self.timesteps)),
                total=self.timesteps,
                desc="sampling diffusion steps",
                disable=not show_progress,
            ):
                self._inpaint(motion, history, history_frames)
                steps = torch.full((batch,), index, device=device, dtype=torch.long)
                conditional = self._predict_window(
                    motion, steps, history_mask, history_frames, conditions=conditions
                )
                predicted_clean = conditional
                for name in guided:
                    without = self._predict_window(
                        motion,
                        steps,
                        history_mask,
                        history_frames,
                        conditions=conditions,
                        drop=name,
                    )
                    predicted_clean = predicted_clean + (scales[name] - 1.0) * (
                        conditional - without
                    )
                self._inpaint(predicted_clean, history, history_frames)
                mean = (
                    extract(
                        self.posterior_clean_coefficient, steps, motion.shape
                    )
                    * predicted_clean
                    + extract(
                        self.posterior_noisy_coefficient, steps, motion.shape
                    )
                    * motion
                )
                if index:
                    variance = extract(self.posterior_variance, steps, motion.shape)
                    motion = mean + torch.sqrt(variance) * torch.randn(
                        motion.shape, device=device, dtype=motion.dtype, generator=generator
                    )
                else:
                    motion = mean
        finally:
            self.model.train(was_training)
        # The history region is re-fixed at the top of every step and discarded
        # here, so it never needs inpainting on the way out.
        return motion[:, history_frames:]
