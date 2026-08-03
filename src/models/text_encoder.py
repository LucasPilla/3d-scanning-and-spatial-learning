"""
Encode text prompts as sequences of frozen conditioning tokens.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    DistilBertModel,
    T5EncoderModel,
)

# Annotations run about 47 tokens on average, so this keeps most of them whole
# while bounding the cross-attention memory and the on-disk feature cache.
DEFAULT_MAX_TOKENS = 64

# Selectable backbones, for configuration validation and `--help` text.
TEXT_ENCODER_NAMES = ("clip", "bert", "t5")


def build_text_encoder(method: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> nn.Module:
    """
    Construct the configured text encoder backbone.
    """

    cls = None
    method = str(method).lower()
    if method == "clip":
        cls = ClipTextEncoder
    if method == "bert":
        cls = BertTextEncoder
    if method == "t5":
        cls = T5TextEncoder

    if cls is not None:
        return cls(max_tokens)

    raise ValueError(
        f"Unsupported text encoder {method!r}; "
        f"choose one of {TEXT_ENCODER_NAMES}"
    )


class FrozenTextEncoder(nn.Module):

    #: Hugging Face model class and checkpoint.
    backbone: type
    model_name: str
    feature_dim: int
    token_limit: int | None

    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS):
        super().__init__()
        self.max_tokens = int(max_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self.backbone.from_pretrained(self.model_name)
        self.output_dim = int(self.model.config.hidden_size)
        self.model.requires_grad_(False)
        self.model.eval()

    def forward(self, texts) -> torch.Tensor:
        """
        Tokenize and encode prompts as [batch, max_tokens, feature_dim].
        """

        texts = [str(text) for text in texts]
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        ).to(device)

        self.model.eval()
        with torch.no_grad():
            hidden = self.model(**tokens).last_hidden_state

        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        present = torch.tensor(
            [bool(text.strip()) for text in texts], 
            device=device, 
            dtype=torch.bool
        )
        return hidden * mask * present[:, None, None].to(hidden.dtype)


class ClipTextEncoder(FrozenTextEncoder):
    """
    CLIP's frozen text tower.

    Its causal transformer has only 77 learned positions, so it truncates long
    annotations harder than the others — roughly a quarter of this corpus.
    """

    backbone = CLIPTextModel
    model_name = "openai/clip-vit-base-patch32"
    feature_dim = 512
    token_limit = 77


class BertTextEncoder(FrozenTextEncoder):
    """
    DistilBERT's frozen encoder, the text tower MDM's DiP uses.

    Bidirectional and cheap, with 512 positions — no annotation here comes
    close to that, so nothing is truncated in practice.
    """

    backbone = DistilBertModel
    model_name = "distilbert-base-uncased"
    feature_dim = 768
    token_limit = 512


class T5TextEncoder(FrozenTextEncoder):
    """
    T5's frozen encoder.

    Uses relative position bias rather than a learned position table, so it
    places no hard limit on `max_tokens`.
    """

    backbone = T5EncoderModel
    model_name = "t5-small"
    feature_dim = 512
    token_limit = None


