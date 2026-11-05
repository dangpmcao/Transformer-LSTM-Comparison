"""Pre-normalization, learned-position, causal Transformer language model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


MODEL_SIZES = {
    "s": (4, 256, 4),
    "m": (6, 384, 6),
    "l": (8, 640, 10),
}


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 16384
    max_positions: int = 512
    layers: int = 4
    width: int = 256
    heads: int = 4
    feedforward_multiplier: int = 4
    dropout: float = 0.1
    bias: bool = True
    activation: str = "gelu"
    position_encoding: str = "learned_absolute"

    @classmethod
    def for_size(cls, size: str, *, dropout: float = 0.1) -> TransformerConfig:
        if size not in MODEL_SIZES:
            raise ValueError(f"Unknown Transformer size: {size}")
        layers, width, heads = MODEL_SIZES[size]
        return cls(layers=layers, width=width, heads=heads, dropout=dropout)

    def validate(self) -> None:
        if (self.vocab_size <= 4 or self.max_positions <= 0 or self.layers <= 0
                or self.width <= 0 or self.heads <= 0 or self.width % self.heads
                or self.feedforward_multiplier <= 0 or not 0 <= self.dropout < 1
                or self.activation != "gelu" or self.position_encoding != "learned_absolute"):
            raise ValueError("Invalid Transformer configuration")

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=config.bias)
        self.output = nn.Linear(config.width, config.width, bias=config.bias)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=-1)
        q, k, v = (part.view(batch, time, self.heads, self.head_width).transpose(1, 2)
                   for part in (q, k, v))
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, time, width)
        return self.residual_dropout(self.output(attended))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width, elementwise_affine=True, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.feedforward_norm = nn.LayerNorm(config.width, elementwise_affine=True, bias=config.bias)
        self.feedforward = nn.Sequential(
            nn.Linear(config.width, config.feedforward_multiplier * config.width, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.feedforward_multiplier * config.width, config.width, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feedforward(self.feedforward_norm(x))


class TransformerLM(nn.Module):
    """Logits [batch, time, vocab]; each call starts with an independent BOS window."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.max_positions, config.width)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.width, elementwise_affine=True, bias=config.bias)
        self.output_head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight
        self.apply(self._initialize)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()  # PAD is trainable via the tied output head.

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be long [batch, time]")
        batch, time = input_ids.shape
        if batch == 0 or time == 0 or time > self.config.max_positions:
            raise ValueError("Invalid Transformer window shape")
        if attention_mask is not None:
            if (attention_mask.shape != input_ids.shape or attention_mask.dtype != torch.bool
                    or attention_mask.device != input_ids.device or not bool(attention_mask[:, 0].all())
                    or bool((attention_mask[:, 1:] & ~attention_mask[:, :-1]).any())):
                raise ValueError("attention_mask must be a nonempty right-padded prefix per row")
            # With right padding, a valid query's causal keys are all valid. Padding
            # queries are discarded by the shared loss mask.
        expected_positions = torch.arange(time, device=input_ids.device).expand(batch, -1)
        if position_ids is None:
            position_ids = expected_positions
        if (position_ids.shape != input_ids.shape or position_ids.dtype != torch.long
                or position_ids.device != input_ids.device
                or not torch.equal(position_ids, expected_positions)):
            raise ValueError("position_ids must restart at zero in every window")
        x = self.embedding_dropout(self.token_embedding(input_ids) + self.position_embedding(position_ids))
        for block in self.blocks:
            x = block(x)
        return self.output_head(self.final_norm(x))

    def parameter_counts(self) -> dict[str, int]:
        # The tied output head is the same Parameter, so count it only once.
        # Learned positions belong to the non-token-embedding matching budget.
        embedding = self.token_embedding.weight.numel()
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"token_embedding": embedding, "non_embedding": total - embedding, "total": total}
