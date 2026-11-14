"""Classic multi-layer LSTM language model with tied input/output embeddings.

embedding (d) -> dropout -> LSTM layers (hidden h, dropout between layers)
-> linear output adapter (h -> d) -> dropout -> tied output head (E^T).

Why the adapter: the proposal ties input/output embeddings in both families AND uses
h != d, so the LSTM output (size h) must be mapped to the embedding width d before the
tied head (the TA asked to include "any projection needed to tie embeddings"; the
protocol fixes include_lstm_output_adapter: true). It is a plain linear map, the
minimal projection, applied AFTER the recurrence; nn.LSTM(proj_size=...) is not used
because it projects inside the recurrence (an LSTMP), which is no longer a classic LSTM.

Hidden widths are chosen so non-embedding parameters (LSTM + adapter) match the
Transformer's non-embedding parameters within 0.2% (protocol tolerance: 5%).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

# size -> (layers, embedding width d, hidden width h)
MODEL_SIZES = {
    "s": (2, 256, 472),
    "m": (2, 384, 873),
    "l": (2, 640, 1690),
    "m_deep": (4, 384, 588),  # Depth-confound control at size M.
}
# Each size's embedding width comes from this protocol size (matching.embedding_widths).
PROTOCOL_SIZE = {"s": "s", "m": "m", "l": "l", "m_deep": "m"}


@dataclass(frozen=True)
class LSTMConfig:
    vocab_size: int = 16384
    max_positions: int = 512
    layers: int = 2
    width: int = 256
    hidden: int = 472
    dropout: float = 0.1

    @classmethod
    def for_size(cls, size: str, *, dropout: float = 0.1) -> LSTMConfig:
        if size not in MODEL_SIZES:
            raise ValueError(f"Unknown LSTM size: {size}")
        layers, width, hidden = MODEL_SIZES[size]
        return cls(layers=layers, width=width, hidden=hidden, dropout=dropout)

    def validate(self) -> None:
        if (self.vocab_size <= 4 or self.max_positions <= 0 or self.layers <= 0
                or self.width <= 0 or self.hidden <= 0 or not 0 <= self.dropout < 1):
            raise ValueError("Invalid LSTM configuration")

    def to_dict(self) -> dict:
        return asdict(self)


class LSTMLM(nn.Module):
    """Logits [batch, time, vocab]; every call starts from zero hidden AND cell states."""

    def __init__(self, config: LSTMConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.embedding_dropout = nn.Dropout(config.dropout)
        # nn.LSTM's dropout applies between recurrent layers only (none after the last).
        self.lstm = nn.LSTM(config.width, config.hidden, num_layers=config.layers, batch_first=True,
                            dropout=config.dropout if config.layers > 1 else 0.0)
        self.output_adapter = nn.Linear(config.hidden, config.width, bias=True)
        self.adapter_dropout = nn.Dropout(config.dropout)
        self.output_head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight
        self._initialize()

    def _initialize(self) -> None:
        """Protocol: matrices N(0, 0.02), biases 0, forget gate input bias 1, PAD row 0."""
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if parameter.ndim >= 2:
                    nn.init.normal_(parameter, mean=0.0, std=0.02)
                else:
                    nn.init.zeros_(parameter)
            hidden = self.config.hidden
            for layer in range(self.config.layers):
                # PyTorch gate order is (input, forget, cell, output).
                getattr(self.lstm, f"bias_ih_l{layer}")[hidden:2 * hidden].fill_(1.0)
            self.token_embedding.weight[0].zero_()  # PAD stays trainable via the tied head.

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Same interface as TransformerLM. An LSTM has no positions; position_ids are checked only.

        Right padding needs no mask: a causal recurrence never lets later (padding) inputs
        affect earlier valid outputs, and padding targets are dropped by the shared loss mask.
        """
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be long [batch, time]")
        batch, time = input_ids.shape
        if batch == 0 or time == 0 or time > self.config.max_positions:
            raise ValueError("Invalid LSTM window shape")
        if attention_mask is not None and (attention_mask.shape != input_ids.shape
                                           or not bool(attention_mask[:, 0].all())
                                           or bool((attention_mask[:, 1:] & ~attention_mask[:, :-1]).any())):
            raise ValueError("attention_mask must be a nonempty right-padded prefix per row")
        if position_ids is not None and not torch.equal(
                position_ids, torch.arange(time, device=input_ids.device).expand(batch, -1)):
            raise ValueError("position_ids must restart at zero in every window")
        # hx=None means fresh zero (h, c) for every layer and row: no state crosses windows.
        outputs, _ = self.lstm(self.embedding_dropout(self.token_embedding(input_ids)), None)
        return self.output_head(self.adapter_dropout(self.output_adapter(outputs)))

    def parameter_counts(self) -> dict[str, int]:
        """Tied head counted once; the adapter belongs to the non-embedding budget."""
        embedding = self.token_embedding.weight.numel()
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"token_embedding": embedding, "non_embedding": total - embedding, "total": total}
