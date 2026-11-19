"""Float32 inference with exact BOS/reset/right-padding semantics."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from transformer.model import TransformerLM


class Scorer:
    def __init__(self, model: TransformerLM, device: str, batch_size: int = 16,
                 progress: Callable[[], None] | None = None):
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.progress = progress

    def _batches(self, lengths: Sequence[int]):
        if not lengths or min(lengths) < 1 or max(lengths) > self.model.config.max_positions:
            raise ValueError('Inputs must have 1..max_positions tokens')
        order = np.argsort(lengths, kind='stable')
        for offset in range(0, len(order), self.batch_size):
            chosen = order[offset:offset + self.batch_size]
            if self.progress:
                self.progress()
            yield chosen

    @torch.inference_mode()
    def windows(self, targets: Sequence[Sequence[int]]) -> list[np.ndarray]:
        """All per-target NLLs; BOS is an input and is never a scored target."""
        lengths = [len(x) for x in targets]
        output = [None] * len(targets)
        for chosen in self._batches(lengths):
            width = max(lengths[i] for i in chosen)
            ids = torch.zeros((len(chosen), width), dtype=torch.long, device=self.device)
            truth = torch.zeros_like(ids)
            for row, i in enumerate(chosen):
                sequence = torch.as_tensor(targets[i], dtype=torch.long, device=self.device)
                truth[row, :lengths[i]] = sequence
                ids[row, 0] = 1
                ids[row, 1:lengths[i]] = sequence[:-1]
            logits = self.model(ids)  # [B,T,V]; padded queries are never included.
            losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), truth.reshape(-1),
                                     reduction='none').reshape(ids.shape).double().cpu().numpy()
            for row, i in enumerate(chosen):
                value = losses[row, :lengths[i]].copy()
                if not np.isfinite(value).all():
                    raise FloatingPointError('Nonfinite scored target')
                output[i] = value
        return output

    @torch.inference_mode()
    def last(self, inputs: Sequence[Sequence[int]], targets: Sequence[int]) -> np.ndarray:
        """Score only final valid logits, avoiding the unused [B,T,V] output head.

        The Transformer trunk is identical to model.forward. LayerNorm/output head
        operate independently per position, so selecting hidden states before the
        head preserves the causal target score up to float32 GEMM rounding.
        """
        if len(inputs) != len(targets):
            raise ValueError('One target is required per input history')
        lengths = [len(x) for x in inputs]
        output = np.empty(len(inputs), dtype=np.float64)
        for chosen in self._batches(lengths):
            width = max(lengths[i] for i in chosen)
            ids = torch.zeros((len(chosen), width), dtype=torch.long, device=self.device)
            for row, i in enumerate(chosen):
                ids[row, :lengths[i]] = torch.as_tensor(inputs[i], dtype=torch.long, device=self.device)
            positions = torch.arange(width, device=self.device).expand_as(ids)
            x = self.model.embedding_dropout(self.model.token_embedding(ids) +
                                              self.model.position_embedding(positions))
            for block in self.model.blocks:
                x = block(x)
            last = torch.as_tensor([lengths[i] - 1 for i in chosen], device=self.device)
            x = x[torch.arange(len(chosen), device=self.device), last]
            logits = self.model.output_head(self.model.final_norm(x))  # [B,V]
            truth = torch.as_tensor([targets[i] for i in chosen], dtype=torch.long, device=self.device)
            value = F.cross_entropy(logits, truth, reduction='none').double().cpu().numpy()
            if not np.isfinite(value).all():
                raise FloatingPointError('Nonfinite final-target score')
            output[chosen] = value
        return output
