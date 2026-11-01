"""Public names for next-token datasets, batches, losses, and fresh LSTM states.

The implementations remain in windows.py because existing experiment snapshots
pin its exact source hash. These exports do not change tensors or data order.
"""

from .windows import (
    ArticleWindows as NextTokenWindowDataset,
    WindowOrder as SeededWindowSampler,
    budget_mask as limit_loss_mask_to_budget,
    initial_lstm_state as create_lstm_zero_state,
    masked_cross_entropy as compute_masked_cross_entropy,
    window_loader as create_batch_loader,
)

__all__ = [
    "NextTokenWindowDataset", "SeededWindowSampler", "create_batch_loader",
    "limit_loss_mask_to_budget", "compute_masked_cross_entropy", "create_lstm_zero_state",
]
