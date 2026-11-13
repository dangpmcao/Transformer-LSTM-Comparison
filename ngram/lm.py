"""Map shared token IDs to KenLM words and score target windows in natural-log units.

KenLM sees each token ID as a decimal word ("523"). Its sentence markers stand in for
the tokenizer's controls: <s> is BOS (start of every window) and </s> is EOS (end of
an article or BLiMP sentence). Every window is scored from a fresh <s> state, exactly
as the neural models restart at every window, so no history crosses a window boundary.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

from dataset.common import PROJECT_ROOT
from tokenizer.core import SPECIAL_IDS

LN10 = math.log(10.0)  # protocol: multiply_base10_log_probability_by_ln10
ORDER = 5
EOS_WORD = "</s>"
DEFAULT_MODEL_ROOT = PROJECT_ROOT.parent / "models" / "ngram"
DEFAULT_KENLM_BIN = PROJECT_ROOT.parent / ".tools" / "kenlm" / "build" / "bin"
_NEVER_TARGETS = {SPECIAL_IDS["<pad>"], SPECIAL_IDS["<bos>"], SPECIAL_IDS["<unk>"]}

# Target IDs -> one natural-log probability per target, plus how many targets were OOV.
TargetLogProbs = Callable[[Sequence[int]], tuple[np.ndarray, int]]


def id_to_word(token_id: int) -> str:
    token_id = int(token_id)
    if token_id == SPECIAL_IDS["<eos>"]:
        return EOS_WORD
    if token_id in _NEVER_TARGETS or token_id < 0:
        raise ValueError(f"Token ID {token_id} can never be an n-gram word or target")
    return str(token_id)


def article_line(article_ids: Sequence[int]) -> str:
    """One training line: content IDs without the final EOS (KenLM appends </s> itself)."""
    ids = [int(token) for token in article_ids]
    if not ids or ids[-1] != SPECIAL_IDS["<eos>"] or SPECIAL_IDS["<eos>"] in ids[:-1]:
        raise ValueError("An article must end with exactly one EOS")
    return " ".join(id_to_word(token) for token in ids[:-1])


class KenLMScorer:
    """Thin wrapper around a loaded kenlm.Model using its incremental state API."""

    def __init__(self, model_path: Path | str):
        import kenlm  # Imported lazily: training/export and the tests do not need it.

        self._kenlm = kenlm
        self.model = kenlm.Model(str(model_path))
        if self.model.order != ORDER:
            raise ValueError(f"Expected a {ORDER}-gram model, got order {self.model.order}")

    def target_log_probs(self, targets: Sequence[int]) -> tuple[np.ndarray, int]:
        """Natural-log P(target | up to 4 previous targets), starting from <s> (BOS)."""
        state, next_state = self._kenlm.State(), self._kenlm.State()
        self.model.BeginSentenceWrite(state)
        values = np.empty(len(targets), dtype=np.float64)
        oov = 0
        for position, token in enumerate(targets):
            word = id_to_word(token)
            if word != EOS_WORD and word not in self.model:
                oov += 1  # Scored as KenLM's <unk>; counted and reported, never dropped.
            values[position] = self.model.BaseScore(state, word, next_state) * LN10
            state, next_state = next_state, state
        return values, oov


def window_scorer(target_log_probs: TargetLogProbs) -> Callable[[list[list[int]]], list[float]]:
    """Adapter for comparison.blimp_scorer.ScoreWindows: summed natural log per window."""

    def score(windows: list[list[int]]) -> list[float]:
        return [float(target_log_probs(window)[0].sum()) for window in windows]

    return score


def iter_article_lines(tokens: np.ndarray, offsets: np.ndarray) -> Iterable[str]:
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        yield article_line(np.asarray(tokens[int(start):int(end)]))
