"""E5 routing signals, all computed from the SAME Tier 0 logits.

Orientation convention, applied to every signal without exception:

    HIGHER = MORE CONFIDENT = LESS need to escalate.

Entropy is naturally the other way round, so the entropy signal is returned NEGATED.
Mixing orientations is the kind of silent sign error that produces a plausible AUROC of
1 - x instead of x, so the convention is enforced by `SIGNALS` rather than remembered.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def max_softmax(logits: np.ndarray) -> np.ndarray:
    """Highest posterior probability. The standard confidence baseline."""
    return softmax(logits).max(axis=-1)


def margin(logits: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 posterior. Measures decisiveness, not absolute confidence."""
    p = np.sort(softmax(logits), axis=-1)
    return p[..., -1] - p[..., -2]


def neg_entropy(logits: np.ndarray) -> np.ndarray:
    """Negated Shannon entropy of the posterior, in nats.

    Negated so that higher means more confident, matching every other signal. Uses the
    full distribution rather than only its top two entries, so it can separate cases
    that `margin` cannot.
    """
    p = softmax(logits)
    return (p * np.log(np.clip(p, 1e-12, None))).sum(axis=-1)


CLOSED_FORM: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "max_softmax": max_softmax,
    "margin": margin,
    "neg_entropy": neg_entropy,
}


def feature_matrix(logits: np.ndarray) -> np.ndarray:
    """Features for the trained difficulty predictor.

    Deliberately built from the same logits as the closed-form signals and nothing else:
    the comparison E5 makes is "does a LEARNED combination of these beat the best single
    one", which is only meaningful if the learner sees no extra information.
    """
    p = softmax(logits)
    srt = np.sort(p, axis=-1)
    return np.column_stack([
        srt[:, -1],                                   # max softmax
        srt[:, -1] - srt[:, -2],                      # margin
        (p * np.log(np.clip(p, 1e-12, None))).sum(-1),  # neg entropy
        srt[:, -3] if p.shape[1] >= 3 else np.zeros(len(p)),  # 3rd mass
        logits.max(axis=-1),                          # raw top logit (scale, not shape)
        logits.std(axis=-1),                          # logit spread
    ])
