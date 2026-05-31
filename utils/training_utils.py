"""Teaching utilities for training objectives and alignment toy examples.

These helpers are intentionally small and explicit. They are not meant to be
drop-in replacements for framework training loops. Their role is to support
notebooks that explain:

- next-token cross entropy
- masked SFT losses
- pairwise reward-model objectives
- PPO clipping intuition
- group-relative normalization used by GRPO-style examples
"""

from __future__ import annotations

import numpy as np


def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a numerically stable softmax distribution from logits."""

    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def cross_entropy_from_logits(logits: np.ndarray, target_index: int) -> float:
    """Compute cross entropy for one categorical prediction.

    Args:
        logits:
            A 1D array of raw scores of shape ``[vocab_size]``.
        target_index:
            The correct class index.
    """

    probabilities = stable_softmax(logits)
    return float(-np.log(probabilities[target_index]))


def sequence_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute per-token and mean cross entropy for a sequence.

    Args:
        logits:
            Shape ``[seq_len, vocab_size]``.
        targets:
            Shape ``[seq_len]`` with the correct token id at each position.

    Returns:
        ``(per_token_loss, mean_loss)``
    """

    probabilities = stable_softmax(logits, axis=-1)
    token_indices = np.arange(len(targets))
    per_token_loss = -np.log(probabilities[token_indices, targets])
    return per_token_loss, float(per_token_loss.mean())


def masked_sequence_cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
    loss_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute a masked token-level loss, as in SFT.

    The loss mask usually marks assistant tokens with ``1`` and prompt tokens
    with ``0``. This lets notebooks show clearly why assistant-only supervision
    is so common in instruction tuning.
    """

    per_token_loss, _ = sequence_cross_entropy(logits, targets)
    masked_loss = (per_token_loss * loss_mask).sum() / max(float(loss_mask.sum()), 1.0)
    return per_token_loss, float(masked_loss)


def pairwise_reward_loss(chosen_score: float, rejected_score: float) -> float:
    """Compute a logistic pairwise preference loss for reward modeling."""

    margin = chosen_score - rejected_score
    return float(-np.log(1.0 / (1.0 + np.exp(-margin))))


def ppo_clipped_objective(
    old_logprob: np.ndarray,
    new_logprob: np.ndarray,
    advantage: np.ndarray,
    clip_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the core PPO clipped objective terms.

    Returns:
        A tuple ``(ratio, clipped_ratio, objective)`` so notebooks can print
        and inspect every intermediate quantity.
    """

    ratio = np.exp(new_logprob - old_logprob)
    clipped_ratio = np.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    objective = np.minimum(ratio * advantage, clipped_ratio * advantage)
    return ratio, clipped_ratio, objective


def group_relative_advantages(rewards: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Normalize a set of rewards within a sample group.

    This is a simple teaching version of the intuition often used in
    GRPO-style explanations: what matters is not only the raw reward, but how a
    candidate answer ranks relative to its peers for the same prompt.
    """

    group_mean = rewards.mean()
    group_std = rewards.std() + epsilon
    return (rewards - group_mean) / group_std


def summarize_training_stages(stage_descriptions: list[dict[str, str]]) -> list[str]:
    """Format a stage list into readable one-line summaries."""

    summaries: list[str] = []
    for stage in stage_descriptions:
        summaries.append(
            f"{stage['stage']}: input={stage['input_data']} | "
            f"objective={stage['objective']} | output={stage['output']}"
        )
    return summaries
