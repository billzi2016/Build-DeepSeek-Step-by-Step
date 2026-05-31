"""Teaching utilities for attention, masks, and cache intuition.

The project notebooks repeatedly revisit the same ideas:

- how softmax turns scores into probabilities
- how a causal mask blocks future positions
- how a single attention head maps Q, K, V to an output
- how multi-head attention reshapes tensors
- how key/value caches grow during autoregressive decoding

This module collects those ideas in one place with deliberately explicit code.
Nothing here is optimized. The point is to make shapes and intermediate values
easy to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AttentionComputation:
    """Bundle together the intermediate tensors of one attention computation.

    Keeping these values grouped is useful in notebooks because a learner often
    wants to inspect not only the final output, but also:

    - raw attention scores
    - masked scores
    - normalized attention weights
    - the final value-weighted output
    """

    scores: np.ndarray
    masked_scores: np.ndarray
    weights: np.ndarray
    output: np.ndarray


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a numerically stable softmax.

    Why subtract the maximum value first:
        ``exp(large_number)`` can overflow quickly. Subtracting the row-wise
        maximum leaves the final probabilities unchanged but avoids instability.
    """

    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def build_causal_mask(seq_len: int) -> np.ndarray:
    """Create a lower-triangular causal mask.

    Returns:
        A matrix of shape ``[seq_len, seq_len]`` where:

        - ``1`` means "this key position is visible to this query position"
        - ``0`` means "this would look into the future and must be blocked"
    """

    return np.tril(np.ones((seq_len, seq_len), dtype=np.int32))


def apply_additive_mask(scores: np.ndarray, mask: np.ndarray, blocked_value: float = -1e9) -> np.ndarray:
    """Apply an additive mask to attention scores.

    The standard teaching trick is to replace blocked locations with a very
    negative number. After softmax, those positions receive probability mass
    extremely close to zero.
    """

    masked_scores = scores.copy()
    masked_scores = np.where(mask == 1, masked_scores, blocked_value)
    return masked_scores


def single_head_self_attention(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    use_causal_mask: bool = True,
) -> AttentionComputation:
    """Run one self-attention head with full intermediate outputs.

    Args:
        queries:
            Array of shape ``[seq_len, d_k]``.
        keys:
            Array of shape ``[seq_len, d_k]``.
        values:
            Array of shape ``[seq_len, d_v]``.
        use_causal_mask:
            Whether to block future positions.

    Returns:
        An ``AttentionComputation`` object containing the full path from scores
        to output.
    """

    d_k = queries.shape[-1]

    # The dot product measures how well each query aligns with each key.
    scores = queries @ keys.T

    # Dividing by sqrt(d_k) keeps score magnitudes from growing too large as
    # dimensionality increases. This stabilizes the downstream softmax.
    scores = scores / np.sqrt(d_k)

    if use_causal_mask:
        mask = build_causal_mask(seq_len=queries.shape[0])
        masked_scores = apply_additive_mask(scores, mask)
    else:
        masked_scores = scores

    weights = stable_softmax(masked_scores, axis=-1)

    # Attention output is a weighted sum over the value vectors.
    output = weights @ values

    return AttentionComputation(
        scores=scores,
        masked_scores=masked_scores,
        weights=weights,
        output=output,
    )


def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    """Split a ``[seq_len, d_model]`` tensor into ``[num_heads, seq_len, d_head]``.

    This is the cleanest small-scale layout for teaching. Framework code often
    uses batch dimensions too, but removing batch first makes the head split
    easier to reason about.
    """

    seq_len, d_model = x.shape
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    d_head = d_model // num_heads
    reshaped = x.reshape(seq_len, num_heads, d_head)
    return np.transpose(reshaped, (1, 0, 2))


def combine_heads(x: np.ndarray) -> np.ndarray:
    """Reverse ``split_heads``.

    Args:
        x:
            Array of shape ``[num_heads, seq_len, d_head]``.

    Returns:
        Array of shape ``[seq_len, d_model]`` where ``d_model = num_heads * d_head``.
    """

    num_heads, seq_len, d_head = x.shape
    transposed = np.transpose(x, (1, 0, 2))
    return transposed.reshape(seq_len, num_heads * d_head)


def estimate_kv_cache_size(
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """Estimate the size of a KV cache in bytes for one layer.

    Why multiply by 2:
        We store both keys and values.
    """

    return seq_len * num_kv_heads * head_dim * dtype_bytes * 2


def append_to_kv_cache(
    cached_keys: np.ndarray | None,
    cached_values: np.ndarray | None,
    new_keys: np.ndarray,
    new_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Append newly generated token states to an existing KV cache.

    Shapes:
        cached_keys / cached_values:
            ``[past_seq_len, num_kv_heads, head_dim]`` or ``None`` on the first step.
        new_keys / new_values:
            ``[new_seq_len, num_kv_heads, head_dim]``.
    """

    if cached_keys is None or cached_values is None:
        return new_keys.copy(), new_values.copy()

    updated_keys = np.concatenate([cached_keys, new_keys], axis=0)
    updated_values = np.concatenate([cached_values, new_values], axis=0)
    return updated_keys, updated_values
