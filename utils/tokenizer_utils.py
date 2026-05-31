"""Teaching utilities for tokenization and tiny BPE experiments.

This module is intentionally verbose.

The goal is not to provide the fastest tokenizer implementation. The goal is
to provide a small set of functions that notebooks can import when they want to
show:

- how text is split into primitive symbols
- how BPE counts adjacent pairs
- how repeated merges build larger subword units
- how encoding and decoding work step by step

Most functions below favor readability over cleverness. Intermediate variable
names are deliberately explicit so a reader can follow the transformation path
without mentally decompiling compact Python.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class BPEMerge:
    """A single merge rule learned during BPE training.

    Attributes:
        left:
            The left symbol in the merged pair.
        right:
            The right symbol in the merged pair.
        merged:
            The new symbol created by combining ``left`` and ``right``.
        frequency:
            How often the pair appeared at the moment it was merged.

    Keeping the merge frequency is useful in notebooks because students often
    want to inspect not only *what* got merged, but also *why* it was chosen.
    """

    left: str
    right: str
    merged: str
    frequency: int


def split_to_char_tokens(text: str, end_of_word: str = "</w>") -> list[list[str]]:
    """Split text into a list of words, where each word is a list of symbols.

    Args:
        text:
            Raw input text. The implementation uses simple whitespace splitting
            because the point here is BPE intuition, not production tokenization.
        end_of_word:
            Marker appended to the end of each word. This mirrors the classic
            BPE teaching setup where the model can distinguish internal symbols
            from the end of a word.

    Returns:
        A nested list. Each outer item is a word. Each inner item is a symbol
        sequence such as ``["h", "e", "l", "l", "o", "</w>"]``.
    """

    tokenized_words: list[list[str]] = []
    for raw_word in text.strip().split():
        symbols = list(raw_word)
        symbols.append(end_of_word)
        tokenized_words.append(symbols)
    return tokenized_words


def count_pair_frequencies(tokenized_words: Iterable[Iterable[str]]) -> Counter[tuple[str, str]]:
    """Count how often each adjacent symbol pair appears.

    In BPE, the key question at each training step is:
    "Which adjacent pair appears most often right now?"

    This function answers that question for the current symbolization.
    """

    pair_counter: Counter[tuple[str, str]] = Counter()
    for word_symbols in tokenized_words:
        word_symbols = list(word_symbols)
        for index in range(len(word_symbols) - 1):
            pair = (word_symbols[index], word_symbols[index + 1])
            pair_counter[pair] += 1
    return pair_counter


def merge_symbol_pair(
    tokenized_words: Iterable[Iterable[str]],
    pair_to_merge: tuple[str, str],
) -> list[list[str]]:
    """Apply one BPE merge to every word in the corpus.

    Args:
        tokenized_words:
            Current corpus representation.
        pair_to_merge:
            The adjacent symbol pair to be merged, for example ``("l", "o")``.

    Returns:
        A fresh nested list with the target pair collapsed wherever it appears.

    Why we return a fresh structure instead of editing in place:
        In teaching notebooks, it is useful to compare "before merge" and
        "after merge" states. Returning a new list keeps the transition explicit.
    """

    left_symbol, right_symbol = pair_to_merge
    merged_symbol = left_symbol + right_symbol

    merged_corpus: list[list[str]] = []
    for word_symbols in tokenized_words:
        word_symbols = list(word_symbols)
        new_word: list[str] = []
        position = 0

        while position < len(word_symbols):
            has_next = position < len(word_symbols) - 1
            if has_next and word_symbols[position] == left_symbol and word_symbols[position + 1] == right_symbol:
                new_word.append(merged_symbol)
                position += 2
            else:
                new_word.append(word_symbols[position])
                position += 1

        merged_corpus.append(new_word)

    return merged_corpus


def train_tiny_bpe(
    text: str,
    num_merges: int,
    end_of_word: str = "</w>",
) -> tuple[list[list[str]], list[BPEMerge]]:
    """Train a very small BPE model for educational use.

    Args:
        text:
            Training corpus.
        num_merges:
            Maximum number of merge operations to perform.
        end_of_word:
            End-of-word marker.

    Returns:
        A tuple ``(final_corpus_representation, merge_history)``.

    Notes:
        - If the corpus runs out of repeated pairs early, training stops.
        - This implementation is intentionally small and transparent rather than
          optimized for scale.
    """

    tokenized_words = split_to_char_tokens(text, end_of_word=end_of_word)
    merge_history: list[BPEMerge] = []

    for _ in range(num_merges):
        pair_counts = count_pair_frequencies(tokenized_words)
        if not pair_counts:
            break

        most_frequent_pair, frequency = pair_counts.most_common(1)[0]
        if frequency < 2:
            # If the best pair appears only once, continuing to merge teaches
            # very little. We stop early so the merge history stays meaningful.
            break

        merged_symbol = most_frequent_pair[0] + most_frequent_pair[1]
        merge_history.append(
            BPEMerge(
                left=most_frequent_pair[0],
                right=most_frequent_pair[1],
                merged=merged_symbol,
                frequency=frequency,
            )
        )
        tokenized_words = merge_symbol_pair(tokenized_words, most_frequent_pair)

    return tokenized_words, merge_history


def encode_with_merges(
    word: str,
    merge_history: Iterable[BPEMerge],
    end_of_word: str = "</w>",
) -> list[str]:
    """Encode one word using a previously learned merge history.

    This function deliberately replays merges one by one. That makes it ideal
    for notebooks because the reader can inspect how a character sequence
    gradually turns into subword units.
    """

    symbols = list(word) + [end_of_word]

    for merge in merge_history:
        symbols = merge_symbol_pair([symbols], (merge.left, merge.right))[0]

    return symbols


def decode_subword_tokens(tokens: Iterable[str], end_of_word: str = "</w>") -> str:
    """Decode a sequence of BPE-style subword tokens back into plain text.

    The implementation removes the end-of-word marker and joins recovered words
    with spaces. This mirrors the simple teaching setup used throughout the
    notebooks.
    """

    recovered_words: list[str] = []
    current_word_parts: list[str] = []

    for token in tokens:
        if token.endswith(end_of_word):
            clean_piece = token[: -len(end_of_word)]
            current_word_parts.append(clean_piece)
            recovered_words.append("".join(current_word_parts))
            current_word_parts = []
        else:
            current_word_parts.append(token)

    if current_word_parts:
        recovered_words.append("".join(current_word_parts))

    return " ".join(recovered_words)


def format_merge_history(merge_history: Iterable[BPEMerge]) -> list[str]:
    """Turn merge rules into readable strings for notebook printing."""

    formatted: list[str] = []
    for step_index, merge in enumerate(merge_history, start=1):
        formatted.append(
            f"step {step_index:02d}: ({merge.left}, {merge.right}) -> {merge.merged} "
            f"| frequency={merge.frequency}"
        )
    return formatted
