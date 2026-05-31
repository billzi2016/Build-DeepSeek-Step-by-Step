"""Teaching helpers for plots used across the notebooks.

The notebooks frequently need the same kinds of figures:

- matrix heatmaps
- attention weight maps
- token-by-position similarity maps
- selected dimension traces across positions

These helpers keep plotting code small while still making the plotting logic
readable. The comments below are intentionally explanatory because this project
is meant to teach through the code itself.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create a figure output directory if it does not already exist."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_heatmap(
    matrix: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    cmap: str = "viridis",
    figsize: tuple[int, int] = (10, 4),
    colorbar_label: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a 2D matrix as a heatmap.

    Why heatmaps are useful in this project:
        Many LLM structures are easiest to understand as grids:

        - positional encoding tables
        - attention weights
        - similarity matrices
        - relative bias matrices

    A heatmap turns hidden numeric structure into a directly visible pattern.
    """

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    colorbar = fig.colorbar(image, ax=ax)
    if colorbar_label is not None:
        colorbar.set_label(colorbar_label)
    fig.tight_layout()
    return fig, ax


def plot_line_traces(
    x_values: np.ndarray,
    traces: dict[str, np.ndarray],
    title: str,
    x_label: str,
    y_label: str,
    figsize: tuple[int, int] = (10, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot several named 1D traces on the same axes.

    This is especially useful for showing:

    - how selected positional-encoding dimensions vary with position
    - how reward changes with distance
    - how angle growth behaves under RoPE scaling
    """

    fig, ax = plt.subplots(figsize=figsize)
    for label, trace in traces.items():
        ax.plot(x_values, trace, label=label)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Compute a cosine-similarity matrix for row vectors.

    If ``matrix`` has shape ``[n_items, dim]``, the result has shape
    ``[n_items, n_items]`` and shows how similar every pair of row vectors is.
    """

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-12)
    normalized = matrix / safe_norms
    return normalized @ normalized.T


def save_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    """Save a figure and return the resolved path."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    return output_path
