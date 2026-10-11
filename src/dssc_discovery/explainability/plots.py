"""SHAP visualization helpers (Chunk 5.3): global bar and beeswarm,
local waterfall, dependence. Plain matplotlib returning Figures --
UI-agnostic, so scripts, reports (11.x) and Streamlit all call the same
functions. Every figure prints the interpretation caveat on itself, so
it can't be cropped away from the numbers it qualifies.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from dssc_discovery.explainability.base import SHAPExplanationSet  # noqa: E402

_CAVEAT_SHORT = "Associational, not causal -- see interpretation_caveat."


def _stamp_caveat(fig: Figure, explanation: SHAPExplanationSet) -> None:
    text = _CAVEAT_SHORT
    if explanation.interaction_caveat:
        text += " Interaction values: low confidence."
    fig.text(0.01, 0.005, text, fontsize=7, color="dimgray", ha="left", va="bottom")


def _top_features(explanation: SHAPExplanationSet, max_features: int) -> list[int]:
    importance = np.abs(explanation.values).mean(axis=0)
    return list(np.argsort(importance)[::-1][:max_features])


def plot_global_bar(explanation: SHAPExplanationSet, max_features: int = 15) -> Figure:
    order = _top_features(explanation, max_features)[::-1]
    importance = np.abs(explanation.values).mean(axis=0)
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(order) + 1.5))
    ax.barh([explanation.feature_names[i] for i in order], importance[order])
    ax.set_xlabel("mean |SHAP value| (PCE %)")
    ax.set_title(f"{explanation.model_name}: global feature importance")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _stamp_caveat(fig, explanation)
    return fig


def plot_beeswarm(explanation: SHAPExplanationSet, max_features: int = 15, seed: int = 0) -> Figure:
    order = _top_features(explanation, max_features)[::-1]
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(order) + 1.5))
    scatter = None
    for position, j in enumerate(order):
        values = explanation.values[:, j]
        colour = explanation.data[:, j]
        jitter = rng.uniform(-0.25, 0.25, len(values))
        scatter = ax.scatter(values, position + jitter, c=colour, cmap="coolwarm", s=12)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([explanation.feature_names[j] for j in order])
    ax.axvline(0.0, color="grey", linewidth=1)
    ax.set_xlabel("SHAP value (impact on predicted PCE %)")
    ax.set_title(f"{explanation.model_name}: SHAP beeswarm")
    if scatter is not None:
        fig.colorbar(scatter, ax=ax, label="feature value (preprocessed)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _stamp_caveat(fig, explanation)
    return fig


def plot_waterfall(explanation: SHAPExplanationSet, row: int, max_features: int = 10) -> Figure:
    """Local explanation for one row -- e.g. a BO-recommended condition:
    why the surrogate favours it, distinct from the acquisition rationale."""
    values = explanation.values[row]
    order = list(np.argsort(np.abs(values))[::-1])
    shown, rest = order[:max_features], order[max_features:]
    labels = [explanation.feature_names[j] for j in shown]
    contributions = [float(values[j]) for j in shown]
    if rest:
        labels.append(f"{len(rest)} other features")
        contributions.append(float(values[rest].sum()))

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(labels) + 1.8))
    running = explanation.base_value
    for position, (label, value) in enumerate(zip(labels, contributions, strict=True)):
        ax.barh(position, value, left=running, color="tab:red" if value > 0 else "tab:blue")
        running += value
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(explanation.base_value, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel(f"predicted PCE (%): base {explanation.base_value:.2f} -> {running:.2f}")
    ax.set_title(f"{explanation.model_name}: local explanation, row {row}")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _stamp_caveat(fig, explanation)
    return fig


def plot_dependence(
    explanation: SHAPExplanationSet, feature: str, color_by: str | None = None
) -> Figure:
    j = explanation.feature_names.index(feature)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    colour = explanation.data[:, explanation.feature_names.index(color_by)] if color_by else None
    points = ax.scatter(explanation.data[:, j], explanation.values[:, j], c=colour, s=14)
    if color_by:
        fig.colorbar(points, ax=ax, label=color_by)
    ax.axhline(0.0, color="grey", linewidth=1)
    ax.set_xlabel(f"{feature} (preprocessed)")
    ax.set_ylabel("SHAP value (PCE %)")
    ax.set_title(f"{explanation.model_name}: dependence on {feature}")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _stamp_caveat(fig, explanation)
    return fig
