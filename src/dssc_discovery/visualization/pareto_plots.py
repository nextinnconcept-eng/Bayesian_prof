"""PCE-vs-stability Pareto front reporting (Chunk 9.3). Shows the
trade-off itself, never a collapsed weighted score; dominated points are
drawn greyed out, not removed, so the full picture stays visible.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


def dominance_classification(data: pd.DataFrame, objectives: tuple[str, str]) -> pd.Series:
    """True for non-dominated rows (maximize both objectives). A row is
    dominated if another row is >= on both and > on at least one. Rows
    missing either objective are excluded (NaN in the result)."""
    values = data[list(objectives)].to_numpy(dtype=float)
    complete = ~np.isnan(values).any(axis=1)
    result = pd.Series(np.nan, index=data.index, dtype=object)
    observed = values[complete]
    flags = []
    for row in observed:
        dominated = np.any(np.all(observed >= row, axis=1) & np.any(observed > row, axis=1))
        flags.append(not dominated)
    result.loc[data.index[complete]] = flags
    return result


def plot_pareto_front(
    data: pd.DataFrame,
    objectives: tuple[str, str] = ("power_conversion_efficiency_percent", "retention_percent"),
    labels: tuple[str, str] = ("PCE (%)", "Retention (%)"),
    title: str = "PCE vs stability trade-off",
) -> Figure:
    pareto = dominance_classification(data, objectives)
    complete = pareto.notna()
    front = data[complete & (pareto == True)].sort_values(objectives[0])  # noqa: E712
    dominated = data[complete & (pareto == False)]  # noqa: E712

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(
        dominated[objectives[0]],
        dominated[objectives[1]],
        color="lightgrey",
        s=20,
        label="dominated",
    )
    ax.scatter(
        front[objectives[0]], front[objectives[1]], color="tab:red", s=30, label="Pareto front"
    )
    ax.step(front[objectives[0]], front[objectives[1]], where="post", color="tab:red", alpha=0.6)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    n_missing = int((~complete).sum())
    if n_missing:
        ax.text(
            0.99,
            0.01,
            f"{n_missing} rows without both objectives not shown",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="dimgray",
        )
    ax.legend()
    fig.tight_layout()
    return fig
