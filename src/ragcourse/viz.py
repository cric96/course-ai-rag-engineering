"""Charts with one consistent style across the course, sized for a projector."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

BLUE = "#2a78d6"  # the data
NAVY = (
    "#173f73"  # a third run, after GRAY and BLUE (checked with the dataviz validator)
)
GRAY = "#c3c2b7"  # context, de-emphasized; also axis lines
INK = "#1f2937"  # text, same as the slides
FONTS = ["Source Sans 3", "DejaVu Sans"]  # the first one installed wins

STYLE = {
    # Real names, not "sans-serif": generic names resolve at draw time, outside STYLE.
    "font.family": FONTS,
    "font.size": 18,
    "text.color": INK,
    "ytick.color": INK,
    "axes.edgecolor": GRAY,
    "axes.titlesize": 20,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 16,
}


def bar_chart(
    counts: pd.Series, title: str, gray: Sequence[str] = (), fmt: str = "{:g}"
) -> Figure:
    """Horizontal bars in series order, each labeled with its value (e.g. "{:.0%}")."""
    with plt.rc_context(STYLE):
        # A bare Figure (not pyplot): Jupyter shows it once, as the cell result.
        fig = Figure(figsize=(10, 1.2 + 0.6 * len(counts)), layout="constrained")
        ax = fig.subplots()
        labels = [str(label) for label in counts.index]
        colors = [GRAY if label in gray else BLUE for label in labels]
        bars = ax.barh(labels, counts.to_numpy(), color=colors, height=0.6)

        ax.invert_yaxis()  # first item on top
        ax.bar_label(bars, fmt=fmt, padding=6)  # values at the tips: no x axis
        ax.set_title(title)
        ax.xaxis.set_visible(False)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    return fig


def compare_chart(table: pd.DataFrame, title: str, fmt: str = "{:g}") -> Figure:
    """Runs side by side, a group of bars per row: reference gray, then blue, navy."""
    if len(table.columns) not in (2, 3):
        raise ValueError("compare_chart compares two or three runs")
    with plt.rc_context(STYLE):
        n = len(table.columns)
        fig = Figure(figsize=(10, 1.6 + 0.5 * n * len(table)), layout="constrained")
        ax = fig.subplots()
        rows, height = np.arange(len(table)), 0.8 / n
        colors = [GRAY, BLUE, NAVY][:n]
        for i, (run, color) in enumerate(zip(table.columns, colors, strict=True)):
            shift = (i - (n - 1) / 2) * height  # bars centered on their row
            bars = ax.barh(
                rows + shift, table[run], height * 0.95, color=color, label=run
            )
            ax.bar_label(bars, fmt=fmt, padding=6)

        ax.set_yticks(rows, [str(label) for label in table.index])
        ax.invert_yaxis()  # first row on top, reference above the new run
        # Legend just above the bars, title above the legend.
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=n, frameon=False)
        ax.set_title(title, pad=52)
        ax.xaxis.set_visible(False)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    return fig
