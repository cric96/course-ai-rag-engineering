"""Charts with one consistent style across the course, sized for a projector."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

BLUE = "#2a78d6"  # the data
GRAY = "#c3c2b7"  # context, de-emphasized
INK = "#1f2937"  # text, same as the slides

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Fira Sans", "DejaVu Sans"],
    "font.size": 16,
    "text.color": INK,
    "ytick.color": INK,
    "axes.titlesize": 20,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 16,
}


def bar_chart(counts: pd.Series, title: str, gray: Sequence[str] = ()) -> Figure:
    """Horizontal bars in series order, each labeled with its value."""
    with plt.rc_context(STYLE):
        # A bare Figure (not pyplot): Jupyter shows it once, as the cell result.
        fig = Figure(figsize=(10, 1.2 + 0.6 * len(counts)), layout="constrained")
        ax = fig.subplots()
        labels = [str(label) for label in counts.index]
        colors = [GRAY if label in gray else BLUE for label in labels]
        bars = ax.barh(labels, counts.to_numpy(), color=colors, height=0.6)

        ax.invert_yaxis()  # first item on top
        ax.bar_label(bars, padding=6)  # values at the tips: no x axis needed
        ax.set_title(title)
        ax.xaxis.set_visible(False)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    return fig
