import pandas as pd
from matplotlib.colors import to_hex

from ragcourse.viz import BLUE, FONTS, GRAY, bar_chart


def test_bar_chart_draws_one_bar_per_item_with_its_value():
    fig = bar_chart(pd.Series({"testo": 200, "tabella": 7}), title="Domande")

    ax = fig.axes[0]
    assert len(ax.patches) == 2
    assert [label.get_text() for label in ax.texts] == ["200", "7"]


def test_bar_chart_keeps_the_series_order_top_to_bottom():
    fig = bar_chart(pd.Series({"primo": 1, "secondo": 5}), title="t")

    top, bottom = fig.axes[0].patches
    assert top.get_y() < bottom.get_y()  # y axis is inverted: smaller y is on top


def test_bar_chart_grays_out_context_items():
    fig = bar_chart(pd.Series({"a": 1, "b": 2}), title="t", gray=["b"])

    a, b = fig.axes[0].patches
    assert to_hex(a.get_facecolor()) == BLUE
    assert to_hex(b.get_facecolor()) == GRAY


def test_bar_chart_names_its_font_so_it_survives_outside_the_style_context():
    fig = bar_chart(pd.Series({"a": 1}), title="t")
    assert fig.axes[0].title.get_fontfamily()[0] == FONTS[0]


def test_bar_chart_formats_values_for_example_as_percentages():
    fig = bar_chart(pd.Series({"hit@5": 0.282}), title="t", fmt="{:.0%}")
    assert fig.axes[0].texts[0].get_text() == "28%"
