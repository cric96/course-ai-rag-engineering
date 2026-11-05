import pandas as pd
import pytest
from matplotlib.colors import to_hex

from ragcourse.viz import FONTS, GRAY, ORANGE, TEAL, bar_chart, compare_chart


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
    assert to_hex(a.get_facecolor()) == ORANGE
    assert to_hex(b.get_facecolor()) == GRAY


def test_bar_chart_names_its_font_so_it_survives_outside_the_style_context():
    fig = bar_chart(pd.Series({"a": 1}), title="t")
    assert fig.axes[0].title.get_fontfamily()[0] == FONTS[0]


def test_bar_chart_formats_values_for_example_as_percentages():
    fig = bar_chart(pd.Series({"hit@5": 0.282}), title="t", fmt="{:.0%}")
    assert fig.axes[0].texts[0].get_text() == "28%"


TABLE = pd.DataFrame(
    {"baseline": {"hit@1": 0.1, "hit@5": 0.3}, "filtro": {"hit@1": 0.2, "hit@5": 0.6}}
)


def test_compare_chart_draws_the_reference_gray_and_the_new_run_orange():
    fig = compare_chart(TABLE, title="t", fmt="{:.0%}")

    ax = fig.axes[0]
    colors = [to_hex(bar.get_facecolor()) for bar in ax.patches]
    assert colors == [GRAY, GRAY, ORANGE, ORANGE]
    assert [label.get_text() for label in ax.texts] == ["10%", "30%", "20%", "60%"]


def test_compare_chart_names_both_runs_in_a_legend():
    fig = compare_chart(TABLE, title="t")
    assert [text.get_text() for text in fig.axes[0].get_legend().get_texts()] == [
        "baseline",
        "filtro",
    ]


def test_compare_chart_adds_a_third_run_in_dark_teal():
    fig = compare_chart(TABLE.assign(altro=0.5), title="t")

    colors = [to_hex(bar.get_facecolor()) for bar in fig.axes[0].patches]
    assert colors == [GRAY, GRAY, ORANGE, ORANGE, TEAL, TEAL]


def test_compare_chart_compares_two_or_three_runs():
    with pytest.raises(ValueError, match="two or three"):
        compare_chart(TABLE.assign(altro=0.5, ancora=0.1), title="t")
