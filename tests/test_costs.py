from datetime import date

import pytest

from ragcourse.costs import api_cost


def usage(inputs=1000, outputs=200, cached=0, reasoning=0):
    return {
        "model": {
            "input_tokens": inputs,
            "output_tokens": outputs,
            "input_token_details": {"cache_read": cached},
            "output_token_details": {"reasoning": reasoning},
        }
    }


NOVEMBER = date(2026, 11, 8)


def test_flash_counts_cached_input_at_its_discounted_rate():
    cost = api_cost("google_genai:gemini-3.8-flash", usage(cached=400), NOVEMBER)

    assert cost == pytest.approx((600 * 0.75 + 400 * 0.075 + 200 * 3.75) / 1e6)


def test_reasoning_is_already_in_output_tokens():
    cost = api_cost("google_genai:gemini-4-argon", usage(reasoning=150), NOVEMBER)

    assert cost == pytest.approx((1000 * 2 + 200 * 10) / 1e6)


def test_announced_price_changes_follow_the_request_date():
    argon = "google_genai:gemini-4-argon"
    flash = "google_genai:gemini-3.8-flash"

    assert api_cost(argon, usage(), date(2026, 12, 1)) == pytest.approx(
        2 * api_cost(argon, usage(), NOVEMBER)
    )
    assert api_cost(flash, usage(), date(2027, 1, 1)) == pytest.approx(
        2 * api_cost(flash, usage(), NOVEMBER)
    )


def test_missing_usage_or_unknown_prices_are_not_reported_as_free():
    assert api_cost("google_genai:gemini-3.8-flash", {}, NOVEMBER) is None
    assert api_cost("google_genai:unknown", usage(), NOVEMBER) is None
    assert api_cost("google_genai:gemini-3.8-flash", {"model": {}}, NOVEMBER) is None


def test_gemma_and_local_models_have_no_api_token_charge():
    assert api_cost("google_genai:gemma-4-31b-it", {}, NOVEMBER) == 0
    assert api_cost("local:example", {}, NOVEMBER) == 0


def test_all_reported_calls_are_counted():
    calls = usage()
    calls["second"] = calls["model"].copy()

    assert api_cost("google_genai:gemini-3.8-flash", calls, NOVEMBER) == pytest.approx(
        2 * api_cost("google_genai:gemini-3.8-flash", usage(), NOVEMBER)
    )
