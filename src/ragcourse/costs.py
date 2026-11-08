"""Estimate standard Gemini API token charges from reported usage, in USD."""

from datetime import date

PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"
PRICING_CHECKED = "08/11/2026"

# Standard input / output / cached input, USD per million tokens.
# Google publishes these introductory rates and the dates they double.
PRICES = {
    "gemini-3.8-flash": ((0.75, 3.75, 0.075), date(2027, 1, 1)),
    "gemini-4-argon": ((2.0, 10.0, 0.10), date(2026, 12, 1)),
}


def api_cost(model: str, usage: dict, day: date | None = None) -> float | None:
    """Sum reported calls; missing usage or an unknown price means unknown cost."""
    if model.startswith("local:") or model == "google_genai:gemma-4-31b-it":
        return 0.0
    name = model.removeprefix("google_genai:")
    if name not in PRICES or not usage:
        return None
    rates, increase = PRICES[name]
    multiplier = 2 if (day or date.today()) >= increase else 1
    input_rate, output_rate, cached_rate = rates
    total = 0.0
    for tokens in usage.values():
        if "input_tokens" not in tokens or "output_tokens" not in tokens:
            return None
        inputs, outputs = tokens["input_tokens"], tokens["output_tokens"]
        cached = tokens.get("input_token_details", {}).get("cache_read", 0)
        cached = min(cached, inputs)
        # LangChain includes reasoning in output_tokens; do not add it again.
        total += (inputs - cached) * input_rate + cached * cached_rate
        total += outputs * output_rate
    return total * multiplier / 1_000_000
