"""How the models are reached, which the study treats as held constant.

Section 5.1 claims one gateway serves every model, so that the client, the
request format, the retry policy and the network path are shared and the model
is the only thing that differs between arms. The commercial pair used to go
straight to api.openai.com on a second key, which made the claim false. These
tests pin the gateway as the default and the escape hatch as deliberate.
"""
from __future__ import annotations

import json
import os

import pytest

from app import config
from app.web import web_server as w

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GATEWAY = "https://openrouter.ai/api/v1"
DIRECT = "https://api.openai.com/v1"


@pytest.fixture
def models_under():
    """load_models() under a chosen route, with the setting restored after."""
    before = config.MODEL_ROUTE

    def load(route: str) -> dict:
        config.MODEL_ROUTE = route
        return w.load_models()

    yield load
    config.MODEL_ROUTE = before


def test_the_default_sends_every_model_through_one_gateway(models_under):
    models = models_under("gateway")
    assert models, "no models declared"
    for mid, cfg in models.items():
        assert cfg["base_url"] == GATEWAY, f"{mid} bypasses the gateway"
        assert cfg["api_key_env"] == "OPENROUTER_API_KEY", f"{mid} uses another key"
        assert cfg["_route"] == "gateway", mid


def test_direct_restores_the_vendor_path_for_the_models_that_declare_one(models_under):
    models = models_under("direct")
    assert models["gpt-4.1-mini"]["base_url"] == DIRECT
    assert models["gpt-5"]["base_url"] == DIRECT
    for mid in ("gpt-4.1-mini", "gpt-5"):
        assert models[mid]["api_key_env"] == "OPENAI_API_KEY", mid
        assert models[mid]["_route"] == "direct", mid
        assert "/" not in models[mid]["model"], (
            f"{mid} keeps its gateway-qualified id on the direct route")
    # A model with no direct route is unaffected by the switch.
    for mid in ("llama-3.3-70b", "qwen3.5-9b"):
        assert models[mid]["base_url"] == GATEWAY, mid
        assert models[mid]["_route"] == "gateway", mid


def test_the_route_does_not_move_the_price(models_under):
    """OpenRouter lists the GPT pair at OpenAI's prices, so cost is unaffected.

    If that ever stops being true, the cost analysis of RQ3 changes with the
    route and this test is the place that says so.
    """
    gw, direct = models_under("gateway"), models_under("direct")
    for mid in ("gpt-4.1-mini", "gpt-5"):
        assert gw[mid]["price_in"] == direct[mid]["price_in"], mid
        assert gw[mid]["price_out"] == direct[mid]["price_out"], mid


def test_the_label_says_which_route_is_in_use(models_under):
    """The panel shows the label, and a mixed campaign has to be visible."""
    gw, direct = models_under("gateway"), models_under("direct")
    assert "OpenRouter" in gw["gpt-5"]["label"]
    assert "direct" in direct["gpt-5"]["label"].lower()


def test_an_unknown_route_falls_back_to_the_gateway(models_under):
    """A typo must not silently split the campaign across two paths."""
    models = models_under("something-new")
    assert models["gpt-5"]["base_url"] == GATEWAY
    assert models["gpt-5"]["_route"] == "gateway"


def test_every_declared_direct_route_names_a_key_and_a_model():
    """An incomplete block would send the run to the gateway id on OpenAI."""
    with open(os.path.join(_ROOT, "models.json"), encoding="utf-8") as fh:
        entries = json.load(fh)["models"]
    declared = [m for m in entries if m.get("route_direct")]
    assert declared, "no model declares a direct route any more"
    for m in declared:
        block = m["route_direct"]
        for field in ("model", "base_url", "api_key_env"):
            assert block.get(field), f"{m['id']}.route_direct lacks {field}"


def test_the_route_is_recorded_with_every_run():
    """A campaign that mixes the routes is not one campaign."""
    from app.benchmark.store import COLUMNS
    assert "model_route" in COLUMNS
