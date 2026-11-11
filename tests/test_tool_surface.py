"""
The published MCP surface is experimental apparatus, so it is pinned by tests.

Descriptions and parameter schemas are what induce the behaviour this study
measures — no model here is fine-tuned. A silent rename, a lost default or a
dropped parameter description would change the measurement without changing
any result, which is the failure mode hardest to notice after the fact. These
tests fail loudly instead.
"""
from __future__ import annotations

import pytest

from app import config
from app.mcp import server

EXPECTED_TOOLS = {
    "geocode_address",
    "find_shelters_near",
    "travel_time_to_shelter",
    "shelter_details",
    "compute_isochrone",
    "shelters_in_area",
    "population_in_area",
    "shelter_service_area",
    "reverse_geocode",
}


def _tools() -> dict:
    return {t.name: t for t in server.mcp._tool_manager.list_tools()}


def test_exactly_the_specified_tools():
    assert set(_tools()) == EXPECTED_TOOLS


@pytest.mark.parametrize("name,params", [
    ("geocode_address", {"query"}),
    ("find_shelters_near", {"lat", "lon", "radius_m"}),
    ("travel_time_to_shelter", {"lat", "lon", "shelter_id", "speed_profile"}),
    ("shelter_details", {"shelter_id"}),
    ("compute_isochrone", {"lat", "lon", "minutes", "speed_profile"}),
    ("shelters_in_area", {"polygon"}),
    ("population_in_area", {"polygon"}),
    ("shelter_service_area", {"shelter_id", "minutes", "speed_profile"}),
    ("reverse_geocode", {"lat", "lon"}),
])
def test_parameter_names(name, params):
    schema = _tools()[name].parameters
    assert set(schema.get("properties", {})) == params


@pytest.mark.parametrize("name,required", [
    ("geocode_address", {"query"}),
    ("find_shelters_near", {"lat", "lon"}),
    ("travel_time_to_shelter", {"lat", "lon", "shelter_id"}),
    ("shelter_details", {"shelter_id"}),
    ("compute_isochrone", {"lat", "lon"}),
    ("shelters_in_area", {"polygon"}),
    ("population_in_area", {"polygon"}),
    ("shelter_service_area", {"shelter_id"}),
    ("reverse_geocode", {"lat", "lon"}),
])
def test_required_parameters(name, required):
    schema = _tools()[name].parameters
    assert set(schema.get("required", [])) == required


def test_defaults_match_the_specification():
    props = _tools()["find_shelters_near"].parameters["properties"]
    assert props["radius_m"]["default"] == 800

    iso = _tools()["compute_isochrone"].parameters["properties"]
    assert iso["minutes"]["default"] == 10
    assert iso["speed_profile"]["default"] == "adult"

    svc = _tools()["shelter_service_area"].parameters["properties"]
    assert svc["minutes"]["default"] == 10
    assert svc["speed_profile"]["default"] == "adult"


@pytest.mark.skipif(config.TOOL_DESCRIPTIONS == "bare",
                    reason="the bare arm publishes no descriptions at all, by design")
def test_every_parameter_carries_a_description():
    """A parameter with no description is a parameter the model must guess."""
    for name, tool in _tools().items():
        for param, spec in tool.parameters.get("properties", {}).items():
            assert spec.get("description"), f"{name}.{param} has no description"


@pytest.mark.skipif(config.TOOL_DESCRIPTIONS != "rich",
                    reason="the cues are what the minimal set deliberately removes")
def test_speed_profile_descriptions_carry_the_natural_language_cues():
    """The mapping from 'my elderly mother' to `reduced_mobility` has to come
    from somewhere, and the only place it can come from is here."""
    for name in ("travel_time_to_shelter", "compute_isochrone", "shelter_service_area"):
        text = _tools()[name].parameters["properties"]["speed_profile"]["description"].lower()
        assert "elderly" in text
        assert "urgen" in text          # urgency / urgent
        for value in config.SPEED_PROFILES:
            assert value in text


def test_descriptions_never_leak_the_database():
    """The model must not be able to learn the schema from the tool surface.

    Whole words only: "geometry" is domain vocabulary the specification uses,
    "geom" is a column name. The distinction is the point of the rule.
    """
    import re

    fragments = ("select ", "postgis", "st_", "adaposturi", "enviroo.")
    words = ("geom", "table", "column", "sql", "schema", "join")
    for name, tool in _tools().items():
        blob = (tool.description or "").lower()
        for param in tool.parameters.get("properties", {}).values():
            blob += " " + str(param.get("description", "")).lower()
        for term in fragments:
            assert term not in blob, f"{name} leaks implementation detail: {term!r}"
        for term in words:
            assert not re.search(rf"\b{term}\b", blob), \
                f"{name} leaks implementation detail: {term!r}"


def test_descriptions_fit_the_provider_limit():
    """Descriptions are truncated at 1024 characters on the way to the model;
    a truncated description is a changed experiment."""
    for name, tool in _tools().items():
        assert len(tool.description or "") <= 1024, name


@pytest.mark.skipif(config.TOOL_DESCRIPTIONS != "rich",
                    reason="usage guidance is what the minimal set deliberately removes")
def test_the_two_geocoding_tools_state_their_direction():
    """Two tools whose names differ by one word are the easiest pair to confuse.
    Nothing but the descriptions keeps them apart, so the direction each one
    runs in has to be stated in both."""
    fwd = _tools()["geocode_address"].description.lower()
    rev = _tools()["reverse_geocode"].description.lower()
    assert "coordinates" in fwd and "coordinates" in rev
    assert "inverse" in rev
    # The wasteful chain is point -> name -> point again; the reverse tool has
    # to say it is not a step towards the others.
    assert "not a step" in rev


# ── The second description set ────────────────────────────────────────────
# The minimal set is the other arm of the description experiment. It is a
# separate table, so it can drift from the published surface without anything
# failing at runtime until a benchmark run silently mixes the two regimes.

def test_minimal_set_covers_exactly_the_published_tools():
    from app.mcp.descriptions_minimal import MINIMAL
    assert set(MINIMAL) == set(_tools()), (
        "the minimal set and the published surface have diverged")


def test_minimal_set_describes_every_parameter():
    """A parameter left undescribed is not `minimal`, it is broken: the arm
    would measure a missing field rather than the absence of guidance."""
    from app.mcp.descriptions_minimal import MINIMAL
    for name, tool in _tools().items():
        described = set(MINIMAL[name].get("params", {}))
        published = set(tool.parameters.get("properties", {}))
        assert described == published, f"{name}: {published ^ described}"


def test_minimal_set_keeps_the_enum_values_but_drops_the_cues():
    """The line between minimal and broken.

    `speed_profile` has no enum in its schema, so its permitted values live
    only in the description: remove them and the tool cannot be called at all,
    which is breakage, not minimalism. What must go is any hint about WHICH
    value a given sentence calls for - that mapping is the thing under test.
    """
    from app.mcp.descriptions_minimal import MINIMAL
    for name in ("travel_time_to_shelter", "compute_isochrone", "shelter_service_area"):
        text = MINIMAL[name]["params"]["speed_profile"].lower()
        for value in config.SPEED_PROFILES:
            assert value in text, f"{name}: {value} is not callable without being named"
        for cue in ("elderly", "child", "slowly", "urgen", "hurry", "default", "otherwise"):
            assert cue not in text, f"{name}: the minimal set still carries the cue {cue!r}"


def test_minimal_set_is_substantially_shorter():
    """If the two arms were close in size, the experiment would have no contrast."""
    from app.mcp.descriptions_minimal import MINIMAL
    minimal = sum(len(e["description"]) + sum(len(v) for v in e["params"].values())
                  for e in MINIMAL.values())
    published = sum(len(t.description or "")
                    + sum(len(p.get("description", ""))
                          for p in t.parameters.get("properties", {}).values())
                    for t in _tools().values())
    if config.TOOL_DESCRIPTIONS == "rich":
        assert minimal * 3 < published, (
            f"minimal={minimal} vs rich={published}: too close to contrast")


# -- Guidance is held constant; only the descriptions move -----------------
# The descriptions are one of several layers that can tell a model what to do.
# The others - the `message` of a failed call, the `note` of a successful one,
# and the system prompt - are now settings of their own, held constant across
# the arms, so that `rich` and `bare` differ in their published descriptions
# and in nothing else. These tests pin that separation down.

def _envelope_under(regime: str, guidance: str = "on"):
    """The envelope module's output under a regime and guidance setting."""
    from app import config
    from app.mcp import envelope
    before_r, before_g = config.TOOL_DESCRIPTIONS, config.GUIDANCE_RUNTIME
    config.TOOL_DESCRIPTIONS, config.GUIDANCE_RUNTIME = regime, guidance
    try:
        return (envelope.failure("geocoding_ambiguous", "Ask the user which one they mean.",
                                 candidates=[{"lat": 1.0}]),
                envelope.ok("snap", demand_ratio=None, note="A null is not a zero."))
    finally:
        config.TOOL_DESCRIPTIONS, config.GUIDANCE_RUNTIME = before_r, before_g


def test_runtime_guidance_does_not_follow_the_description_regime():
    """With guidance `on`, a bare-description arm still gets message and note.

    This is the point of the separation: the service produces those strings at
    runtime and does not publish them, so emptying the descriptions must not
    silently empty them too.
    """
    for regime in ("rich", "minimal", "bare"):
        fail, okay = _envelope_under(regime, guidance="on")
        assert fail["message"], f"{regime} lost its error message"
        assert okay["note"], f"{regime} lost its note"
        assert fail["candidates"] == [{"lat": 1.0}]
        assert "demand_ratio" in okay and okay["demand_ratio"] is None


def test_runtime_guidance_off_strips_advice_but_keeps_facts():
    for regime in ("rich", "minimal", "bare"):
        fail, okay = _envelope_under(regime, guidance="off")
        assert fail["error_code"] == "geocoding_ambiguous"
        assert "message" not in fail, f"{regime} is still telling the model what to do"
        assert "note" not in okay, f"{regime} is still explaining its nulls"
        # Facts about the result are not advice: they stay either way.
        assert fail["candidates"] == [{"lat": 1.0}]
        assert "demand_ratio" in okay and okay["demand_ratio"] is None


def test_guidance_auto_restores_the_original_coupling():
    """`auto` reproduces the first campaign, where guidance followed the regime."""
    fail, okay = _envelope_under("rich", guidance="auto")
    assert fail["message"] and okay["note"]
    for regime in ("minimal", "bare", "something-new"):
        fail, okay = _envelope_under(regime, guidance="auto")
        assert "message" not in fail, regime
        assert "note" not in okay, regime


def test_one_system_prompt_serves_every_regime():
    """The arms must receive a byte-identical system prompt.

    The prompt was a second experimental factor: pairing a guided prompt with
    the documented arm meant the measured difference was not the descriptions'
    alone. One prompt for every regime is what removes it.
    """
    from app.web.web_server import SYSTEM_SHARED, _system_for, prompt_fingerprint
    for regime in ("rich", "minimal", "bare", "mixed", "something-new"):
        assert _system_for(regime) is SYSTEM_SHARED, regime
    assert prompt_fingerprint("rich") == prompt_fingerprint("bare")


def test_the_shared_prompt_leaks_nothing_that_is_under_test():
    """It may carry conduct, but no rule the descriptions are measured on.

    Every term below is stated in a published description and is what one of
    the benchmark tasks turns on. If the prompt named any of them, the bare arm
    would be handed back exactly what emptying the descriptions took away.
    """
    from app.web.web_server import SYSTEM_SHARED
    # Collapsed, because the prompt is hard-wrapped: `do not estimate` spans
    # two lines in the source and would not be found as a substring.
    lowered = " ".join(SYSTEM_SHARED.lower().split())
    for leak in ("metre", "800", "5,000", "5 km", "non-functional", "usable",
                 "nominal", "reduced_mobility", "km/h", "ratio", "capacity",
                 "isochrone", "straight line", "travel_time", "find_shelters",
                 "shelter_service_area", "population_in_area"):
        assert leak not in lowered, f"the shared prompt states something under test: {leak!r}"
    # The role and the conduct that holds for any tool-using assistant stay.
    assert "civil-shelter accessibility" in SYSTEM_SHARED
    assert "language the user writes in" in SYSTEM_SHARED
    for conduct in ("from memory", "do not estimate", "never substitute",
                    "undefined value", "returned nothing"):
        assert conduct in lowered, f"the shared prompt lost its conduct: {conduct!r}"


def test_legacy_prompt_pairing_is_still_reproducible():
    """The first campaign's pairing survives behind ENVIROO_SYSTEM_PROMPT=legacy."""
    from app import config as cfg
    from app.web.web_server import SYSTEM, SYSTEM_MINIMAL, _system_for
    before = cfg.SYSTEM_PROMPT_REGIME
    cfg.SYSTEM_PROMPT_REGIME = "legacy"
    try:
        assert _system_for("rich") is SYSTEM
        for regime in ("minimal", "bare", "something-new"):
            assert _system_for(regime) is SYSTEM_MINIMAL, regime
    finally:
        cfg.SYSTEM_PROMPT_REGIME = before





# ── The bare set, and the ordering of the three arms ──────────────────────

def test_bare_set_covers_exactly_the_published_tools():
    from app.mcp.descriptions_bare import BARE
    assert set(BARE) == set(_tools())


def test_bare_set_describes_every_parameter():
    from app.mcp.descriptions_bare import BARE
    for name, tool in _tools().items():
        assert set(BARE[name]["params"]) == set(tool.parameters.get("properties", {})), name


def test_bare_publishes_nothing_at_all():
    """Every description empty - the name and the schema are the whole surface."""
    from app.mcp.descriptions_bare import BARE
    for name, entry in BARE.items():
        assert entry["description"] == "", name
        assert all(v == "" for v in entry["params"].values()), name


def test_the_enum_survives_in_the_schema_not_in_prose():
    """The callability floor, moved where it belongs.

    speed_profile has no free-text values: it is an enum, and the schema says
    so in every regime. Without this the bare arm could not call the tool at
    all, and it would be measuring breakage instead of the absence of
    documentation.
    """
    for name in ("travel_time_to_shelter", "compute_isochrone", "shelter_service_area"):
        spec = _tools()[name].parameters["properties"]["speed_profile"]
        assert spec.get("enum") == list(config.SPEED_PROFILES), (
            f"{name}: the permitted values are not in the schema")


def test_the_three_arms_are_ordered_and_far_apart():
    """Three points on one axis only mean something if they are separated."""
    from app.mcp.descriptions_bare import BARE
    from app.mcp.descriptions_minimal import MINIMAL
    size = lambda d: sum(len(e["description"]) + sum(len(v) for v in e["params"].values())
                         for e in d.values())
    bare, minimal = size(BARE), size(MINIMAL)
    assert bare == 0, "the bare arm is meant to publish nothing"
    rich = sum(len(t.description or "")
               + sum(len(p.get("description", ""))
                     for p in t.parameters.get("properties", {}).values())
               for t in _tools().values())
    if config.TOOL_DESCRIPTIONS == "rich":
        assert bare < minimal < rich / 3, f"bare={bare} minimal={minimal} rich={rich}"


def test_an_unrecognised_arm_fails_closed_under_auto():
    """A new arm must not inherit guidance by being unrecognised.

    Under `auto` the switch asks whether the regime IS rich rather than whether
    it is not minimal, so an unknown value fails closed - without documentation
    - which is the safe direction for an experiment about documentation.
    """
    from app import config as cfg
    from app.mcp import envelope
    before_r, before_g = cfg.TOOL_DESCRIPTIONS, cfg.GUIDANCE_RUNTIME
    cfg.GUIDANCE_RUNTIME = "auto"
    try:
        for regime in ("minimal", "bare", "something-new"):
            cfg.TOOL_DESCRIPTIONS = regime
            assert "message" not in envelope.failure("x", "advice"), regime
            assert "note" not in envelope.ok("snap", note="advice"), regime
    finally:
        cfg.TOOL_DESCRIPTIONS, cfg.GUIDANCE_RUNTIME = before_r, before_g
