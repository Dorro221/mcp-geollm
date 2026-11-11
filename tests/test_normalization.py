"""
The normalization rules of the ETL and the validators of the tool boundary.

These encode research decisions, not conveniences: an unrecorded capacity is
not zero, an unrecorded status is not "functional", and a walking time outside
the supported range is refused rather than quietly replaced by one that is.
"""
from __future__ import annotations

import pytest

from app import config
from app.mcp import envelope, server
from app.mcp.providers import routing
from db import etl_shelters as etl


# ── Status and type ───────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("Funcțional", config.STATUS_FUNCTIONAL),
    ("Parțial-funcțional", config.STATUS_PARTIAL),
    ("Nefuncțional", config.STATUS_NON_FUNCTIONAL),
    ("-", config.STATUS_UNKNOWN),
    ("", config.STATUS_UNKNOWN),
    (None, config.STATUS_UNKNOWN),
    ("something else entirely", config.STATUS_UNKNOWN),
])
def test_status_normalization(raw, expected):
    assert etl.normalize_status(raw) == expected


def test_unknown_status_is_never_treated_as_functional():
    """Assuming a shelter works is the one error with real consequences."""
    assert etl.normalize_status("???") not in config.USABLE_STATUSES


@pytest.mark.parametrize("raw,expected", [
    ("Public", "public"), ("public", "public"), ("PUBLIC", "public"),
    ("Privat", "private"), ("-", "unknown"), (None, "unknown"),
])
def test_type_normalization(raw, expected):
    assert etl.normalize_type(raw) == expected


# ── Capacity ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("150", 150), (" 80 ", 80), ("87 persoane", 87),
    ("-", None), ("", None), (None, None), ("0", None),
])
def test_capacity_parsing(raw, expected):
    assert etl.normalize_capacity(raw) == expected


def test_unrecorded_capacity_stays_unrecorded():
    """'No capacity recorded' and 'no places' are different facts; merging
    them would fabricate the second."""
    assert etl.normalize_capacity("-") is None
    assert etl.effective_capacity(None, config.STATUS_FUNCTIONAL) is None


@pytest.mark.parametrize("nominal,status,expected", [
    (100, config.STATUS_FUNCTIONAL, 100),
    (100, config.STATUS_PARTIAL, 50),
    (100, config.STATUS_NON_FUNCTIONAL, 0),
    (100, config.STATUS_UNKNOWN, 0),
])
def test_effective_capacity(nominal, status, expected):
    assert etl.effective_capacity(nominal, status) == expected


def test_shelter_identifier_is_stable_and_padded():
    assert etl.shelter_identifier(42) == "S00042"
    assert etl.shelter_identifier(6117) == "S06117"


# ── Coordinates ───────────────────────────────────────────────────────────

def test_coordinates_inside_the_study_area_pass_through():
    assert envelope.check_coordinates(46.7712, 23.5900) == (46.7712, 23.5900)


@pytest.mark.parametrize("lat,lon", [
    (48.9, 2.35),        # Paris
    (23.59, 46.77),      # lat/lon swapped
    (0, 0),
])
def test_coordinates_outside_the_study_area_are_refused(lat, lon):
    with pytest.raises(envelope.ToolFailure) as exc:
        envelope.check_coordinates(lat, lon)
    assert exc.value.error_code == envelope.INVALID_COORDINATES


def test_swapped_coordinates_get_a_hint_about_swapping():
    with pytest.raises(envelope.ToolFailure) as exc:
        envelope.check_coordinates(23.59, 46.77)
    assert "swapped" in exc.value.message


# ── Speed profiles and time budgets ───────────────────────────────────────

def test_speed_profile_default_and_validation():
    assert envelope.check_speed_profile(None) == "adult"
    assert envelope.check_speed_profile("REDUCED_MOBILITY") == "reduced_mobility"
    with pytest.raises(envelope.ToolFailure) as exc:
        envelope.check_speed_profile("sprinting")
    assert exc.value.error_code == envelope.INVALID_PROFILE


def test_profiles_differ_by_exactly_the_specified_speeds():
    """Ten minutes of walking covers 500 m, 833 m and 1167 m; the profile is
    the only thing that varies."""
    assert routing.minutes_for(1000, "adult") == 12.0
    assert routing.minutes_for(1000, "reduced_mobility") == 20.0
    assert routing.minutes_for(1000, "hurried") == pytest.approx(8.6, abs=0.05)


def test_minutes_out_of_range_is_not_precomputed_not_a_substitution():
    for value in (2, 45, 100):
        with pytest.raises(envelope.ToolFailure) as exc:
            envelope.check_minutes(value)
        assert exc.value.error_code == envelope.NOT_PRECOMPUTED
        assert str(config.MIN_MINUTES) in exc.value.message
        assert str(config.MAX_MINUTES) in exc.value.message


def test_minutes_default_and_range():
    assert envelope.check_minutes(None) == float(config.DEFAULT_MINUTES)
    assert envelope.check_minutes(5) == 5.0
    assert envelope.check_minutes(30) == 30.0


# ── Radius ────────────────────────────────────────────────────────────────

def test_radius_default_and_clamping():
    assert envelope.clamp_radius(None) == (config.DEFAULT_RADIUS_M, None)
    assert envelope.clamp_radius(1200) == (1200, None)
    radius, note = envelope.clamp_radius(99000)
    assert radius == config.MAX_RADIUS_M
    assert note and "5000" in note      # clamped, and said so


def test_zero_radius_is_refused():
    with pytest.raises(envelope.ToolFailure):
        envelope.clamp_radius(0)


# ── Polygons ──────────────────────────────────────────────────────────────

SQUARE = {"type": "Polygon",
          "coordinates": [[[23.58, 46.76], [23.60, 46.76], [23.60, 46.78],
                           [23.58, 46.78], [23.58, 46.76]]]}


def test_polygon_accepted_as_object_and_as_json_text():
    import json
    assert server._as_polygon(SQUARE)
    assert server._as_polygon(json.dumps(SQUARE))


def test_feature_is_refused_with_an_actionable_message():
    """Passing the whole isochrone response instead of its polygon is a real
    modelling error; hiding it would remove it from the measurement."""
    with pytest.raises(envelope.ToolFailure) as exc:
        server._as_polygon({"type": "Feature", "geometry": SQUARE})
    assert exc.value.error_code == envelope.INVALID_GEOMETRY
    assert "polygon" in exc.value.message.lower()


@pytest.mark.parametrize("value", [
    None, 42, "not json at all",
    {"type": "Point", "coordinates": [23.5, 46.7]},
    {"type": "Polygon"},
])
def test_malformed_areas_are_refused(value):
    with pytest.raises(envelope.ToolFailure) as exc:
        server._as_polygon(value)
    assert exc.value.error_code == envelope.INVALID_GEOMETRY


# ── Envelope ──────────────────────────────────────────────────────────────

def test_success_envelope_carries_snapshot_and_units():
    payload = envelope.ok("snap-1", count=3)
    assert payload["snapshot_id"] == "snap-1"
    assert payload["units"] == {"distance": "m", "time": "min"}
    assert payload["count"] == 3


def test_failure_shape_has_no_envelope():
    """A failure carries its code and its advice, and nothing else.

    The advice is guidance, so it belongs to the documented regime alone: the
    reduced arms publish the code by itself. Pinned for both, because the
    shape is what the specification fixes - neither arm may acquire a
    `snapshot_id` or a `units` block on the way out.
    """
    from app import config

    payload = envelope.failure(envelope.SHELTER_NOT_FOUND, "no such shelter")
    if config.TOOL_DESCRIPTIONS == "rich":
        assert payload == {"error_code": "shelter_not_found", "message": "no such shelter"}
    else:
        assert payload == {"error_code": "shelter_not_found"}


def test_capacity_totals_never_merge_nominal_and_effective():
    rows = [
        {"capacity_nominal": 100, "capacity_effective": 50},
        {"capacity_nominal": 200, "capacity_effective": 200},
        {"capacity_nominal": None, "capacity_effective": None},
    ]
    totals = server._capacity_totals(rows)
    assert totals["total_capacity_nominal"] == 300
    assert totals["total_capacity_effective"] == 250
    # An unrecorded capacity is declared, not silently folded into the total.
    assert totals["shelters_with_unrecorded_capacity"] == 1
