"""
ENVIROO — MCP server for civil-shelter accessibility analysis.

Eight tools, deliberately small and deliberately incomplete on their own:
spatial search is separate from routing, and computing a walking-time polygon
is separate from measuring what is inside it. Any question worth asking
therefore requires composing several tools, which is precisely what the study
measures.

No model is fine-tuned here. Every behaviour the system needs — asking for a
missing location, choosing between two similar tools, reading "my elderly
mother" as `reduced_mobility` — has to be induced by the tool descriptions
and parameter schemas below. They are experimental apparatus, not
documentation: change the wording and you change the measurement.

Transport: Streamable HTTP at /mcp (default port 9100).
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any, Callable

from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app import config
from app.logging_utils import RequestIdMiddleware, setup_logging
from app.mcp import repository, toollog
from app.mcp.envelope import (
    AREA_TOO_LARGE,
    INTERNAL_ERROR,
    INVALID_GEOMETRY,
    NOT_PRECOMPUTED,
    SHELTER_NOT_FOUND,
    ToolFailure,
    check_coordinates,
    check_minutes,
    check_speed_profile,
    clamp_radius,
    failure,
    ok,
)
from app.mcp.descriptions_bare import BARE as BARE_DESCRIPTIONS
from app.mcp.descriptions_minimal import MINIMAL as MINIMAL_DESCRIPTIONS
from app.mcp.fastmcp_utils import apply_descriptions, dedent_tool_descriptions
from app.mcp.providers import geocoding, routing

log = setup_logging("enviroo_mcp")

mcp = FastMCP(name="enviroo-shelter-mcp")


# ── Shared parameter annotations ──────────────────────────────────────────
# The natural-language cues that map onto the enum values live here. The
# study measures exactly this extraction, so they are written once and reused
# rather than paraphrased per tool.

SPEED_PROFILE_DESC = (
    "Walking speed. Use `reduced_mobility` for an elderly person, a small child, "
    "someone with a disability, or anyone described as walking slowly or with "
    "difficulty; `hurried` when the user conveys urgency or asks how fast they "
    "could get there; `adult` otherwise. Speeds: reduced_mobility 3 km/h, "
    "adult 5 km/h, hurried 7 km/h."
)
MINUTES_DESC = (
    f"Walking time budget in minutes. Supported values range from "
    f"{config.MIN_MINUTES} to {config.MAX_MINUTES}."
)
POLYGON_DESC = (
    "Area to search, as a GeoJSON Polygon — typically the `polygon` field returned "
    "by compute_isochrone. Pass the polygon object itself, not the whole response "
    "and not a GeoJSON Feature."
)

# The permitted values live in the SCHEMA, not only in the prose. The
# specification calls this an enum; stating it as one means a regime can empty
# the description without making the tool impossible to call, and it is the
# same constraint in all three arms so they differ in text alone.
SpeedProfile = Annotated[str, Field(description=SPEED_PROFILE_DESC,
                                    json_schema_extra={"enum": list(config.SPEED_PROFILES)})]
Minutes = Annotated[float, Field(description=MINUTES_DESC)]


# ── Tool boundary ─────────────────────────────────────────────────────────

async def _current_snapshot_id() -> str:
    try:
        return await repository.snapshot_id()
    except Exception:
        return "unavailable"


async def _boundary(name: str, args: dict[str, Any],
                    impl: Callable[..., Any], *impl_args) -> dict[str, Any]:
    """Run one tool call: log it, and convert every outcome into a response.

    Exactly two shapes leave here — the success envelope or the failure
    object. An unexpected exception becomes an explicit failure rather than a
    plausible-looking number, because a model cannot tell the difference and
    a reader of the study cannot either.
    """
    snapshot = await _current_snapshot_id()
    with toollog.record(name, args, snapshot) as holder:
        try:
            result = await impl(*impl_args)
        except ToolFailure as tf:
            result = tf.as_response()
            log.info("%s -> %s: %s", name, tf.error_code, tf.message)
        except Exception as exc:            # noqa: BLE001 - reported, never hidden
            log.exception("%s failed unexpectedly", name)
            result = failure(
                INTERNAL_ERROR,
                f"The '{name}' operation could not be completed ({type(exc).__name__}). "
                "Report that the computation failed; do not estimate the answer.",
            )
        holder[0] = result
        return result


# ── Shaping helpers ───────────────────────────────────────────────────────

def _shelter_row(row: dict, *, with_distance: bool = False) -> dict:
    """One shelter as every tool reports it.

    `capacity_nominal` and `capacity_effective` are always both present and
    never merged: the gap between them is a finding of the study, not a
    detail to tidy away.
    """
    out = {
        "shelter_id": row["shelter_id"],
        "address": row.get("address") or "",
        "locality": row.get("locality") or "",
        "county": row.get("county") or "",
    }
    if with_distance:
        out["distance_m"] = row.get("distance_m")
    out["capacity_nominal"] = row.get("capacity_nominal")
    out["capacity_effective"] = row.get("capacity_effective")
    out["type"] = row.get("type")
    out["status"] = row.get("status")
    return out


def _capacity_totals(rows: list[dict]) -> dict:
    nominal = sum(r["capacity_nominal"] for r in rows if r.get("capacity_nominal") is not None)
    effective = sum(r["capacity_effective"] for r in rows if r.get("capacity_effective") is not None)
    unknown = sum(1 for r in rows if r.get("capacity_nominal") is None)
    totals = {"total_capacity_nominal": nominal, "total_capacity_effective": effective}
    if unknown:
        # Silence here would misrepresent the total as complete.
        totals["shelters_with_unrecorded_capacity"] = unknown
    return totals


# The population layer need not cover the whole study area. Below this
# fraction of a polygon lying over it, the answer says so; at zero, no number
# is produced at all - the same rule the routing tools follow when a shelter
# cannot be reached.
POPULATION_COVERAGE_MIN = 0.99


async def _population_note(geojson: str) -> tuple[float, str | None]:
    """Coverage fraction for a polygon, and the note it obliges, if any."""
    covered = await repository.population_coverage_fraction(geojson)
    if covered <= 0.0:
        return covered, (
            "This area lies outside the population layer, which covers Cluj-Napoca "
            "only. The number of residents here is unknown - report that it is "
            "unknown, and do not treat it as zero."
        )
    if covered < POPULATION_COVERAGE_MIN:
        return covered, (
            f"Only {covered * 100:.0f}% of this area is covered by the population "
            "layer, which covers Cluj-Napoca only. The figure counts the covered "
            "part alone and therefore understates the true total; say so rather "
            "than presenting it as complete."
        )
    return covered, None


def _as_polygon(value: Any) -> str:
    """Accept a GeoJSON Polygon and return it as text for PostGIS.

    A JSON string is parsed — that is a transport quirk, not a modelling
    error. A Feature or a whole tool response is refused with an explanatory
    message: passing the wrong object is a real mistake and hiding it would
    remove it from the measurement.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise ToolFailure(
                INVALID_GEOMETRY,
                "The supplied area is not valid GeoJSON. Pass the `polygon` object "
                "returned by compute_isochrone.",
            )
    if not isinstance(value, dict):
        raise ToolFailure(
            INVALID_GEOMETRY,
            "The supplied area must be a GeoJSON Polygon object, for example "
            '{"type": "Polygon", "coordinates": [[[lon, lat], ...]]}.',
        )
    geom_type = str(value.get("type", "")).lower()
    if geom_type in ("feature", "featurecollection"):
        raise ToolFailure(
            INVALID_GEOMETRY,
            "A GeoJSON Feature was supplied instead of a Polygon. Pass the `polygon` "
            "field returned by compute_isochrone, which is the geometry itself.",
        )
    if geom_type not in ("polygon", "multipolygon"):
        raise ToolFailure(
            INVALID_GEOMETRY,
            f"The supplied area is of type '{value.get('type')}'; a Polygon is "
            "required. Use compute_isochrone to obtain one.",
        )
    if not value.get("coordinates"):
        raise ToolFailure(
            INVALID_GEOMETRY, "The supplied polygon has no coordinates."
        )
    return json.dumps(value, ensure_ascii=False)


async def _validated_area(value: Any) -> tuple[str, float]:
    """Validate, repair and measure a supplied polygon.

    Returns the GeoJSON text and its area in km². Refuses anything larger
    than the configured limit: a city-wide polygon turns a spatial join into
    a full scan and turns a specific question into a meaningless one.
    """
    geojson = _as_polygon(value)
    try:
        info = await repository.inspect_polygon(geojson)
    except Exception as exc:
        raise ToolFailure(
            INVALID_GEOMETRY,
            "The supplied polygon could not be read. Check that it is a GeoJSON "
            "Polygon with [longitude, latitude] pairs in WGS 84.",
        ) from exc
    if not info or info.get("is_empty"):
        raise ToolFailure(
            INVALID_GEOMETRY, "The supplied polygon is empty after validation."
        )
    area_km2 = float(info.get("area_km2") or 0.0)
    if area_km2 > config.MAX_AREA_KM2:
        raise ToolFailure(
            AREA_TOO_LARGE,
            f"The supplied area covers {area_km2:.1f} km², above the limit of "
            f"{config.MAX_AREA_KM2:g} km². Ask about a smaller area — for example "
            "the area reachable on foot from a specific address.",
        )
    return geojson, round(area_km2, 3)


# ══════════════════════════════════════════════════════════════════════════
# 1. geocode_address
# ══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def geocode_address(
    query: Annotated[str, Field(description=(
        "Address, landmark or place name, for example \"Strada Memorandumului 21\" "
        "or \"Piata Unirii\". Include the street number when the user gave one, and "
        "the town when it is known."
    ))],
) -> dict[str, Any]:
    """Converts a free-text address, landmark or place name in the study area into
    geographic coordinates. Use this first whenever the user refers to a location by
    name rather than by coordinates. If the user has not given any location at all, ask
    them for one - do not guess a location and do not use a well-known landmark as a
    stand-in. If the result matches only a street or a wider area rather than an exact
    address, say so when reporting the answer, because everything computed from it
    inherits that imprecision.
    """
    return await _boundary("geocode_address", {"query": query}, _geocode_address, query)


async def _geocode_address(query: str) -> dict[str, Any]:
    match = await geocoding.provider.geocode(query)
    return ok(await repository.snapshot_id(), **match)


# ══════════════════════════════════════════════════════════════════════════
# 2. find_shelters_near
# ══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def find_shelters_near(
    lat: Annotated[float, Field(description="Latitude of the search centre, WGS 84.")],
    lon: Annotated[float, Field(description="Longitude of the search centre, WGS 84.")],
    radius_m: Annotated[int, Field(description=(
        "Search radius in metres. 800 m corresponds roughly to a ten-minute walk in a "
        "straight line; widen it if the first search returns nothing."
    ))] = config.DEFAULT_RADIUS_M,
) -> dict[str, Any]:
    """Lists shelters located within a given straight-line radius of a point, ordered by
    distance, and returns their capacity, type and functional status. Distances are
    measured in a straight line, not along streets, so the closest shelter listed here
    is not necessarily the fastest to reach on foot: use `travel_time_to_shelter` when
    the user asks how long it takes to get there. Every shelter in the radius is
    returned, including non-functional ones, so ALWAYS state the status of each
    shelter you present: a non-functional shelter holds nobody and must never be
    offered as somewhere to go, and a partially functional one offers fewer places
    than its nominal capacity suggests. When the nearest shelters are non-functional,
    say so and point to the nearest usable one instead. If no shelter is found, say so
    plainly and offer to search a wider radius rather than reporting the nearest
    shelter outside it.
    """
    return await _boundary(
        "find_shelters_near", {"lat": lat, "lon": lon, "radius_m": radius_m},
        _find_shelters_near, lat, lon, radius_m,
    )


async def _find_shelters_near(lat: float, lon: float, radius_m: int) -> dict[str, Any]:
    lat, lon = check_coordinates(lat, lon)
    radius, note = clamp_radius(radius_m)
    cap = config.MAX_SHELTERS_RETURNED
    # One row over the cap, so truncation can be detected and declared. A
    # silently truncated list reads as a complete one, and a model would
    # report "there are 50 shelters" where the truth is "at least 50".
    # Every shelter, whatever its status. A user is entitled to know that the
    # nearest shelter to them cannot be used, and hiding non-functional ones
    # here would make that invisible. The status travels with every row, and
    # the description obliges the model to report it.
    rows = await repository.shelters_near(lat, lon, radius, usable_only=False, limit=cap + 1)
    truncated = len(rows) > cap
    rows = rows[:cap]
    notes = [n for n in (note,) if n]
    if truncated:
        notes.append(
            f"Only the {cap} nearest shelters are listed; more shelters lie "
            f"within {radius} m. Say so rather than presenting the list as complete."
        )
    payload: dict[str, Any] = {
        "count": len(rows),
        "radius_m": radius,
        "shelters": [_shelter_row(r, with_distance=True) for r in rows],
    }
    if notes:
        payload["note"] = " ".join(notes)
    return ok(await repository.snapshot_id(), **payload)


# ══════════════════════════════════════════════════════════════════════════
# 3. travel_time_to_shelter
# ══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def travel_time_to_shelter(
    lat: Annotated[float, Field(description="Latitude of the starting point, WGS 84.")],
    lon: Annotated[float, Field(description="Longitude of the starting point, WGS 84.")],
    shelter_id: Annotated[str, Field(description=(
        "Identifier of the target shelter, as returned by `find_shelters_near`."
    ))],
    speed_profile: SpeedProfile = config.DEFAULT_SPEED_PROFILE,
) -> dict[str, Any]:
    """Computes the walking time and route distance from a point to one specific shelter,
    following the street network. Use it after `find_shelters_near` to establish which
    nearby shelter is genuinely fastest to reach: street-network time can differ
    substantially from straight-line distance where rivers, railways or major roads
    force a detour. Call it once per candidate shelter when comparing a small number of
    options; do not call it for every shelter in a wide radius. Choose the speed profile
    that matches the person described by the user.
    """
    return await _boundary(
        "travel_time_to_shelter",
        {"lat": lat, "lon": lon, "shelter_id": shelter_id, "speed_profile": speed_profile},
        _travel_time_to_shelter, lat, lon, shelter_id, speed_profile,
    )


async def _travel_time_to_shelter(lat: float, lon: float, shelter_id: str,
                                  speed_profile: str) -> dict[str, Any]:
    lat, lon = check_coordinates(lat, lon)
    profile = check_speed_profile(speed_profile)
    shelter = await repository.shelter_by_id(shelter_id)
    if shelter is None:
        raise ToolFailure(
            SHELTER_NOT_FOUND,
            f"No shelter with the identifier '{shelter_id}' exists. Use the identifier "
            "exactly as returned by a search tool.",
        )

    straight_line = await repository.straight_line_m(lat, lon, shelter["shelter_id"])
    route = await routing.provider.walk_route(lat, lon, shelter["lat"], shelter["lon"])

    if route.get("unreachable"):
        return ok(
            await repository.snapshot_id(),
            shelter_id=shelter["shelter_id"],
            walk_time_min=None,
            distance_m=None,
            speed_profile=profile,
            straight_line_m=straight_line,
            note=("This shelter could not be reached on the pedestrian network from the "
                  "given point. Report that the walking time is unknown; the "
                  "straight-line distance is not a walking time."),
        )

    distance_m = route["distance_m"]
    return ok(
        await repository.snapshot_id(),
        shelter_id=shelter["shelter_id"],
        walk_time_min=routing.minutes_for(distance_m, profile),
        distance_m=distance_m,
        speed_profile=profile,
        straight_line_m=straight_line,
    )


# ══════════════════════════════════════════════════════════════════════════
# 4. shelter_details
# ══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def shelter_details(
    shelter_id: Annotated[str, Field(description=(
        "Identifier of the shelter, as returned by any search tool."
    ))],
) -> dict[str, Any]:
    """Returns everything known about a single shelter: address, nominal and effective
    capacity, type, functional status and geometry. Unlike the search tools, this one
    also returns shelters that are non-functional, because users are entitled to know
    that a shelter near them exists but cannot currently be used. When effective
    capacity is lower than nominal capacity, report both and explain that the
    difference reflects the shelter's condition.
    """
    return await _boundary(
        "shelter_details", {"shelter_id": shelter_id}, _shelter_details, shelter_id,
    )


async def _shelter_details(shelter_id: str) -> dict[str, Any]:
    shelter = await repository.shelter_by_id(shelter_id)
    if shelter is None:
        raise ToolFailure(
            SHELTER_NOT_FOUND,
            f"No shelter with the identifier '{shelter_id}' exists. Use the identifier "
            "exactly as returned by a search tool.",
        )
    payload = _shelter_row(shelter)
    payload["geometry"] = shelter["geometry"]
    return ok(await repository.snapshot_id(), **payload)


# ══════════════════════════════════════════════════════════════════════════
# 5. compute_isochrone
# ══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def compute_isochrone(
    lat: Annotated[float, Field(description="Latitude of the origin, WGS 84.")],
    lon: Annotated[float, Field(description="Longitude of the origin, WGS 84.")],
    minutes: Minutes = config.DEFAULT_MINUTES,
    speed_profile: SpeedProfile = config.DEFAULT_SPEED_PROFILE,
) -> dict[str, Any]:
    """Computes the area reachable on foot from a point within a given number of minutes,
    following the street network, and returns it as a polygon. This polygon is the
    input to `shelters_in_area` and `population_in_area`: use it whenever the user asks
    what lies within a walking time rather than within a distance. The result depends
    strongly on the speed profile, so choose it to match the person described. Do not
    approximate an isochrone with a circular radius: barriers such as rivers and
    railways make the reachable area markedly different from a circle.
    """
    return await _boundary(
        "compute_isochrone",
        {"lat": lat, "lon": lon, "minutes": minutes, "speed_profile": speed_profile},
        _compute_isochrone, lat, lon, minutes, speed_profile,
    )


async def _compute_isochrone(lat: float, lon: float, minutes: float,
                             speed_profile: str) -> dict[str, Any]:
    lat, lon = check_coordinates(lat, lon)
    profile = check_speed_profile(speed_profile)
    budget = check_minutes(minutes)
    result = await routing.provider.isochrone(lat, lon, budget, profile)
    return ok(
        await repository.snapshot_id(),
        polygon=result["polygon"],
        minutes=budget,
        speed_profile=profile,
        area_km2=result["area_km2"],
        origin={"lat": lat, "lon": lon},
    )


# ══════════════════════════════════════════════════════════════════════════
# 6. shelters_in_area
# ══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def shelters_in_area(
    polygon: Annotated[dict, Field(description=POLYGON_DESC)],
) -> dict[str, Any]:
    """Lists the shelters contained in a polygon and sums their capacity. Pass the polygon
    returned by `compute_isochrone` to answer questions about what is reachable within
    a walking time. The response reports both nominal and effective capacity: nominal is
    what the records state, effective accounts for the condition of each shelter. Report
    both when they differ, and do not present the total as the number of people who can
    shelter unless you also know how many people live in that area - for that,
    combine this tool with `population_in_area`.
    """
    return await _boundary(
        "shelters_in_area", {"polygon": "<polygon>"}, _shelters_in_area, polygon,
    )


# A polygon within the area limit can still contain more shelters than are
# useful to list. The cap is generous; what matters is that reaching it is
# declared, because the capacity total would otherwise be an undercount
# presented as a total.
AREA_SHELTER_CAP = config.MAX_SHELTERS_RETURNED * 20


async def _shelters_in_area(polygon: Any) -> dict[str, Any]:
    geojson, _area = await _validated_area(polygon)
    rows = await repository.shelters_in_polygon(
        geojson, usable_only=True, limit=AREA_SHELTER_CAP + 1
    )
    truncated = len(rows) > AREA_SHELTER_CAP
    rows = rows[:AREA_SHELTER_CAP]
    payload: dict[str, Any] = {
        "count": len(rows),
        **_capacity_totals(rows),
        "shelters": [_shelter_row(r) for r in rows],
    }
    if truncated:
        payload["note"] = (
            f"The area contains more than {AREA_SHELTER_CAP} usable shelters; the list "
            "and the capacity totals cover only those listed. Report the totals as a "
            "lower bound, or ask about a smaller area."
        )
    return ok(await repository.snapshot_id(), **payload)


# ══════════════════════════════════════════════════════════════════════════
# 7. population_in_area
# ══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def population_in_area(
    polygon: Annotated[dict, Field(description=(
        "Area to measure, as a GeoJSON Polygon - typically the `polygon` field returned "
        "by compute_isochrone."
    ))],
) -> dict[str, Any]:
    """Returns the estimated resident population inside a polygon, aggregated from
    gridded population data. Pass the polygon returned by `compute_isochrone` to find
    how many people live within a walking time of a point. Combined with
    `shelters_in_area` over the same polygon, this answers whether shelter capacity is
    sufficient for the people living there - two different questions that must not be
    confused: having a shelter within reach is not the same as having a place in it.
    The figure is a residential estimate, so state it as approximate and note that it
    reflects where people live rather than where they are during the day.
    """
    return await _boundary(
        "population_in_area", {"polygon": "<polygon>"}, _population_in_area, polygon,
    )


async def _population_in_area(polygon: Any) -> dict[str, Any]:
    geojson, area_km2 = await _validated_area(polygon)
    covered, note = await _population_note(geojson)
    row = await repository.population_in_polygon(geojson) or {}
    payload: dict[str, Any] = {
        # No coverage, no number. A polygon outside the layer has an unknown
        # population, which is not the same fact as an empty one.
        "population": (None if covered <= 0.0
                       else int(round(float(row.get("population") or 0.0)))),
        "cells_counted": int(row.get("cells_counted") or 0),
        "area_km2": area_km2,
        "method": config.POPULATION_METHOD,
        "coverage_fraction": round(covered, 3),
    }
    if note:
        payload["note"] = note
    return ok(await repository.snapshot_id(), **payload)


# ══════════════════════════════════════════════════════════════════════════
# 8. shelter_service_area
# ══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def shelter_service_area(
    shelter_id: Annotated[str, Field(description=(
        "Identifier of the shelter to analyse, as returned by any search tool."
    ))],
    minutes: Annotated[float, Field(description=(
        f"Walking time defining the served area, {config.MIN_MINUTES} to "
        f"{config.MAX_MINUTES} minutes."
    ))] = config.DEFAULT_MINUTES,
    speed_profile: SpeedProfile = config.DEFAULT_SPEED_PROFILE,
) -> dict[str, Any]:
    """Reverses the usual question: instead of asking which shelters a person can reach,
    it returns the area from which a given shelter can be reached within a walking time,
    the population living in that area, and the ratio between that population and the
    shelter's effective capacity. A high ratio means many people depend on a shelter
    that cannot hold them all. Use it when the user asks how many people a shelter
    serves, or whether it is sufficient. If the shelter is not functional, the ratio is
    not defined and is returned as null - report this honestly and explain why rather
    than substituting a number.
    """
    return await _boundary(
        "shelter_service_area",
        {"shelter_id": shelter_id, "minutes": minutes, "speed_profile": speed_profile},
        _shelter_service_area, shelter_id, minutes, speed_profile,
    )


async def _shelter_service_area(shelter_id: str, minutes: float,
                                speed_profile: str) -> dict[str, Any]:
    profile = check_speed_profile(speed_profile)
    budget = check_minutes(minutes)
    shelter = await repository.shelter_by_id(shelter_id)
    if shelter is None:
        raise ToolFailure(
            SHELTER_NOT_FOUND,
            f"No shelter with the identifier '{shelter_id}' exists. Use the identifier "
            "exactly as returned by a search tool.",
        )
    sid = shelter["shelter_id"]

    # The served area and its population are materialized: the same shelter is
    # asked about repeatedly across runs, and recomputing would make the
    # answer depend on when it was asked.
    cached = await repository.service_area_get(sid, budget, profile)
    if cached is None:
        if os.getenv("ENVIROO_STRICT_PRECOMPUTE", "").lower() in ("1", "true", "yes"):
            raise ToolFailure(
                NOT_PRECOMPUTED,
                f"The served area for {budget:g} minutes at the '{profile}' walking "
                "speed has not been precomputed for this shelter. Ask the user to "
                "choose one of the precomputed combinations.",
            )
        iso = await routing.provider.isochrone(
            shelter["lat"], shelter["lon"], budget, profile, inbound=True
        )
        polygon = iso["polygon"]
        geojson = json.dumps(polygon, ensure_ascii=False)
        pop_row = await repository.population_in_polygon(geojson) or {}
        population = float(pop_row.get("population") or 0.0)
        area_km2 = iso.get("area_km2") or float(pop_row.get("area_km2") or 0.0)
        await repository.service_area_put(sid, budget, profile, polygon, population, area_km2)
        cached = {
            "polygon": polygon, "population_served": population, "area_km2": area_km2,
        }

    # Measured from the polygon being reported, whichever path produced it.
    covered, coverage_note = await _population_note(
        json.dumps(cached["polygon"], ensure_ascii=False))

    population = (None if covered <= 0.0
                  else int(round(float(cached["population_served"] or 0.0))))
    effective = shelter.get("capacity_effective")
    payload: dict[str, Any] = {
        "shelter_id": sid,
        "service_area": cached["polygon"],
        "population_served": population,
        "coverage_fraction": round(covered, 3),
        "capacity_nominal": shelter.get("capacity_nominal"),
        "capacity_effective": effective,
        "status": shelter.get("status"),
        "minutes": budget,
        "speed_profile": profile,
        "area_km2": round(float(cached.get("area_km2") or 0.0), 3),
    }

    if effective and population is not None:
        payload["demand_ratio"] = round(population / effective, 1)
    elif population is None:
        # Capacity is known, demand is not. Reporting a ratio would require
        # inventing the half that is missing.
        payload["demand_ratio"] = None
        payload["note"] = coverage_note
    else:
        payload["demand_ratio"] = None
        if shelter.get("status") == config.STATUS_NON_FUNCTIONAL:
            payload["note"] = ("This shelter is non-functional; its effective capacity is "
                               "zero, so a demand ratio is not defined.")
        elif shelter.get("capacity_nominal") is None:
            payload["note"] = ("The capacity of this shelter is not recorded in the source "
                               "data, so a demand ratio cannot be computed.")
        else:
            payload["note"] = ("The functional status of this shelter is not recorded, so "
                               "its effective capacity is treated as zero and a demand "
                               "ratio is not defined.")

    if coverage_note and payload.get("note") != coverage_note:
        # Two independent caveats can apply at once; neither replaces the other.
        payload["note"] = " ".join(filter(None, (payload.get("note"), coverage_note)))
    return ok(await repository.snapshot_id(), **payload)


# ══════════════════════════════════════════════════════════════════════════
# 9. reverse_geocode
# ══════════════════════════════════════════════════════════════════════════
#
# Appended rather than placed next to `geocode_address` on purpose: the eight
# specified tools keep the positions they had in `tools/list`, so runs made
# before and after this addition differ by one appended tool and not by a
# reordered prompt.

@mcp.tool()
async def reverse_geocode(
    lat: Annotated[float, Field(description="Latitude of the point to describe, WGS 84.")],
    lon: Annotated[float, Field(description="Longitude of the point to describe, WGS 84.")],
) -> dict[str, Any]:
    """Converts coordinates into the nearest street address or place name - the inverse
    of `geocode_address`. Use it when the user has given a point rather than a name, for
    instance a position picked on a map, and the answer reads better with a place in it.
    It is not a step towards the other tools: they all take coordinates directly, so
    turning a point into a name and resolving that name again wastes a call and loses
    precision. The result states how far the matched place lies from the point asked
    about; when that distance is large the name describes the surrounding area rather
    than the point itself, and reporting it without that caveat would overstate what is
    known.
    """
    return await _boundary("reverse_geocode", {"lat": lat, "lon": lon},
                           _reverse_geocode, lat, lon)


async def _reverse_geocode(lat: float, lon: float) -> dict[str, Any]:
    lat, lon = check_coordinates(lat, lon)
    match = await geocoding.provider.reverse(lat, lon)
    return ok(await repository.snapshot_id(), **match)


dedent_tool_descriptions(mcp)

# The second arm of the description experiment. Applied after registration so
# the rich text above stays the single source for the default regime: the
# minimal set is a replacement, never a parallel definition that could drift.
DESCRIPTION_SETS = {"minimal": MINIMAL_DESCRIPTIONS, "bare": BARE_DESCRIPTIONS}

if config.TOOL_DESCRIPTIONS in DESCRIPTION_SETS:
    n = apply_descriptions(mcp, DESCRIPTION_SETS[config.TOOL_DESCRIPTIONS])
    log.warning("PUBLISHING THE %s DESCRIPTION SET (%d tools). Runs made now are NOT "
                "comparable with runs made under another set.",
                config.TOOL_DESCRIPTIONS.upper(), n)
elif config.TOOL_DESCRIPTIONS != "rich":
    raise RuntimeError(
        f"ENVIROO_TOOL_DESCRIPTIONS={config.TOOL_DESCRIPTIONS!r} is not a description "
        f"set; use 'rich', or one of {sorted(DESCRIPTION_SETS)}.")


def description_fingerprint() -> dict:
    """What the models are actually being shown, as a checkable summary.

    The regime name comes from configuration and can drift from reality if the
    servers are started with different environments; the character count comes
    from the published surface and cannot. Recorded with every benchmark row so
    a mismatch is visible in the data rather than assumed away.
    """
    tools = mcp._tool_manager.list_tools()
    chars = sum(len(t.description or "") for t in tools)
    chars += sum(len(p.get("description", ""))
                 for t in tools
                 for p in (t.parameters or {}).get("properties", {}).values())
    # Reported so that a client can check, rather than assume, that the runtime
    # guidance layer was held constant while the descriptions changed.
    from app.mcp.envelope import _guided
    return {"regime": config.TOOL_DESCRIPTIONS, "tools": len(tools), "chars": chars,
            "guidance_runtime": _guided()}


# ── HTTP shell: /health plus the Streamable HTTP transport at /mcp ────────

@asynccontextmanager
async def _lifespan(_: FastAPI):
    # A mounted sub-application does not run its own lifespan; FastMCP needs
    # this context to initialize its Streamable HTTP sessions.
    async with mcp.session_manager.run():
        yield
    await repository.close()


app = FastAPI(title="ENVIROO Shelter MCP Server", lifespan=_lifespan)
app.add_middleware(RequestIdMiddleware, logger_name="enviroo_mcp.http")


@app.get("/health", include_in_schema=False)
async def health():
    try:
        data = await repository.health()
        return {"status": "ok", "service": "enviroo-shelter-mcp",
                "port": config.MCP_PORT, "descriptions": description_fingerprint(),
                **data}
    except Exception as exc:
        return {"status": "error", "service": "enviroo-shelter-mcp",
                "port": config.MCP_PORT, "detail": str(exc)}


@app.get("/map/shelters", include_in_schema=False)
async def map_shelters(ids: str = ""):
    """Coordinates for the map in the web UI. Not part of the MCP surface.

    The tools return what the specification says they return; drawing a
    result on a map needs geometry the model has no use for. Serving it here,
    outside `tools/list`, keeps the two concerns from contaminating each
    other.
    """
    wanted = [i.strip() for i in ids.split(",") if i.strip()]
    rows = await repository.shelters_by_ids(wanted)
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
            "properties": _shelter_row(row),
        } for row in rows],
    }


app.mount("/", mcp.streamable_http_app())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.mcp.server:app",
        host=config.MCP_HOST,
        port=config.MCP_PORT,
        reload=False,
    )
