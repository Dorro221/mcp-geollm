"""The minimal description set — the second arm of the description experiment.

The published descriptions are the experimental apparatus: they are what
induces the behaviour the study measures, since no model here is fine-tuned.
That makes "how much of the measured competence comes from the model, and how
much from the descriptions?" a question the study cannot answer with one
description set. This module supplies the other set.

The cut is principled rather than aggressive. MINIMAL means everything an API
reference would carry and nothing more:

  * what the tool does, in one sentence;
  * what each parameter is, its unit, and - where the value space is not in
    the JSON Schema - its permitted values.

REMOVED is everything written to shape behaviour:

  * when to prefer this tool over a neighbouring one;
  * what to do when the user has not supplied something (ask, never guess);
  * the natural-language cues that map onto enum values ("an elderly person"
    -> reduced_mobility);
  * how to report the result honestly - declare imprecision, state functional
    status, never present nominal capacity as available places.

`speed_profile` keeps its list of permitted values and `minutes` keeps its
range, because neither is expressed in the schema: without them the tool
cannot be called correctly at all, and the arm would be measuring breakage
rather than the absence of guidance. What it loses is any hint about WHICH
value a given sentence calls for, which is precisely the mapping under test.

Selected by ENVIROO_TOOL_DESCRIPTIONS=minimal. The regime is recorded with
every benchmark row, so the two arms can never be silently mixed.
"""
from __future__ import annotations

# One entry per published tool: the description the model sees, and a
# description per parameter. Keys must match the published names exactly;
# `apply_descriptions` fails loudly if they drift.
MINIMAL: dict[str, dict] = {
    "geocode_address": {
        "description": "Converts an address or place name into geographic coordinates.",
        "params": {
            "query": "Address or place name.",
        },
    },
    "find_shelters_near": {
        "description": "Lists shelters within a straight-line radius of a point, ordered by "
                       "distance.",
        "params": {
            "lat": "Latitude, WGS 84.",
            "lon": "Longitude, WGS 84.",
            "radius_m": "Search radius in metres.",
        },
    },
    "travel_time_to_shelter": {
        "description": "Computes the walking time and route distance from a point to one "
                       "shelter.",
        "params": {
            "lat": "Latitude of the starting point, WGS 84.",
            "lon": "Longitude of the starting point, WGS 84.",
            "shelter_id": "Identifier of the shelter.",
            "speed_profile": "Walking speed. One of: reduced_mobility, adult, hurried.",
        },
    },
    "shelter_details": {
        "description": "Returns the full record of one shelter.",
        "params": {
            "shelter_id": "Identifier of the shelter.",
        },
    },
    "compute_isochrone": {
        "description": "Computes the area reachable on foot within a time budget.",
        "params": {
            "lat": "Latitude of the origin, WGS 84.",
            "lon": "Longitude of the origin, WGS 84.",
            "minutes": "Time budget in minutes, from 5 to 30.",
            "speed_profile": "Walking speed. One of: reduced_mobility, adult, hurried.",
        },
    },
    "shelters_in_area": {
        "description": "Lists the shelters inside a polygon and sums their capacity.",
        "params": {
            "polygon": "Area as a GeoJSON Polygon.",
        },
    },
    "population_in_area": {
        "description": "Returns the resident population inside a polygon.",
        "params": {
            "polygon": "Area as a GeoJSON Polygon.",
        },
    },
    "shelter_service_area": {
        "description": "Returns the area served by a shelter and the population within it.",
        "params": {
            "shelter_id": "Identifier of the shelter.",
            "minutes": "Walking time defining the served area, from 5 to 30.",
            "speed_profile": "Walking speed. One of: reduced_mobility, adult, hurried.",
        },
    },
    "reverse_geocode": {
        "description": "Converts coordinates into the nearest address or place name.",
        "params": {
            "lat": "Latitude of the point, WGS 84.",
            "lon": "Longitude of the point, WGS 84.",
        },
    },
}
