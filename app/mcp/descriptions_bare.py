"""The bare description set — no descriptions at all.

The lower bound of the experiment. Every tool and every parameter publishes an
empty description, so what reaches the model is the JSON Schema and nothing
else: the tool name, the parameter names, their types, their defaults, and the
`enum` on speed_profile.

The name is the documentation. `find_shelters_near(lat, lon, radius_m)` says
what it says by being called that, and a model has to decide from the name
alone whether radius_m is metres or kilometres, whether the distance is
straight-line or along streets, and which of the nine tools answers "how long
would it take to walk there".

The enum is NOT an exception to this. It used to live in the description text,
which meant emptying the description made the tool uncallable; it now lives in
the schema, as `enum`, in all three regimes. That is a constraint on what may
be sent, not an explanation of what to send, and holding it constant is what
lets the three arms differ in text alone.

Selected by ENVIROO_TOOL_DESCRIPTIONS=bare, or by running
app/mcp/server_bare.py on its own port. 0 characters, against `minimal`'s 1214
and `rich`'s 7547.
"""
from __future__ import annotations

# Every published tool, every published parameter, empty. The keys still have
# to be exhaustive: `apply_descriptions` refuses a set that forgets a
# parameter, because a forgotten one would silently keep the text of whichever
# regime was applied before it.
BARE: dict[str, dict] = {
    "geocode_address": {
        "description": "",
        "params": {"query": ""},
    },
    "find_shelters_near": {
        "description": "",
        "params": {"lat": "", "lon": "", "radius_m": ""},
    },
    "travel_time_to_shelter": {
        "description": "",
        "params": {"lat": "", "lon": "", "shelter_id": "", "speed_profile": ""},
    },
    "shelter_details": {
        "description": "",
        "params": {"shelter_id": ""},
    },
    "compute_isochrone": {
        "description": "",
        "params": {"lat": "", "lon": "", "minutes": "", "speed_profile": ""},
    },
    "shelters_in_area": {
        "description": "",
        "params": {"polygon": ""},
    },
    "population_in_area": {
        "description": "",
        "params": {"polygon": ""},
    },
    "shelter_service_area": {
        "description": "",
        "params": {"shelter_id": "", "minutes": "", "speed_profile": ""},
    },
    "reverse_geocode": {
        "description": "",
        "params": {"lat": "", "lon": ""},
    },
}
