"""
ENVIROO — central configuration.

Everything that defines the *experimental apparatus* lives here and nowhere
else: the study area, the walking speeds behind `speed_profile`, the limits
that guard the tools, and the identity of the data snapshot. A study is only
reproducible if these constants are stated in one place and reported with
every answer, which is why `snapshot_id` travels in every successful
response envelope.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv(override=True)

# ── Database ──────────────────────────────────────────────────────────────
# ENVIROO reads its own schema (`enviroo`), built by the ETL from the
# read-only source inventory. It never writes to the source tables.
def _env(name: str, default: str) -> str:
    """Environment value, treating an empty variable as "not set".

    A key left blank in .env is a common way of saying "I have not filled
    this in yet"; taking it literally would produce failures far from their
    cause (an empty sslmode reaches the driver as an invalid value).
    """
    return (os.getenv(name) or "").strip() or default


PG_HOST = _env("PGHOST", "localhost")
PG_PORT = int(_env("PGPORT", "5432"))
PG_DATABASE = _env("PGDATABASE", "postgis")
PG_USER = _env("PGUSER", "postgres")
PG_PASSWORD = os.getenv("PGPASSWORD", "postgres")
PG_SSLMODE = _env("PGSSLMODE", "prefer")
PG_POOL_MIN = int(_env("PGPOOL_MIN_SIZE", "1"))
PG_POOL_MAX = int(_env("PGPOOL_MAX_SIZE", "10"))

SCHEMA = _env("ENVIROO_SCHEMA", "enviroo")
SOURCE_TABLE = _env("ENVIROO_SOURCE_TABLE", "public.adaposturi")


def dsn() -> str:
    return (
        f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
        f"?sslmode={PG_SSLMODE}"
    )


# ── Coordinate systems ────────────────────────────────────────────────────
# The tool boundary speaks EPSG:4326 only. Every metric computation happens
# in EPSG:3844 (Stereo70), the Romanian national projection: distances in
# metres there are true distances, which they are not in degrees.
SRID_WGS84 = 4326
SRID_METRIC = 3844


# ── Study area ────────────────────────────────────────────────────────────
# Bounding box of Romania, slightly enlarged. Coordinates outside it are
# rejected with `invalid_coordinates` rather than silently producing an
# empty result that a model would report as "no shelters nearby".
#
# The shelter inventory is national and stays that way. The POPULATION layer
# covers only Cluj-Napoca (see POPULATION_COVERAGE below), so the two tools
# that read it declare where they have no data instead of returning zero.
STUDY_AREA = {"lat_min": 43.4, "lat_max": 48.4, "lon_min": 20.1, "lon_max": 30.1}
STUDY_AREA_NAME = "Romania"


def in_study_area(lat: float, lon: float) -> bool:
    return (
        STUDY_AREA["lat_min"] <= lat <= STUDY_AREA["lat_max"]
        and STUDY_AREA["lon_min"] <= lon <= STUDY_AREA["lon_max"]
    )


# ── Speed profiles ────────────────────────────────────────────────────────
# One walking speed per profile, in km/h. The routing provider is queried on
# its own pedestrian profile; the *time* is then derived from the returned
# route length and the speed below, so that the three profiles differ by
# exactly the factor stated in the specification and by nothing else.
SPEED_PROFILES: dict[str, float] = {
    "reduced_mobility": 3.0,
    "adult": 5.0,
    "hurried": 7.0,
}
DEFAULT_SPEED_PROFILE = "adult"
# The routing provider's own pedestrian speed, used to rescale time budgets
# when asking it for an isochrone (see providers/routing.py).
PROVIDER_WALK_SPEED_KMH = 5.0


# ── Limits ────────────────────────────────────────────────────────────────
DEFAULT_RADIUS_M = 800
MAX_RADIUS_M = 5000            # clamped with a note, never an error
DEFAULT_MINUTES = 10
MIN_MINUTES = 5
MAX_MINUTES = 30
MAX_AREA_KM2 = float(_env("ENVIROO_MAX_AREA_KM2", "150"))
MAX_SHELTERS_RETURNED = 50

# Nominal capacity is reduced by this factor for a partially functional
# shelter. Recorded in the snapshot so a run can always be reinterpreted.
PARTIAL_CAPACITY_FACTOR = float(_env("ENVIROO_PARTIAL_CAPACITY_FACTOR", "0.5"))

# Rule applied to population cells that a polygon only partly covers. One
# rule for the whole system, exposed to the model as `method`.
POPULATION_METHOD = "area_weighted"

# ── Status vocabulary ─────────────────────────────────────────────────────
STATUS_FUNCTIONAL = "functional"
STATUS_PARTIAL = "partially_functional"
STATUS_NON_FUNCTIONAL = "non_functional"
STATUS_UNKNOWN = "unknown"
USABLE_STATUSES = (STATUS_FUNCTIONAL, STATUS_PARTIAL)


# ── Routing / geocoding provider ──────────────────────────────────────────
ORS_BASE = _env("ORS_BASE", "https://api.openrouteservice.org")
ORS_API_KEY = os.getenv("ORS_API_KEY", "")
ORS_PROFILE = "foot-walking"
ORS_TIMEOUT_S = float(_env("ENVIROO_ORS_TIMEOUT", "30"))

# How long to wait for a TCP/TLS connection to an LLM provider. The OpenAI
# SDK defaults this to 5 seconds while allowing 600 for the response, which
# is the wrong way round for a benchmark sweep: back-to-back requests
# regularly need longer than 5 s to establish a fresh connection, the SDK
# retries twice, and the run dies after ~16 s having never reached the model.
# A whole 23-task sweep was lost to it. Reading stays generous - a reasoning
# model can legitimately think for minutes.
LLM_CONNECT_TIMEOUT_S = float(_env("ENVIROO_LLM_CONNECT_TIMEOUT", "30"))
LLM_READ_TIMEOUT_S = float(_env("ENVIROO_LLM_READ_TIMEOUT", "600"))
GEOCODE_COUNTRY = _env("ENVIROO_GEOCODE_COUNTRY", "RO")

# Cached provider answers are keyed by coordinates rounded to this many
# decimals (~1 m). Without rounding, a caller that passes 46.77120001 would
# miss the cache and reintroduce provider latency into the measurements.
CACHE_COORD_DECIMALS = 5

# Whether geocoding, routing and isochrone answers are cached between calls.
#
# OFF means every prompt and every evaluation run hits the provider afresh, so
# a run reflects the provider as it is at that moment. It also means provider
# latency (200-700 ms, variable) lands inside the measured turn duration and
# therefore inside M16, where it is indistinguishable from a difference between
# models; and that repeated runs of the same task can return slightly different
# geometries. Both are reasons the cache existed, and both should be stated as
# limitations when reporting timings taken with it off.
#
# ON restores the original behaviour: the provider is asked once per distinct
# input, and every later run of the same task sees the identical answer.
PROVIDER_CACHE_ENABLED = _env("ENVIROO_PROVIDER_CACHE", "false").lower() in ("1", "true", "yes", "on")

# ── Tool descriptions ─────────────────────────────────────────────────────
# Which description set the server publishes. The descriptions ARE the
# apparatus - no model here is fine-tuned - so this is an independent
# variable, not a preference.
#
#   rich    (default) the full descriptions: when to prefer a neighbouring
#           tool, what to ask for when something is missing, the cues that map
#           colloquial language onto enum values, how to report honestly.
#   minimal what an API reference would carry and nothing more.
#   bare    the tool name restated as a sentence and the JSON type of each
#           parameter - what a framework generates when nobody documents.
#
# `rich` is the only regime that carries guidance. Under the other two the
# error messages lose their advice, the responses lose their explanatory
# notes, and the system prompt keeps only the role and the language rule.
#
# Every benchmark row records the regime it ran under, so the two arms cannot
# be mixed in one analysis by accident.
TOOL_DESCRIPTIONS = _env("ENVIROO_TOOL_DESCRIPTIONS", "rich").lower()


# ── Guidance layers held constant across the arms ─────────────────────────
# The description regime used to govern three things at once: the published
# tool descriptions, the guidance carried inside tool responses (an error's
# `message`, a result's `note`) and the system prompt. Only the first is the
# apparatus under test, so the other two are settings of their own and are
# held constant across the arms. With the defaults below, `rich` and `bare`
# differ in their published descriptions and in nothing else.
#
#   on   (default) tool responses keep their message and note in every regime
#   off            no regime receives them
#   auto           legacy: they follow TOOL_DESCRIPTIONS, as in the first campaign
GUIDANCE_RUNTIME = _env("ENVIROO_GUIDANCE_RUNTIME", "on").lower()

# ── Which route the models are reached by ─────────────────────────────────
# The study's claim is that one gateway serves every model, so that the client,
# the request format, the retry policy and the network path are shared and the
# model is the only thing that changes. That holds only if the commercial pair
# goes through the gateway too, which is what `gateway` does.
#
#   gateway (default) every model through OpenRouter, one key, one path
#   direct            each model that declares a `route_direct` block in
#                     models.json uses it instead - the GPT pair then goes
#                     straight to api.openai.com on OPENAI_API_KEY
#
# Every result row records the route in use, because a campaign that mixes the
# two is not one campaign.
MODEL_ROUTE = _env("ENVIROO_MODEL_ROUTE", "gateway").lower()


# ── Who is reading the answers ────────────────────────────────────────────
# The expert verdict is the study's primary outcome, and a single reader cannot
# report agreement with anyone. Naming the evaluator is what lets a second one
# judge the same runs without overwriting the first: observations are keyed by
# (run_id, evaluator), and `EVALUATOR_PRIMARY` decides which of the two verdict
# columns of metrics.csv a given reader writes to.
EVALUATOR = _env("ENVIROO_EVALUATOR", "E1").strip() or "E1"
EVALUATOR_PRIMARY = _env("ENVIROO_EVALUATOR_PRIMARY", "E1").strip() or "E1"


# ── Who is reading the answers ────────────────────────────────────────────
# The expert verdict is the study's primary outcome, and a single reader cannot
# report agreement with anyone. Naming the evaluator is what lets a second one
# judge the same runs without overwriting the first: observations are keyed by
# (run_id, evaluator), and `EVALUATOR_PRIMARY` decides which of the two verdict
# columns of metrics.csv a given reader writes to.
EVALUATOR = _env("ENVIROO_EVALUATOR", "E1").strip() or "E1"
EVALUATOR_PRIMARY = _env("ENVIROO_EVALUATOR_PRIMARY", "E1").strip() or "E1"

# Which system prompt the client sends.
#
#   shared (default) one prompt for every regime, so the prompt is not a
#                    second experimental factor
#   legacy           the guided prompt under `rich`, the thin one elsewhere -
#                    the pairing the first campaign ran under
SYSTEM_PROMPT_REGIME = _env("ENVIROO_SYSTEM_PROMPT", "shared").lower()


# ── Ports ─────────────────────────────────────────────────────────────────
MCP_HOST = _env("ENVIROO_MCP_HOST", "0.0.0.0")
MCP_PORT = int(_env("ENVIROO_MCP_PORT", "9100"))
# The minimal-description arm runs as its own process, on its own port, so
# both arms can be up at once and the registry entry swapped in seconds
# rather than a server restarted. See app/mcp/server_minimal.py.
MCP_PORT_MINIMAL = int(_env("ENVIROO_MCP_PORT_MINIMAL", "9101"))
# The bare arm: the tool name is the only documentation. See
# app/mcp/descriptions_bare.py for where the line is drawn.
MCP_PORT_BARE = int(_env("ENVIROO_MCP_PORT_BARE", "9102"))
WEB_PORT = int(_env("ENVIROO_WEB_PORT", "8090"))
