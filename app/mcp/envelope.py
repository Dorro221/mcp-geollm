"""
Response envelope and error vocabulary for the ENVIROO MCP surface.

Two shapes leave a tool, and only two:

    success  {"snapshot_id": ..., "units": {...}, <payload>}
    failure  {"error_code": ..., "message": ...}

`ToolFailure` is raised deep inside the implementation and converted to the
failure shape at the tool boundary. Nothing else is caught silently: an
unexpected exception becomes an `internal_error` with its own message rather
than an invented value, because a model that receives a plausible number has
no way of knowing the computation never happened.
"""
from __future__ import annotations

from typing import Any

from app import config

UNITS = {"distance": "m", "time": "min"}

# ── Error codes (specification §1.6) ──────────────────────────────────────
GEOCODING_FAILED = "geocoding_failed"
GEOCODING_AMBIGUOUS = "geocoding_ambiguous"
SHELTER_NOT_FOUND = "shelter_not_found"
INVALID_COORDINATES = "invalid_coordinates"
INVALID_GEOMETRY = "invalid_geometry"
AREA_TOO_LARGE = "area_too_large"
ROUTING_UNAVAILABLE = "routing_unavailable"
NOT_PRECOMPUTED = "not_precomputed"
INVALID_PROFILE = "invalid_profile"
INTERNAL_ERROR = "internal_error"


class ToolFailure(Exception):
    """A failure the model is expected to read, relay or act upon.

    `extra` carries structured context that belongs in the failure object
    itself — the `candidates` list of an ambiguous geocode, for instance —
    so the model can act without a second call.
    """

    def __init__(self, error_code: str, message: str, **extra: Any):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.extra = extra

    def as_response(self) -> dict[str, Any]:
        return failure(self.error_code, self.message, **self.extra)


def _guided() -> bool:
    """True while tool responses carry their guidance text.

    An error message that says what to do next, and a note that explains why a
    value is null, are guidance - but the service produces them at runtime and
    does not publish them as documentation. They are therefore governed by
    `GUIDANCE_RUNTIME`, not by the description regime, so that the arms of the
    experiment differ in their published descriptions alone.

    `auto` restores the original coupling, under which the first campaign ran.
    Asked there as "is it rich" rather than "is it not minimal", so that adding
    an arm cannot accidentally grant it guidance by default.
    """
    setting = config.GUIDANCE_RUNTIME
    if setting == "on":
        return True
    if setting == "off":
        return False
    return config.TOOL_DESCRIPTIONS == "rich"


def failure(error_code: str, message: str, **extra: Any) -> dict[str, Any]:
    """The failure shape. Under `minimal`, a status code and nothing else.

    `extra` survives because it is data, not advice - the `candidates` list of
    an ambiguous geocode is a fact about the result. The message is advice,
    written for the model to act on, and that is the point of removing it.
    """
    if not _guided():
        return {"error_code": error_code, **extra}
    return {"error_code": error_code, "message": message, **extra}


def ok(snapshot_id: str, **payload: Any) -> dict[str, Any]:
    """Wrap a successful payload in the standard envelope.

    Under `minimal` the `note` field is dropped. The values it explains are
    untouched: `demand_ratio` is still null, the list is still truncated, the
    walking time is still unknown. What the model no longer receives is the
    sentence telling it that a null is not a zero.
    """
    if not _guided():
        payload.pop("note", None)
    return {"snapshot_id": snapshot_id, "units": dict(UNITS), **payload}


# ── Shared validators ─────────────────────────────────────────────────────

def check_coordinates(lat: Any, lon: Any) -> tuple[float, float]:
    """Coordinates must be numbers inside the study area.

    A point outside is refused instead of returning an empty result: "no
    shelter within 800 m of Vienna" is a true statement about this dataset
    and a misleading answer to give a user.
    """
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        raise ToolFailure(
            INVALID_COORDINATES,
            "Latitude and longitude must be numbers in decimal degrees (WGS 84), "
            f"for example lat=46.7712, lon=23.5900. Received lat={lat!r}, lon={lon!r}.",
        )
    if not config.in_study_area(lat_f, lon_f):
        area = config.STUDY_AREA
        raise ToolFailure(
            INVALID_COORDINATES,
            f"The point {lat_f}, {lon_f} lies outside the study area "
            f"({config.STUDY_AREA_NAME}: latitude {area['lat_min']}–{area['lat_max']}, "
            f"longitude {area['lon_min']}–{area['lon_max']}). Check that latitude and "
            "longitude have not been swapped.",
        )
    return lat_f, lon_f


def check_speed_profile(value: Any) -> str:
    if value is None:
        return config.DEFAULT_SPEED_PROFILE
    profile = str(value).strip().lower()
    if profile not in config.SPEED_PROFILES:
        raise ToolFailure(
            INVALID_PROFILE,
            f"Unknown speed profile {value!r}. Supported values: "
            f"{', '.join(config.SPEED_PROFILES)}.",
        )
    return profile


def check_minutes(value: Any) -> float:
    """Walking-time budget, in the supported range.

    Out-of-range is `not_precomputed` with the supported values stated, per
    specification: the tool must never quietly answer a different question
    from the one asked.
    """
    if value is None:
        return float(config.DEFAULT_MINUTES)
    try:
        minutes = float(value)
    except (TypeError, ValueError):
        raise ToolFailure(
            NOT_PRECOMPUTED,
            f"`minutes` must be a number between {config.MIN_MINUTES} and "
            f"{config.MAX_MINUTES}. Received {value!r}.",
        )
    if not (config.MIN_MINUTES <= minutes <= config.MAX_MINUTES):
        raise ToolFailure(
            NOT_PRECOMPUTED,
            f"A walking time of {minutes:g} minutes is not available. Supported "
            f"values run from {config.MIN_MINUTES} to {config.MAX_MINUTES} minutes. "
            "Ask the user which of these to use instead of substituting one.",
        )
    return minutes


def clamp_radius(value: Any) -> tuple[int, str | None]:
    """Search radius in metres, clamped rather than refused.

    Returns the radius and, when clamping happened, a note to place in the
    response so the model reports the radius actually used.
    """
    if value is None:
        return config.DEFAULT_RADIUS_M, None
    try:
        radius = int(float(value))
    except (TypeError, ValueError):
        raise ToolFailure(
            INVALID_COORDINATES,
            f"`radius_m` must be a number of metres. Received {value!r}.",
        )
    if radius <= 0:
        raise ToolFailure(
            INVALID_COORDINATES,
            f"`radius_m` must be greater than zero. Received {radius}.",
        )
    if radius > config.MAX_RADIUS_M:
        return config.MAX_RADIUS_M, (
            f"The requested radius of {radius} m exceeds the maximum of "
            f"{config.MAX_RADIUS_M} m; the search used {config.MAX_RADIUS_M} m. "
            "Report the radius actually used."
        )
    return radius, None
