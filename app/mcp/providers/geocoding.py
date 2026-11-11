"""
GeocodingProvider — free text in, coordinates out.

The tools never talk to a geocoding service directly; they talk to this
adapter. Swapping the provider (Pelias behind OpenRouteService today,
Nominatim or a national gazetteer tomorrow) must not change a single line of
the MCP surface, because the surface is part of the experiment and a changed
description would silently change what is being measured.

Answers are cached by normalized query string. The same addresses recur in
every evaluation run, and caching removes provider latency variance from the
measurements.
"""
from __future__ import annotations

import math
import re
from typing import Any

import httpx

from app import config
from app.logging_utils import log_timing, setup_logging
from app.mcp import repository
from app.mcp.envelope import GEOCODING_AMBIGUOUS, GEOCODING_FAILED, ToolFailure

log = setup_logging("enviroo_mcp")

CACHE_KIND = "geocode"
REVERSE_CACHE_KIND = "reverse_geocode"

# Pelias layers, mapped onto the three levels the specification exposes.
# Everything coarser than a street is an `area`: the distinction that matters
# to a user is "this is the building" vs "this is the street" vs "this is
# somewhere in that part of town".
_HOUSE_LAYERS = {"address", "venue", "building"}
_STREET_LAYERS = {"street"}

# Two results are treated as genuinely distinct places when they are more
# than this far apart. Below it, they are the same place described twice
# (a venue and its address), and returning "ambiguous" would be noise.
_DISTINCT_PLACE_M = 1500
# ...and only if the second result is nearly as good as the first.
_CONFIDENCE_GAP = 0.08

# Beyond this, the nearest addressable place is a description of the
# neighbourhood rather than of the point asked about. The distance is
# returned either way; past this threshold the answer also says so, because
# a label presented without it reads as the address of the point itself.
_FAR_MATCH_M = 500


def _normalize(query: str) -> str:
    return re.sub(r"\s+", " ", str(query or "").strip().lower())


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _match_level(layer: str) -> str:
    layer = (layer or "").lower()
    if layer in _HOUSE_LAYERS:
        return "house"
    if layer in _STREET_LAYERS:
        return "street"
    return "area"


class GeocodingProvider:
    """Adapter over the configured geocoding service."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or config.ORS_BASE).rstrip("/")
        self.api_key = api_key if api_key is not None else config.ORS_API_KEY

    async def _search(self, query: str) -> list[dict[str, Any]]:
        if not self.api_key:
            raise ToolFailure(
                GEOCODING_FAILED,
                "The geocoding service is not configured, so addresses cannot be "
                "resolved. Ask the user to give coordinates instead.",
            )
        params = {
            "api_key": self.api_key,
            "text": query,
            "boundary.country": config.GEOCODE_COUNTRY,
            "size": 5,
        }
        try:
            with log_timing(log, "geocode.provider", query=query):
                async with httpx.AsyncClient(timeout=config.ORS_TIMEOUT_S) as client:
                    response = await client.get(f"{self.base_url}/geocode/search", params=params)
                    response.raise_for_status()
                    data = response.json()
        except Exception as exc:
            raise ToolFailure(
                GEOCODING_FAILED,
                f"The geocoding service could not be reached, so '{query}' was not "
                "resolved. Say so and offer to try again or to accept coordinates.",
            ) from exc

        results = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            coords = feature.get("geometry", {}).get("coordinates", [None, None])
            if coords[0] is None:
                continue
            results.append({
                "lat": round(float(coords[1]), 6),
                "lon": round(float(coords[0]), 6),
                "matched_name": props.get("label", ""),
                "match_level": _match_level(props.get("layer", "")),
                "confidence": round(float(props.get("confidence") or 0.0), 2),
            })
        return results

    async def geocode(self, query: str) -> dict[str, Any]:
        """Best match for `query`, or a failure the model can act on."""
        text = str(query or "").strip()
        if not text:
            raise ToolFailure(
                GEOCODING_FAILED,
                "No location was given. Ask the user for an address, a landmark or a "
                "place name before computing anything.",
            )

        key = _normalize(text)
        cached = await repository.cache_get(CACHE_KIND, key)
        if cached is not None:
            if "error_code" in cached:
                raise ToolFailure(cached["error_code"], cached["message"],
                                  **{k: v for k, v in cached.items()
                                     if k not in ("error_code", "message")})
            return cached

        results = await self._search(text)
        if not results:
            failure = {
                "error_code": GEOCODING_FAILED,
                "message": (
                    f"No location matched '{text}'. Ask the user to rephrase, to add "
                    "the town, or to give a street and number."
                ),
            }
            await repository.cache_put(CACHE_KIND, key, failure)
            raise ToolFailure(failure["error_code"], failure["message"])

        best = results[0]
        rivals = [
            r for r in results[1:]
            if best["confidence"] - r["confidence"] <= _CONFIDENCE_GAP
            and _haversine_m(best["lat"], best["lon"], r["lat"], r["lon"]) > _DISTINCT_PLACE_M
        ]
        if rivals:
            failure = {
                "error_code": GEOCODING_AMBIGUOUS,
                "message": (
                    f"'{text}' matches several distinct places equally well. Ask the "
                    "user which one they mean before computing anything."
                ),
                "candidates": [best] + rivals,
            }
            await repository.cache_put(CACHE_KIND, key, failure)
            raise ToolFailure(failure["error_code"], failure["message"],
                              candidates=failure["candidates"])

        await repository.cache_put(CACHE_KIND, key, best)
        return best

    async def _reverse(self, lat: float, lon: float) -> list[dict[str, Any]]:
        if not self.api_key:
            raise ToolFailure(
                GEOCODING_FAILED,
                "The geocoding service is not configured, so coordinates cannot be "
                "described. Report the coordinates themselves instead.",
            )
        params = {
            "api_key": self.api_key,
            "point.lat": lat,
            "point.lon": lon,
            "boundary.country": config.GEOCODE_COUNTRY,
            "size": 1,
        }
        try:
            with log_timing(log, "reverse_geocode.provider", lat=lat, lon=lon):
                async with httpx.AsyncClient(timeout=config.ORS_TIMEOUT_S) as client:
                    response = await client.get(f"{self.base_url}/geocode/reverse",
                                                params=params)
                    response.raise_for_status()
                    data = response.json()
        except Exception as exc:
            raise ToolFailure(
                GEOCODING_FAILED,
                f"The geocoding service could not be reached, so {lat}, {lon} could not "
                "be described. Report the coordinates themselves and say that no "
                "address was available.",
            ) from exc

        results = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            coords = feature.get("geometry", {}).get("coordinates", [None, None])
            if coords[0] is None:
                continue
            results.append({
                "address": props.get("label", ""),
                "locality": props.get("locality") or props.get("localadmin") or "",
                "county": props.get("region") or "",
                "match_level": _match_level(props.get("layer", "")),
                # Distance from the point asked about to the place that was
                # matched. Without it the caller cannot tell a doorway from a
                # label pinned half a kilometre away.
                "distance_m": round(_haversine_m(lat, lon,
                                                 float(coords[1]), float(coords[0]))),
            })
        return results

    async def reverse(self, lat: float, lon: float) -> dict[str, Any]:
        """Nearest addressable place to a point, or a failure the model can act on."""
        d = config.CACHE_COORD_DECIMALS
        key = f"{round(lat, d)},{round(lon, d)}"
        cached = await repository.cache_get(REVERSE_CACHE_KIND, key)
        if cached is not None:
            if "error_code" in cached:
                raise ToolFailure(cached["error_code"], cached["message"])
            return cached

        results = await self._reverse(lat, lon)
        if not results:
            miss = {
                "error_code": GEOCODING_FAILED,
                "message": (
                    f"No addressable place was found near {lat}, {lon}. Report the "
                    "coordinates themselves rather than naming a place that is not "
                    "there."
                ),
            }
            await repository.cache_put(REVERSE_CACHE_KIND, key, miss)
            raise ToolFailure(miss["error_code"], miss["message"])

        best = results[0]
        if best["distance_m"] > _FAR_MATCH_M:
            best["note"] = (
                f"The nearest addressable place is {best['distance_m']} m from the point "
                "asked about, so this describes the surrounding area rather than the "
                "point itself. Say so when reporting it."
            )
        await repository.cache_put(REVERSE_CACHE_KIND, key, best)
        return best


provider = GeocodingProvider()
