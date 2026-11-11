"""
RoutingProvider — walking routes and walking-time polygons.

The provider is asked for *pedestrian network geometry*; the walking time is
derived here, from the route length and the speed of the requested profile.
That split is deliberate. A routing service exposes one pedestrian speed;
this study needs three that differ by exactly the factor stated in the
specification (3, 5 and 7 km/h) and by nothing else. Deriving the time makes
the profile the only variable, which is what the evaluation measures.

The same rescaling applies to isochrones: a 10-minute polygon at 3 km/h
covers the distance the provider reaches in 6 minutes at its own 5 km/h, so
the provider is asked for 6 minutes. The polygon is a true 10-minute polygon
for a person walking at 3 km/h, computed on the same street network.

Results are cached by rounded coordinates, because the evaluation runs the
same origins over and over and provider latency would otherwise leak into
every measurement.
"""
from __future__ import annotations

from typing import Any

import httpx

from app import config
from app.logging_utils import log_timing, setup_logging
from app.mcp import repository
from app.mcp.envelope import ROUTING_UNAVAILABLE, ToolFailure

log = setup_logging("enviroo_mcp")

ROUTE_CACHE = "route"
ISOCHRONE_CACHE = "isochrone"


def _round(value: float) -> float:
    return round(float(value), config.CACHE_COORD_DECIMALS)


def speed_kmh(profile: str) -> float:
    return config.SPEED_PROFILES[profile]


def minutes_for(distance_m: float, profile: str) -> float:
    """Walking time for a route length, at the profile's speed."""
    metres_per_minute = speed_kmh(profile) * 1000.0 / 60.0
    return round(distance_m / metres_per_minute, 1)


class RoutingProvider:
    """Adapter over the configured routing service (pedestrian profile)."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or config.ORS_BASE).rstrip("/")
        self.api_key = api_key if api_key is not None else config.ORS_API_KEY

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ToolFailure(
                ROUTING_UNAVAILABLE,
                "The routing service is not configured, so walking times and reachable "
                "areas cannot be computed. Report this rather than estimating them.",
            )
        return {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json, application/geo+json",
        }

    async def _post(self, path: str, body: dict, stage: str, **fields) -> dict:
        try:
            with log_timing(log, stage, **fields):
                async with httpx.AsyncClient(timeout=config.ORS_TIMEOUT_S) as client:
                    response = await client.post(
                        f"{self.base_url}{path}", headers=self._headers(), json=body
                    )
                    response.raise_for_status()
                    return response.json()
        except ToolFailure:
            raise
        except httpx.HTTPStatusError as exc:
            # 404 from the directions endpoint means "no pedestrian route",
            # which is a fact about the network, not an outage. It is handled
            # by the caller, so it is surfaced distinctly.
            if exc.response.status_code == 404:
                return {"_no_route": True}
            raise ToolFailure(
                ROUTING_UNAVAILABLE,
                "The routing service did not respond successfully "
                f"({exc.response.status_code}). Report that the walking computation "
                "could not be performed; do not substitute an estimate.",
            ) from exc
        except Exception as exc:
            raise ToolFailure(
                ROUTING_UNAVAILABLE,
                "The routing service did not respond. Report that the walking "
                "computation could not be performed; do not substitute an estimate.",
            ) from exc

    # ── Route between two points ─────────────────────────────────────────

    async def walk_route(self, from_lat: float, from_lon: float,
                         to_lat: float, to_lon: float) -> dict[str, Any]:
        """Pedestrian route length, in metres.

        Returns `{"distance_m": ...}` or `{"unreachable": True}` when the two
        points are not connected on the pedestrian network. The walking time
        is not computed here: it depends on the speed profile, which is a
        property of the person, not of the street.
        """
        key = (f"{_round(from_lat)},{_round(from_lon)}->"
               f"{_round(to_lat)},{_round(to_lon)}")
        cached = await repository.cache_get(ROUTE_CACHE, key)
        if cached is not None:
            return cached

        body = {
            "coordinates": [[from_lon, from_lat], [to_lon, to_lat]],
            "units": "m",
            "instructions": False,
            "geometry": False,
        }
        data = await self._post(
            f"/v2/directions/{config.ORS_PROFILE}", body,
            "route.provider", frm=[from_lat, from_lon], to=[to_lat, to_lon],
        )
        routes = data.get("routes") or []
        if data.get("_no_route") or not routes:
            result = {"unreachable": True}
        else:
            summary = routes[0].get("summary") or {}
            distance = summary.get("distance")
            if distance is None:
                result = {"unreachable": True}
            else:
                result = {"distance_m": round(float(distance))}
        await repository.cache_put(ROUTE_CACHE, key, result)
        return result

    # ── Reachable area ───────────────────────────────────────────────────

    async def isochrone(self, lat: float, lon: float, minutes: float, profile: str,
                        inbound: bool = False) -> dict[str, Any]:
        """The area reachable on foot within `minutes` at the profile's speed.

        `inbound=True` reverses the question — the area *from which* the point
        can be reached — which is what a shelter's service area means.
        """
        key = (f"{_round(lat)},{_round(lon)}|{minutes:g}|{profile}|"
               f"{'in' if inbound else 'out'}")
        cached = await repository.cache_get(ISOCHRONE_CACHE, key)
        if cached is not None:
            return cached

        # Rescale the budget into the provider's own walking speed.
        provider_seconds = int(round(
            minutes * 60.0 * speed_kmh(profile) / config.PROVIDER_WALK_SPEED_KMH
        ))
        body = {
            "locations": [[lon, lat]],
            "location_type": "destination" if inbound else "start",
            "range_type": "time",
            "range": [provider_seconds],
            "attributes": ["area"],
            "smoothing": 0.3,
        }
        data = await self._post(
            f"/v2/isochrones/{config.ORS_PROFILE}", body,
            "isochrone.provider", lat=lat, lon=lon, minutes=minutes, profile=profile,
        )
        features = data.get("features") or []
        if not features:
            raise ToolFailure(
                ROUTING_UNAVAILABLE,
                "The routing service returned no reachable area for this point. It may "
                "be too far from the pedestrian network; report this instead of "
                "approximating the area with a circle.",
            )
        feature = features[0]
        polygon = feature.get("geometry") or {}
        area_m2 = (feature.get("properties") or {}).get("area")
        result = {
            "polygon": polygon,
            "area_km2": round(float(area_m2) / 1e6, 3) if area_m2 else None,
        }
        await repository.cache_put(ISOCHRONE_CACHE, key, result)
        return result


provider = RoutingProvider()
