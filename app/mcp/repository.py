"""
Data access for ENVIROO. All SQL in the system lives in this module.

Two consequences, both deliberate:

* the MCP surface never leaks a table or column name — descriptions,
  arguments, results and error messages are written in domain terms, and the
  only place that knows otherwise is here;
* every metric computation (distance, containment, intersection area) is done
  by PostGIS on the projected geometry, not in Python on degrees.

The tool boundary speaks EPSG:4326; the transform happens inside these
queries and nowhere else.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import asyncpg

from app import config

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()
_snapshot_cache: dict[str, Any] | None = None

S = config.SCHEMA             # schema holding the ENVIROO snapshot
M = config.SRID_METRIC        # projected CRS: metres are metres
W = config.SRID_WGS84         # the only CRS the tool boundary speaks

# The shelter record as every tool reports it. Kept in one place so the
# search tools, the detail tool and the area tool cannot drift apart.
SHELTER_COLUMNS = (
    "shelter_id, address, locality, county, type, status, "
    "capacity_nominal, capacity_effective"
)
SHELTER_COLUMNS_S = ", ".join("s." + c.strip() for c in SHELTER_COLUMNS.split(","))


async def pool() -> asyncpg.Pool:
    global _pool
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is None:
            _pool = await asyncpg.create_pool(
                dsn=config.dsn(),
                min_size=config.PG_POOL_MIN,
                max_size=config.PG_POOL_MAX,
                command_timeout=60,
            )
    return _pool


async def close() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def fetch(sql: str, *args) -> list[dict]:
    p = await pool()
    async with p.acquire() as con:
        return [dict(r) for r in await con.fetch(sql, *args)]


async def fetchrow(sql: str, *args) -> dict | None:
    p = await pool()
    async with p.acquire() as con:
        row = await con.fetchrow(sql, *args)
    return dict(row) if row else None


async def execute(sql: str, *args) -> str:
    p = await pool()
    async with p.acquire() as con:
        return await con.execute(sql, *args)


# ── Snapshot ──────────────────────────────────────────────────────────────

async def snapshot() -> dict:
    """The current data snapshot, read once per process.

    Its identifier accompanies every successful response: two runs of the
    study are comparable only if they were computed over the same data, and
    the only way for a reader to check that is for the tools to say so.
    """
    global _snapshot_cache
    if _snapshot_cache is not None:
        return _snapshot_cache
    row = await fetchrow(
        f"SELECT * FROM {S}.snapshot WHERE is_current ORDER BY created_at DESC LIMIT 1"
    )
    if row is None:
        raise RuntimeError(
            f"No current snapshot in schema '{S}'. Build it with: python -m db.etl_shelters"
        )
    _snapshot_cache = row
    return row


async def snapshot_id() -> str:
    return (await snapshot())["snapshot_id"]


def _usable() -> list[str]:
    return list(config.USABLE_STATUSES)


# ── Shelters ──────────────────────────────────────────────────────────────

async def shelters_near(lat: float, lon: float, radius_m: int,
                        usable_only: bool = True, limit: int | None = None) -> list[dict]:
    """Shelters within a straight-line radius, nearest first.

    `ST_DWithin` plus the KNN operator on the projected geometry: an index
    scan, not a distance computation over the whole inventory.
    """
    limit = int(limit or config.MAX_SHELTERS_RETURNED)
    usable_clause = "AND s.status = ANY($4::text[])" if usable_only else ""
    sql = f"""
        WITH origin AS (
            SELECT ST_Transform(ST_SetSRID(ST_MakePoint($2, $1), {W}), {M}) AS g
        )
        SELECT {SHELTER_COLUMNS_S},
               round(ST_Distance(s.geom_m, origin.g))::int AS distance_m
        FROM {S}.shelters s, origin
        WHERE ST_DWithin(s.geom_m, origin.g, $3)
          {usable_clause}
        ORDER BY s.geom_m <-> origin.g
        LIMIT {limit}
    """
    args: list[Any] = [lat, lon, float(radius_m)]
    if usable_only:
        args.append(_usable())
    return await fetch(sql, *args)


async def shelter_by_id(shelter_id: str) -> dict | None:
    """One shelter, whatever its status, with its geometry as GeoJSON."""
    row = await fetchrow(
        f"""
        SELECT {SHELTER_COLUMNS},
               ST_Y(geom) AS lat, ST_X(geom) AS lon,
               ST_AsGeoJSON(geom) AS geometry
        FROM {S}.shelters
        WHERE upper(shelter_id) = upper($1)
        """,
        str(shelter_id).strip(),
    )
    if row and isinstance(row.get("geometry"), str):
        # The driver hands back JSON columns as text; the tool boundary
        # publishes a GeoJSON object, not a string containing one.
        row["geometry"] = json.loads(row["geometry"])
    return row


async def straight_line_m(lat: float, lon: float, shelter_id: str) -> int | None:
    """Straight-line distance from a point to one shelter, in metres.

    Computed in the same projected CRS as the search tool, so that the
    straight line a model compares against a route length is the same number
    it was given by `find_shelters_near`.
    """
    row = await fetchrow(
        f"""
        SELECT round(ST_Distance(
                 s.geom_m,
                 ST_Transform(ST_SetSRID(ST_MakePoint($2, $1), {W}), {M})))::int AS d
        FROM {S}.shelters s
        WHERE s.shelter_id = $3
        """,
        lat, lon, shelter_id,
    )
    return row["d"] if row else None


async def shelters_in_polygon(polygon_geojson: str, usable_only: bool = True,
                              limit: int | None = None) -> list[dict]:
    """Shelters lying inside a polygon — one spatial join against the index."""
    usable_clause = "AND s.status = ANY($2::text[])" if usable_only else ""
    sql = f"""
        WITH area AS (
            SELECT ST_Transform(
                     ST_MakeValid(ST_SetSRID(ST_GeomFromGeoJSON($1), {W})), {M}) AS g
        )
        SELECT {SHELTER_COLUMNS_S}
        FROM {S}.shelters s, area
        WHERE ST_Intersects(s.geom_m, area.g)
          {usable_clause}
        ORDER BY s.shelter_id
        LIMIT {int(limit or config.MAX_SHELTERS_RETURNED * 20)}
    """
    args: list[Any] = [polygon_geojson]
    if usable_only:
        args.append(_usable())
    return await fetch(sql, *args)


async def shelters_by_ids(ids: list[str]) -> list[dict]:
    """Geometry and attributes for a list of shelters.

    Serves the map in the web UI, which needs coordinates the MCP surface
    deliberately does not carry: the tools answer questions about shelters,
    the map draws them. Keeping this out of the tool responses keeps the
    payloads the models see small and identical to the specification.
    """
    if not ids:
        return []
    return await fetch(
        f"""
        SELECT {SHELTER_COLUMNS}, ST_Y(geom) AS lat, ST_X(geom) AS lon
        FROM {S}.shelters
        WHERE shelter_id = ANY($1::text[])
        """,
        [str(i).strip() for i in ids][:500],
    )


# ── Geometry checks ───────────────────────────────────────────────────────

async def inspect_polygon(polygon_geojson: str) -> dict | None:
    """Validate a supplied polygon and measure it.

    Trivially repairable geometry (a ring that does not close, a
    self-touching vertex) is repaired by `ST_MakeValid`; anything that does
    not survive parsing is rejected upstream. The area comes back in km² so
    the caller can enforce the area limit before running a heavy query.
    """
    return await fetchrow(
        f"""
        WITH raw AS (
            SELECT ST_SetSRID(ST_GeomFromGeoJSON($1), {W}) AS g
        ), fixed AS (
            SELECT ST_MakeValid(g) AS g, ST_IsValid(g) AS was_valid FROM raw
        )
        SELECT was_valid,
               GeometryType(g) AS geom_type,
               ST_IsEmpty(g) AS is_empty,
               ST_Area(ST_Transform(g, {M})) / 1e6 AS area_km2
        FROM fixed
        """,
        polygon_geojson,
    )


# ── Population ────────────────────────────────────────────────────────────

async def population_in_polygon(polygon_geojson: str) -> dict | None:
    """Resident population inside a polygon, area-weighted.

    A cell the polygon covers halfway contributes half its population. The
    rule is applied here, once, for every tool that counts people, so that
    `population_in_area` and `shelter_service_area` can never disagree about
    the same polygon.
    """
    return await fetchrow(
        f"""
        WITH area AS (
            SELECT ST_Transform(
                     ST_MakeValid(ST_SetSRID(ST_GeomFromGeoJSON($1), {W})), {M}) AS g
        )
        SELECT
            COALESCE(SUM(c.population
                         * ST_Area(ST_Intersection(c.geom_m, area.g))
                         / NULLIF(c.area_m2, 0)), 0) AS population,
            COUNT(*) AS cells_counted,
            MAX(ST_Area(area.g) / 1e6) AS area_km2
        FROM {S}.population_cells c, area
        WHERE ST_Intersects(c.geom_m, area.g)
        """,
        polygon_geojson,
    )


async def population_coverage_fraction(polygon_geojson: str) -> float:
    """How much of a polygon lies over the population layer, 0.0 to 1.0.

    The layer need not cover the whole study area - it currently covers only
    Cluj-Napoca, while the shelter inventory is national. Without this, a
    polygon outside the layer counts zero people, and a polygon half outside
    counts half of them, both indistinguishable from a real answer. The tools
    that read population use this to say what they do not know.

    Coverage is measured against the layer's EXTENT, not against the cells it
    happens to contain. A park or a river inside Cluj holds nobody and so has
    no cell, but its population is known - it is zero. Measuring against the
    cells would report those holes as missing data and attach a caveat to
    every answer in the city.
    """
    extent = await _population_extent()
    if extent is None:
        return 0.0
    row = await fetchrow(
        f"""
        WITH area AS (
            SELECT ST_Transform(
                     ST_MakeValid(ST_SetSRID(ST_GeomFromGeoJSON($1), {W})), {M}) AS g
        )
        SELECT CASE WHEN ST_Area(area.g) > 0
                    THEN COALESCE(ST_Area(ST_Intersection(
                             ST_GeomFromText($2, {M}), area.g)), 0) / ST_Area(area.g)
                    ELSE 0 END AS fraction
        FROM area
        """,
        polygon_geojson, extent,
    )
    return float(row["fraction"]) if row and row["fraction"] is not None else 0.0


_extent_wkt: str | None = None


async def _population_extent() -> str | None:
    """The population layer's footprint, as WKT in the metric CRS.

    Read once: it changes only when the layer is rebuilt, and the ETL runs
    against a stopped server.
    """
    global _extent_wkt
    if _extent_wkt is None:
        row = await fetchrow(
            f"SELECT ST_AsText(ST_Extent(geom_m)) AS wkt FROM {S}.population_cells")
        _extent_wkt = row["wkt"] if row else None
    return _extent_wkt


# ── Provider cache ────────────────────────────────────────────────────────
# The same addresses and origins recur across every evaluation run. Caching
# them removes provider latency variance from the measurements — that is the
# reason the cache exists; speed is a side effect.

async def cache_get(kind: str, key: str) -> Any | None:
    # One switch, checked here rather than at each call site: geocoding,
    # routing and isochrones all reach the provider through these two
    # functions, so this is the only place that has to know.
    if not config.PROVIDER_CACHE_ENABLED:
        return None
    row = await fetchrow(
        f"SELECT payload FROM {S}.provider_cache WHERE kind = $1 AND cache_key = $2",
        kind, key,
    )
    if row is None:
        return None
    payload = row["payload"]
    return json.loads(payload) if isinstance(payload, str) else payload


async def cache_put(kind: str, key: str, payload: Any) -> None:
    if not config.PROVIDER_CACHE_ENABLED:
        return
    await execute(
        f"""
        INSERT INTO {S}.provider_cache (kind, cache_key, payload)
        VALUES ($1, $2, $3::jsonb)
        ON CONFLICT (kind, cache_key)
        DO UPDATE SET payload = EXCLUDED.payload, created_at = now()
        """,
        kind, key, json.dumps(payload, ensure_ascii=False),
    )


# ── Service areas ─────────────────────────────────────────────────────────

async def service_area_get(shelter_id: str, minutes: float, profile: str) -> dict | None:
    row = await fetchrow(
        f"""
        SELECT shelter_id, minutes, speed_profile, polygon,
               population_served, area_km2
        FROM {S}.service_area
        WHERE shelter_id = $1 AND minutes = $2 AND speed_profile = $3
        """,
        shelter_id, float(minutes), profile,
    )
    if row and isinstance(row.get("polygon"), str):
        row["polygon"] = json.loads(row["polygon"])
    return row


async def service_area_put(shelter_id: str, minutes: float, profile: str,
                           polygon: dict, population: float, area_km2: float) -> None:
    await execute(
        f"""
        INSERT INTO {S}.service_area
            (shelter_id, minutes, speed_profile, polygon, geom_m,
             population_served, area_km2)
        VALUES ($1, $2, $3, $4::jsonb,
                ST_Transform(ST_MakeValid(ST_SetSRID(ST_GeomFromGeoJSON($4), {W})), {M}),
                $5, $6)
        ON CONFLICT (shelter_id, minutes, speed_profile) DO UPDATE
            SET polygon = EXCLUDED.polygon, geom_m = EXCLUDED.geom_m,
                population_served = EXCLUDED.population_served,
                area_km2 = EXCLUDED.area_km2, computed_at = now()
        """,
        shelter_id, float(minutes), profile,
        json.dumps(polygon, ensure_ascii=False), float(population), float(area_km2),
    )


# ── Health ────────────────────────────────────────────────────────────────

async def health() -> dict:
    row = await fetchrow(
        f"""
        SELECT (SELECT COUNT(*) FROM {S}.shelters) AS shelters,
               (SELECT COUNT(*) FROM {S}.population_cells) AS population_cells,
               (SELECT COUNT(*) FROM {S}.service_area) AS service_areas
        """
    ) or {}
    row["snapshot_id"] = await snapshot_id()
    return row
