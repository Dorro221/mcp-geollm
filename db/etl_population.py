"""
ETL: the gridded population layer that `population_in_area` reads.

Two modes, and the difference between them matters enough to be stated in
the snapshot itself:

**--source <file.csv>** — import a real gridded product (WorldPop, GHS-POP,
census grid) exported as `lon,lat,population` centroids. Use this whenever
such a product is available: it is a measurement.

**default** — build a *modelled* surface. There is no open gridded census for
the study area in this repository, and `population_in_area` cannot return
nothing, so the default builds a plausible density surface: one normalized
Gaussian kernel per seed settlement, whose integral is that settlement's census
population, over a uniform baseline calibrated so the whole surface totals the
national resident population. The result has realistic spatial structure — dense
towns, sparse countryside — and is exactly reproducible, but a single cell's
value is a model output, not an observation.

This is why the snapshot records `population_dataset`, and why the README
says so in plain words. A study that quietly passed a modelled surface off as
census data would be measuring the model's arithmetic and reporting it as
fact.

Run:  python -m db.etl_population
      python -m db.etl_population --source worldpop_cells.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import math
import os
import sys

import asyncpg

from app import config

S = config.SCHEMA
_HERE = os.path.dirname(os.path.abspath(__file__))
CITIES_CSV = os.path.join(_HERE, "cities.csv")

CELL_M = 1000                 # grid resolution, metres
MASK_KM = 12.0                # keep cells within this distance of a shelter
# Resident population of the study area, 2021 census. The surface is
# calibrated to it: whatever the seed settlements do not account for is spread
# uniformly over the inhabited mask, so the layer totals the true national
# figure instead of an arbitrary density chosen by hand.
NATIONAL_POPULATION = 19_053_815
DATASET_MODELLED = (
    "modelled-urban-kernel-v1 (census-seeded Gaussian surface; NOT an official "
    "gridded census product — see README, 'Population data')"
)


def kernel_sigma_m(population: float) -> float:
    """Kernel width for a settlement of a given size.

    Larger towns spread further, but sub-linearly: doubling the population of
    a town does not double its radius, it grows roughly with the square root
    of the area needed to hold it.
    """
    radius_km = 1.2 * math.sqrt(max(population, 1.0) / 100_000.0) + 1.0
    return radius_km * 1000.0 / 1.5


def load_seeds(path: str) -> list[dict]:
    seeds = []
    with open(path, encoding="utf-8") as fh:
        rows = [line for line in fh if not line.lstrip().startswith("#")]
    for row in csv.DictReader(rows):
        try:
            seeds.append({
                "name": row["name"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "population": float(row["population"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return seeds


async def build_grid(conn: asyncpg.Connection) -> int:
    """Create empty cells covering the inhabited mask.

    The mask is the union of the areas within `MASK_KM` of a shelter. Using
    the inventory itself to define where people live keeps the two layers
    consistent: there is never a populated cell in a region the shelter data
    knows nothing about.
    """
    await conn.execute(f"TRUNCATE {S}.population_cells")
    inserted = await conn.fetchval(f"""
        WITH extent AS (
            -- ST_Extent returns a box2d; the cast to geometry drops the SRID,
            -- which the grid would then carry into every distance test.
            SELECT ST_Expand(
                     ST_SetSRID(ST_Extent(geom_m)::geometry, {config.SRID_METRIC}),
                     {int(MASK_KM * 1000)}) AS g
            FROM {S}.shelters
        ), grid AS (
            SELECT (ST_SquareGrid({CELL_M}, extent.g)).* FROM extent
        ), kept AS (
            SELECT row_number() OVER (ORDER BY i, j) AS cell_id,
                   ST_SetSRID(geom, {config.SRID_METRIC}) AS geom
            FROM grid
            WHERE EXISTS (
                SELECT 1 FROM {S}.shelters s
                WHERE ST_DWithin(s.geom_m, ST_Centroid(grid.geom), {MASK_KM * 1000})
            )
        )
        INSERT INTO {S}.population_cells (cell_id, population, area_m2, geom_m)
        SELECT cell_id, 0, ST_Area(geom), geom FROM kept
        RETURNING 1
    """)
    return await conn.fetchval(f"SELECT COUNT(*) FROM {S}.population_cells") or 0


async def fill_modelled(conn: asyncpg.Connection, seeds: list[dict]) -> None:
    """Distribute each settlement's population over the grid, then normalize.

    Normalizing per settlement is what makes the surface honest about its own
    assumption: whatever the kernel does spatially, the people it spreads add
    up to exactly the census figure it started from.
    """
    cells = await conn.fetch(f"""
        SELECT cell_id, ST_X(ST_Centroid(geom_m)) AS x, ST_Y(ST_Centroid(geom_m)) AS y,
               area_m2
        FROM {S}.population_cells
    """)
    if not cells:
        raise RuntimeError("The population grid is empty; build it first.")

    # Seeds in the projected CRS, so kernel distances are metres.
    seed_points = await conn.fetch(
        f"""
        SELECT ST_X(g) AS x, ST_Y(g) AS y, p AS population FROM (
            SELECT ST_Transform(ST_SetSRID(ST_MakePoint(lon, lat),
                   {config.SRID_WGS84}), {config.SRID_METRIC}) AS g, p
            FROM unnest($1::double precision[], $2::double precision[],
                        $3::double precision[]) AS t(lat, lon, p)
        ) q
        """,
        [s["lat"] for s in seeds], [s["lon"] for s in seeds],
        [s["population"] for s in seeds],
    )

    mask_km2 = sum(float(c["area_m2"]) for c in cells) / 1e6
    seeded = sum(float(s["population"]) for s in seed_points)
    baseline_density = max((NATIONAL_POPULATION - seeded) / mask_km2, 0.0)
    print(f"baseline: {baseline_density:.1f} persons/km² over {mask_km2:,.0f} km² "
          f"(national population minus the {seeded:,.0f} in seed settlements)")

    values = {int(c["cell_id"]): float(c["area_m2"]) / 1e6 * baseline_density
              for c in cells}
    xs = [(int(c["cell_id"]), float(c["x"]), float(c["y"])) for c in cells]

    for seed in seed_points:
        sigma = kernel_sigma_m(seed["population"])
        cutoff = 3.5 * sigma
        two_sigma_sq = 2.0 * sigma * sigma
        weights: dict[int, float] = {}
        total = 0.0
        for cell_id, x, y in xs:
            dx, dy = x - seed["x"], y - seed["y"]
            if abs(dx) > cutoff or abs(dy) > cutoff:
                continue
            d2 = dx * dx + dy * dy
            if d2 > cutoff * cutoff:
                continue
            w = math.exp(-d2 / two_sigma_sq)
            weights[cell_id] = w
            total += w
        if total <= 0:
            continue
        scale = seed["population"] / total
        for cell_id, w in weights.items():
            values[cell_id] += w * scale

    await conn.execute("CREATE TEMP TABLE pop_update (cell_id bigint, population double precision) ON COMMIT DROP")
    await conn.copy_records_to_table(
        "pop_update", records=[(cid, round(v, 3)) for cid, v in values.items()]
    )
    await conn.execute(f"""
        UPDATE {S}.population_cells c
        SET population = u.population
        FROM pop_update u
        WHERE c.cell_id = u.cell_id
    """)


async def fill_from_file(conn: asyncpg.Connection, path: str, cell_m: int) -> str:
    """Import a real gridded product given as `lon,lat,population` centroids."""
    records = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=1):
            try:
                records.append((i, float(row["lon"]), float(row["lat"]),
                                float(row["population"])))
            except (KeyError, TypeError, ValueError):
                continue
    if not records:
        raise RuntimeError(f"No usable rows in {path} (expected columns lon,lat,population)")

    await conn.execute(f"TRUNCATE {S}.population_cells")
    await conn.execute("""
        CREATE TEMP TABLE pop_input (cell_id bigint, lon double precision,
                                     lat double precision, population double precision)
        ON COMMIT DROP
    """)
    await conn.copy_records_to_table("pop_input", records=records)
    half = cell_m / 2.0
    await conn.execute(f"""
        INSERT INTO {S}.population_cells (cell_id, population, area_m2, geom_m)
        SELECT cell_id, population, {cell_m * cell_m},
               ST_Envelope(ST_Buffer(
                   ST_Transform(ST_SetSRID(ST_MakePoint(lon, lat), {config.SRID_WGS84}),
                                {config.SRID_METRIC}), {half}, 1))
        FROM pop_input
    """)
    return f"imported:{os.path.basename(path)} ({len(records)} cells at {cell_m} m)"


async def run(source: str | None, cell_m: int) -> None:
    conn = await asyncpg.connect(dsn=config.dsn())
    try:
        async with conn.transaction():
            if source:
                dataset = await fill_from_file(conn, source, cell_m)
            else:
                cells = await build_grid(conn)
                print(f"grid: {cells} cells of {CELL_M} m within {MASK_KM:g} km of a shelter")
                seeds = load_seeds(CITIES_CSV)
                print(f"seeds: {len(seeds)} settlements, "
                      f"{sum(s['population'] for s in seeds):,.0f} people")
                await fill_modelled(conn, seeds)
                dataset = DATASET_MODELLED

            await conn.execute(
                f"UPDATE {S}.snapshot SET population_dataset = $1 WHERE is_current",
                dataset,
            )

        row = await conn.fetchrow(f"""
            SELECT COUNT(*) AS cells, SUM(population) AS total,
                   MAX(population) AS peak
            FROM {S}.population_cells
        """)
        print(f"population layer: {row['cells']} cells, total {row['total']:,.0f}, "
              f"densest cell {row['peak']:,.0f}")
        print(f"dataset: {dataset}")
    finally:
        await conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the ENVIROO population layer.")
    parser.add_argument("--source", default=None,
                        help="CSV of real gridded population centroids (lon,lat,population).")
    parser.add_argument("--cell-m", type=int, default=CELL_M,
                        help="Cell size in metres for an imported product (default 1000).")
    args = parser.parse_args()
    asyncio.run(run(args.source, args.cell_m))
    return 0


if __name__ == "__main__":
    sys.exit(main())
