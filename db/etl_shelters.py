"""
ETL: source shelter inventory -> `enviroo.shelters` + a named snapshot.

The source table is read-only and is never modified. What this script does is
normalization, and every rule it applies is a research decision that has to
be stated rather than buried:

* **Status.** Four source values (with diacritics) map onto four codes.
  Anything else, including the literal '-', becomes `unknown` — not
  `functional`, because assuming a shelter works is the one error with real
  consequences.
* **Type.** The source spells the same thing three ways ('Public', 'public',
  'PUBLIC'); they are one value here.
* **Capacity.** Stored as text in the source. A value that is not a number
  ('-') becomes NULL, not zero: "no capacity recorded" and "no places" are
  different facts and merging them would fabricate the second.
* **Effective capacity.** Nominal for a functional shelter, a fixed fraction
  of nominal for a partially functional one, zero otherwise. The fraction is
  recorded in the snapshot, so any result can be reinterpreted later.
* **Rows without coordinates** are skipped and counted; a shelter with no
  location cannot participate in a spatial answer, and inventing one would be
  worse than omitting it.

Run:  python -m db.etl_shelters
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from datetime import date

import asyncpg

from app import config

SOURCE = config.SOURCE_TABLE
S = config.SCHEMA

# Source column names, quoted exactly as they exist — diacritics, spaces,
# parentheses and the 'Logitudine' typo included. This is the only file in
# the project allowed to know them.
COL_ID = '"id"'
COL_COUNTY = '"Județ"'
COL_LOCALITY = '"Localitate"'
COL_ADDRESS = '"Adresa adăpostului"'
COL_TYPE = '"Tipul adăpostului(privat/public)"'
COL_LAT = '"Latitudine"'
COL_LON = '"Logitudine"'
COL_CAPACITY = '"Capacitatea adăpostului (Număr Persoane)"'
COL_STATUS = '"Starea defuncționalitate"'

STATUS_MAP = {
    "funcțional": config.STATUS_FUNCTIONAL,
    "functional": config.STATUS_FUNCTIONAL,
    "parțial-funcțional": config.STATUS_PARTIAL,
    "partial-functional": config.STATUS_PARTIAL,
    "nefuncțional": config.STATUS_NON_FUNCTIONAL,
    "nefunctional": config.STATUS_NON_FUNCTIONAL,
}
TYPE_MAP = {
    "public": "public",
    "privat": "private",
    "private": "private",
}


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()


def normalize_status(raw: str | None) -> str:
    return STATUS_MAP.get(_norm(raw), config.STATUS_UNKNOWN)


def normalize_type(raw: str | None) -> str:
    return TYPE_MAP.get(_norm(raw), "unknown")


def normalize_capacity(raw: str | None) -> int | None:
    """Capacity as an integer, or None when the source records none."""
    if raw is None:
        return None
    match = re.search(r"\d+", str(raw))
    if not match:
        return None
    value = int(match.group())
    return value if value > 0 else None


def effective_capacity(nominal: int | None, status: str) -> int | None:
    if nominal is None:
        return None
    if status == config.STATUS_FUNCTIONAL:
        return nominal
    if status == config.STATUS_PARTIAL:
        return int(round(nominal * config.PARTIAL_CAPACITY_FACTOR))
    return 0


def shelter_identifier(source_id: int) -> str:
    """Stable, short, and independent of row order: S + zero-padded source id."""
    return f"S{int(source_id):05d}"


async def run(snapshot_id: str | None, notes: str) -> None:
    conn = await asyncpg.connect(dsn=config.dsn())
    try:
        rows = await conn.fetch(f"""
            SELECT {COL_ID} AS id, {COL_COUNTY} AS county, {COL_LOCALITY} AS locality,
                   {COL_ADDRESS} AS address, {COL_TYPE} AS type_raw,
                   {COL_LAT} AS lat, {COL_LON} AS lon,
                   {COL_CAPACITY} AS capacity_raw, {COL_STATUS} AS status_raw
            FROM {SOURCE}
            ORDER BY {COL_ID}
        """)
        print(f"source rows: {len(rows)}")

        prepared = []
        skipped = 0
        for row in rows:
            lat, lon = row["lat"], row["lon"]
            if lat is None or lon is None or not config.in_study_area(float(lat), float(lon)):
                skipped += 1
                continue
            status = normalize_status(row["status_raw"])
            nominal = normalize_capacity(row["capacity_raw"])
            prepared.append((
                shelter_identifier(row["id"]),
                int(row["id"]),
                (row["address"] or "").strip(),
                (row["locality"] or "").strip(),
                (row["county"] or "").strip(),
                normalize_type(row["type_raw"]),
                status,
                nominal,
                effective_capacity(nominal, status),
                status in config.USABLE_STATUSES,
                float(lon), float(lat),
            ))

        async with conn.transaction():
            await conn.execute(f"TRUNCATE {S}.shelters CASCADE")
            await conn.executemany(
                f"""
                INSERT INTO {S}.shelters
                    (shelter_id, source_id, address, locality, county, type, status,
                     capacity_nominal, capacity_effective, usable, geom, geom_m)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        ST_SetSRID(ST_MakePoint($11, $12), {config.SRID_WGS84}),
                        ST_Transform(ST_SetSRID(ST_MakePoint($11, $12),
                                     {config.SRID_WGS84}), {config.SRID_METRIC}))
                """,
                prepared,
            )

            sid = snapshot_id or f"enviroo-{date.today().isoformat()}"
            await conn.execute(f"UPDATE {S}.snapshot SET is_current = false")
            await conn.execute(
                f"""
                INSERT INTO {S}.snapshot
                    (snapshot_id, is_current, source_table, shelters_total,
                     shelters_skipped, partial_capacity_factor, notes)
                VALUES ($1, true, $2, $3, $4, $5, $6)
                ON CONFLICT (snapshot_id) DO UPDATE
                    SET is_current = true, created_at = now(),
                        shelters_total = EXCLUDED.shelters_total,
                        shelters_skipped = EXCLUDED.shelters_skipped,
                        partial_capacity_factor = EXCLUDED.partial_capacity_factor,
                        notes = EXCLUDED.notes
                """,
                sid, SOURCE, len(prepared), skipped,
                config.PARTIAL_CAPACITY_FACTOR, notes,
            )

        summary = await conn.fetch(f"""
            SELECT status, COUNT(*) AS n,
                   SUM(capacity_nominal) AS nominal,
                   SUM(capacity_effective) AS effective
            FROM {S}.shelters GROUP BY status ORDER BY n DESC
        """)
        print(f"loaded {len(prepared)} shelters, skipped {skipped} without usable coordinates")
        for r in summary:
            print(f"  {r['status']:<22} n={r['n']:<6} nominal={r['nominal']} "
                  f"effective={r['effective']}")
        print(f"snapshot: {snapshot_id or f'enviroo-{date.today().isoformat()}'}")
    finally:
        await conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build enviroo.shelters from the source inventory.")
    parser.add_argument("--snapshot-id", default=None,
                        help="Snapshot identifier (default: enviroo-YYYY-MM-DD).")
    parser.add_argument("--notes", default="Shelter inventory normalized from the source table.",
                        help="Free-text note stored with the snapshot.")
    args = parser.parse_args()
    asyncio.run(run(args.snapshot_id, args.notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
