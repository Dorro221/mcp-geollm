"""Create the ENVIROO schema and its tables (idempotent).

Equivalent to `psql -f db/schema.sql`, but usable wherever psql is not on the
path — which on the study workstation is most of the time.

Run:  python -m db.apply_schema
"""
from __future__ import annotations

import asyncio
import os
import sys

import asyncpg

from app import config

SCHEMA_SQL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")


async def run() -> None:
    with open(SCHEMA_SQL, encoding="utf-8") as fh:
        sql = fh.read()
    conn = await asyncpg.connect(dsn=config.dsn())
    try:
        await conn.execute(sql)
        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = $1 ORDER BY 1",
            config.SCHEMA,
        )
        print(f"schema '{config.SCHEMA}': " + ", ".join(t["table_name"] for t in tables))
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run())
    sys.exit(0)
