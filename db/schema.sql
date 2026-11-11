-- ENVIROO — analysis schema.
--
-- ENVIROO never writes to the source inventory. It builds its own schema by
-- ETL and reads only that, for three reasons: the source table's column
-- names carry diacritics, spaces and a typo that no query should depend on;
-- the normalization rules (status, type, capacity) must be recorded once and
-- applied identically everywhere; and a study needs a snapshot it can name.
--
-- Run:  psql -d postgis -f db/schema.sql

CREATE SCHEMA IF NOT EXISTS enviroo;

-- ── Snapshot ─────────────────────────────────────────────────────────────
-- One row per build. Its id travels in every successful tool response, so
-- any answer in the study can be traced back to the data that produced it.
CREATE TABLE IF NOT EXISTS enviroo.snapshot (
    snapshot_id              text PRIMARY KEY,
    created_at               timestamptz NOT NULL DEFAULT now(),
    is_current               boolean     NOT NULL DEFAULT true,
    source_table             text        NOT NULL,
    shelters_total           integer,
    shelters_skipped         integer,
    partial_capacity_factor  double precision NOT NULL,
    population_dataset       text,
    notes                    text
);

-- ── Shelters ─────────────────────────────────────────────────────────────
-- `status` is one of functional | partially_functional | non_functional |
-- unknown; `type` is public | private | unknown. Both are normalized here so
-- that no tool ever sees the unnormalized source vocabulary.
CREATE TABLE IF NOT EXISTS enviroo.shelters (
    shelter_id          text PRIMARY KEY,
    source_id           integer,
    address             text,
    locality            text,
    county              text,
    type                text NOT NULL,
    status              text NOT NULL,
    capacity_nominal    integer,          -- NULL when the source records none
    capacity_effective  integer,          -- nominal reduced by functional status
    usable              boolean NOT NULL,
    geom                geometry(Point, 4326) NOT NULL,
    geom_m              geometry(Point, 3844) NOT NULL
);

CREATE INDEX IF NOT EXISTS shelters_geom_m_gix ON enviroo.shelters USING GIST (geom_m);
CREATE INDEX IF NOT EXISTS shelters_geom_gix   ON enviroo.shelters USING GIST (geom);
CREATE INDEX IF NOT EXISTS shelters_status_ix  ON enviroo.shelters (status);
CREATE INDEX IF NOT EXISTS shelters_locality_ix ON enviroo.shelters (locality);

-- ── Population grid ──────────────────────────────────────────────────────
-- Regular cells in the projected CRS. `area_m2` is stored rather than
-- recomputed: the area-weighted rule divides by it on every query.
CREATE TABLE IF NOT EXISTS enviroo.population_cells (
    cell_id     bigint PRIMARY KEY,
    population  double precision NOT NULL,
    area_m2     double precision NOT NULL,
    geom_m      geometry(Polygon, 3844) NOT NULL
);

CREATE INDEX IF NOT EXISTS population_cells_gix ON enviroo.population_cells USING GIST (geom_m);

-- ── Provider cache ───────────────────────────────────────────────────────
-- Geocoding results, walking routes and isochrones, keyed by their inputs.
-- Its purpose is measurement hygiene: the same origins recur across every
-- evaluation run, and provider latency must not leak into the timings.
CREATE TABLE IF NOT EXISTS enviroo.provider_cache (
    kind        text NOT NULL,
    cache_key   text NOT NULL,
    payload     jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (kind, cache_key)
);

-- ── Service areas ────────────────────────────────────────────────────────
-- Materialized inverse isochrones and the population inside them.
CREATE TABLE IF NOT EXISTS enviroo.service_area (
    shelter_id        text NOT NULL REFERENCES enviroo.shelters(shelter_id) ON DELETE CASCADE,
    minutes           double precision NOT NULL,
    speed_profile     text NOT NULL,
    polygon           jsonb NOT NULL,
    geom_m            geometry(Geometry, 3844) NOT NULL,
    population_served double precision NOT NULL,
    area_km2          double precision,
    computed_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (shelter_id, minutes, speed_profile)
);

CREATE INDEX IF NOT EXISTS service_area_gix ON enviroo.service_area USING GIST (geom_m);
