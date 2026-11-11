"""
One JSON line per tool invocation (specification §1.7).

This file — not the console, not the application log — is the primary data
source for the study's latency and orchestration metrics, so it is written
with a fixed set of keys and nothing else: timestamp, run, snapshot, tool,
arguments, duration, status, and a compact summary of the result. Whole
geometries are never written; a polygon of 400 vertices would drown the
signal and tell us nothing that `vertices` does not.

`run_id` is what makes the file re-readable after the fact. The web server
forwards its request id on every MCP call, so each line can be attributed to
the benchmark run that caused it. Without it the only link was the timestamp,
and a scoring rule corrected after a campaign could not be applied to that
campaign's own data - it had to be run again.

Never writes to stdout: under a stdio transport that would corrupt the MCP
stream itself.
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from app.logging_utils import REQUEST_ID_HEADER, get_request_id

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
LOG_PATH = os.path.join(
    os.getenv("ENVIROO_TOOL_LOG_DIR", os.path.join(_ROOT, "logs")), "tool_calls.jsonl"
)

_lock = threading.Lock()


def _run_id() -> str | None:
    """The benchmark run this call belongs to, or None outside one.

    The web server forwards its request id as a header on every MCP call and
    the HTTP middleware records it, so the id is always present on the request.
    The tool body, however, runs in a task the Streamable HTTP session manager
    owns, and that task does not inherit the context the middleware set - the
    contextvar alone therefore reads as unset here. The header is read back off
    the request the MCP runtime exposes for the tool call in progress, with the
    contextvar kept as the first source for the transports where it does work.
    """
    rid = get_request_id()
    if rid and rid != "-":
        return rid
    try:
        from mcp.server.lowlevel.server import request_ctx

        request = getattr(request_ctx.get(), "request", None)
        headers = getattr(request, "headers", None)
        if headers is not None:
            rid = headers.get(REQUEST_ID_HEADER)
            if rid:
                return rid
    except Exception:
        # Instrumentation: outside a request, or on a transport that exposes no
        # request, the line is still written - without the attribution.
        pass
    return None


def _write(record: dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _lock:
            with open(LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        # The log is instrumentation. A failure to write it must never turn
        # a working tool call into an error for the user.
        pass


def summarize(result: Any) -> dict[str, Any]:
    """Compact result summary: the signal fields, never the payload."""
    if not isinstance(result, dict):
        return {"kind": type(result).__name__}
    out: dict[str, Any] = {}
    for key in ("count", "population", "cells_counted", "walk_time_min", "distance_m",
                "area_km2", "demand_ratio", "population_served", "match_level",
                "total_capacity_nominal", "total_capacity_effective",
                "capacity_nominal", "capacity_effective", "status", "minutes",
                "speed_profile", "radius_m", "shelter_id"):
        if key in result:
            out[key] = result[key]
    polygon = result.get("polygon") or result.get("service_area")
    if isinstance(polygon, dict):
        try:
            out["vertices"] = sum(len(ring) for ring in polygon.get("coordinates", []))
        except TypeError:
            pass
    if result.get("note"):
        out["note"] = str(result["note"])[:200]
    return out


@contextmanager
def record(tool: str, args: dict[str, Any], snapshot_id: str):
    """Time one invocation and append its line, whatever the outcome.

    Yields a one-element list; the tool stores its result in it so the
    summary can be computed from what was actually returned.
    """
    started = time.perf_counter()
    holder: list[Any] = [None]
    status = "ok"
    error_code = None
    try:
        yield holder
    except Exception as exc:
        status = "exception"
        error_code = type(exc).__name__
        raise
    finally:
        result = holder[0]
        if status == "ok" and isinstance(result, dict) and "error_code" in result:
            status = "error"
            error_code = result["error_code"]
        _write({
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": _run_id(),
            "snapshot_id": snapshot_id,
            "tool": tool,
            "args": args,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "status": status,
            "error_code": error_code,
            "result": summarize(result) if status == "ok" else None,
        })
