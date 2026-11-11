"""
ENVIROO — FastAPI web server.

Plug-and-play by construction:

- MCP servers are URLs and nothing else. At start-up the agent asks each one
  for its tools (`tools/list`, JSON-RPC) and builds both the tool definitions
  it offers the model and the routing table tool -> server. No tool is
  declared statically anywhere in this file, so the study can add, remove or
  replace a server without touching code.
- LLMs are declared in models.json (provider base URL plus the name of the
  environment variable holding its key). Any OpenAI-compatible
  chat-completions endpoint works. The model is chosen per conversation from
  the UI, which is what makes four models comparable over one identical
  harness.

Serves the frontend and the /ws/chat WebSocket. Default port: 8090.
"""
import asyncio
import hashlib
import itertools
import json
import os
import re
import time
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import httpx

from app import config
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

from app.logging_utils import (
    REQUEST_ID_HEADER,
    RequestIdMiddleware,
    get_request_id,
    log_timing,
    new_request_id,
    set_request_id,
    setup_logging,
)

load_dotenv(override=True)

log = setup_logging("enviroo_web")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ── Configuration ─────────────────────────────────────────────────────────
MODELS_FILE = os.path.join(_ROOT, "models.json")
MCP_SERVERS_FILE = os.path.join(_ROOT, "mcp_servers.json")
# Hosts accepted for MCP servers added from the UI (empty = unrestricted).
# In production: MCP_ALLOWED_HOSTS=localhost,127.0.0.1
MCP_ALLOWED_HOSTS = [h.strip() for h in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
MAX_HISTORY = 30      # messages kept in the conversation context
MAX_ROUNDS = 12       # agentic rounds per question; reasoning models that
                      # explore before answering routinely use eight or more

# One system prompt for every description regime. It carries only conduct that
# holds for any tool-using assistant: ground the answer in the tools, ask when
# the location is missing, report what came back, leave geometry to the map,
# match the user's language. It names no tool, no unit, no service limit and no
# rule about operational status or undefined values, so under `bare` it cannot
# restore what the descriptions are being emptied of. Everything under test
# stays in the published descriptions, which are the apparatus being measured.
SYSTEM_SHARED = """You are an assistant for civil-shelter accessibility analysis.

Use the available tools for anything spatial. Do not answer from memory and do
not estimate a value a tool could compute.

If the user has not said where they are, ask; never substitute a landmark.
Report what the tools returned, including when they returned nothing, an error,
or an undefined value. Geometry is drawn on the map automatically - after a tool
that returned a polygon, do not copy its coordinates into another tool call or
into your answer.

Answer in the language the user writes in. Be concise and precise."""

# ── The two prompts of the first campaign ─────────────────────────────────
# Kept so that campaign stays reproducible under ENVIROO_SYSTEM_PROMPT=legacy,
# and not used otherwise. SYSTEM differs from SYSTEM_SHARED in one clause: it
# enumerated what the tools do ("locating an address, finding shelters, walking
# times, ..."), which is the one thing in it that told a bare-regime model what
# its tools were for. SYSTEM_MINIMAL kept only the role and the language rule.
# Pairing the two with the regimes made the prompt a second experimental factor
# alongside the descriptions, which is what the shared prompt above removes.
SYSTEM = """You are an assistant for civil-shelter accessibility analysis.

Use the available tools for anything spatial: locating an address, finding
shelters, walking times, reachable areas, capacity and population. Do not
answer from memory and do not estimate a value a tool could compute.

If the user has not said where they are, ask; never substitute a landmark.
Report what the tools returned, including when they returned nothing, an
error, or an undefined value. Geometry is drawn on the map automatically -
after a tool that returned a polygon, do not copy its coordinates into another
tool call or into your answer.

Answer in the language the user writes in. Be concise and precise."""

# The minimal arm's system prompt. Everything the rich one says about asking
# rather than guessing, about not answering from memory, and about reporting
# empty and undefined results would otherwise supply from the prompt exactly
# what the minimal descriptions withhold - and the two arms would differ far
# less than the surfaces do. What is left is the role and the language rule.
SYSTEM_MINIMAL = """You are an assistant for civil-shelter accessibility analysis.

Answer in the language the user writes in."""


def _system_for(regime: str) -> str:
    """The system prompt to send under a description regime.

    Under the default `shared` setting the answer does not depend on the
    regime: one prompt serves every arm, so the arms differ in their published
    tool descriptions alone and the prompt is not a second factor. `legacy`
    restores the pairing of the first campaign, whose rows are therefore not
    comparable with runs made under `shared` - which is why every row records
    both the setting and the hash of the prompt actually sent.
    """
    if config.SYSTEM_PROMPT_REGIME == "legacy":
        return SYSTEM if regime == "rich" else SYSTEM_MINIMAL
    return SYSTEM_SHARED


def prompt_fingerprint(regime: str) -> str:
    """Short hash of the prompt sent under `regime`, recorded with every run."""
    return hashlib.sha256(_system_for(regime).encode("utf-8")).hexdigest()[:16]


# ── MCP servers (mcp_servers.json, editable from the UI) ──────────────────

def _derive_label(url: str) -> str:
    return urlparse(url).netloc or url


def _validate_server_url(url: str) -> None:
    """Reject dangerous URLs before the backend fetches them (SSRF)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Invalid URL: only http/https are allowed")
    if not parsed.hostname:
        raise ValueError("Invalid URL: missing host")
    if MCP_ALLOWED_HOSTS and parsed.hostname not in MCP_ALLOWED_HOSTS:
        raise ValueError(
            f"Host '{parsed.hostname}' is not allowed (MCP_ALLOWED_HOSTS: "
            f"{', '.join(MCP_ALLOWED_HOSTS)})"
        )


def _default_servers() -> list[dict]:
    urls = [u.strip() for u in os.getenv(
        "MCP_SERVERS", "http://localhost:9100/mcp"
    ).split(",") if u.strip()]
    return [{"url": u, "label": _derive_label(u), "enabled": True} for u in urls]


def load_servers() -> list[dict]:
    """Registered MCP servers from mcp_servers.json; falls back to .env."""
    if os.path.exists(MCP_SERVERS_FILE):
        try:
            with open(MCP_SERVERS_FILE, encoding="utf-8") as f:
                raw = json.load(f).get("servers", [])
            servers: list[dict] = []
            seen: set[str] = set()
            for s in raw:
                url = (s.get("url") or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                servers.append({
                    "url": url,
                    "label": (s.get("label") or "").strip() or _derive_label(url),
                    # Absent means on: a registry written before this flag
                    # existed described servers that were all being queried.
                    "enabled": bool(s.get("enabled", True)),
                })
            if servers:
                return servers
        except Exception as e:
            log.error("mcp_servers.json is invalid (%s); using .env", e)
    return _default_servers()


def save_servers() -> None:
    try:
        with open(MCP_SERVERS_FILE, "w", encoding="utf-8") as f:
            json.dump({"servers": MCP_SERVERS}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error("could not save mcp_servers.json: %s", e)


def _server_urls() -> list[str]:
    """The servers discovery should query: the enabled ones.

    A disabled server stays in the registry with its URL and label, so the
    other description arm is one click away rather than a URL to retype.
    """
    return [s["url"] for s in MCP_SERVERS if s.get("enabled", True)]


MCP_SERVERS: list[dict] = load_servers()


# ── LLM models (models.json) ──────────────────────────────────────────────

def load_models() -> dict[str, dict]:
    """Read models.json and attach each model's key from the environment.

    An entry whose `api_key_env` is set but empty is marked unavailable and
    appears disabled in the UI. `api_key_env: null` means the provider needs
    no key (a local Ollama, for instance).

    Under ENVIROO_MODEL_ROUTE=direct, an entry's `route_direct` block replaces
    the fields it names, which is how the commercial pair is sent straight to
    its vendor instead of through the gateway. The default is the gateway, so
    that one client, one request format and one network path serve every model
    and the model is the only thing that differs between arms.
    """
    with open(MODELS_FILE, encoding="utf-8") as f:
        entries = json.load(f)["models"]
    models: dict[str, dict] = {}
    for m in entries:
        override = m.pop("route_direct", None) or {}
        if config.MODEL_ROUTE == "direct" and override:
            m.update(override)
        m["_route"] = ("direct" if (config.MODEL_ROUTE == "direct" and override)
                       else "gateway")
        key_env = m.get("api_key_env")
        if key_env:
            m["_api_key"] = os.getenv(key_env, "") or None
        else:
            m["_api_key"] = "no-key-required"
        models[m["id"]] = m
    return models


MODELS = load_models()

_clients: dict[str, AsyncOpenAI] = {}


def client_for(cfg: dict) -> AsyncOpenAI:
    """The provider client for a model, one per (endpoint, key).

    The timeout is set explicitly rather than left to the SDK, whose default
    pairs a 5-second CONNECT with a 600-second read. In a sweep that runs
    twenty-three tasks back to back, connections that take longer than five
    seconds to establish are common; the SDK retries twice and the turn fails
    after about sixteen seconds without the request ever reaching the model.
    Reading stays long, because a reasoning model may legitimately take
    minutes to answer.
    """
    cache_key = f"{cfg['base_url']}|{cfg['_api_key']}"
    if cache_key not in _clients:
        _clients[cache_key] = AsyncOpenAI(
            base_url=cfg["base_url"], api_key=cfg["_api_key"],
            timeout=httpx.Timeout(config.LLM_READ_TIMEOUT_S,
                                  connect=config.LLM_CONNECT_TIMEOUT_S),
        )
    return _clients[cache_key]


def default_model_id() -> str | None:
    for mid, cfg in MODELS.items():
        if cfg["_api_key"]:
            return mid
    return None


# ── MCP Streamable HTTP client ────────────────────────────────────────────

_mcp_sessions: dict[str, str] = {}
# One JSON-RPC id sequence per process: parallel calls on the same MCP
# session must not share an id, or their responses can be mixed up.
_rpc_id = itertools.count(1)
_MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def _parse_sse(text: str):
    """Extract the last JSON-RPC result from an SSE stream."""
    result = {}
    for line in text.splitlines():
        if line.startswith("data:"):
            try:
                result = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                pass
    return result


async def _mcp_post(client: httpx.AsyncClient, url: str, payload: dict):
    """POST a JSON-RPC message to an MCP server; parse JSON or SSE."""
    headers = dict(_MCP_HEADERS)
    headers[REQUEST_ID_HEADER] = get_request_id()
    sid = _mcp_sessions.get(url)
    if sid:
        headers["Mcp-Session-Id"] = sid
    r = await client.post(url, json=payload, headers=headers)
    r.raise_for_status()
    new_sid = r.headers.get("mcp-session-id")
    if new_sid:
        _mcp_sessions[url] = new_sid
    ct = r.headers.get("content-type", "")
    if "text/event-stream" in ct:
        return _parse_sse(r.text)
    if r.text.strip():
        return r.json()
    return {}


async def _mcp_ensure_init(client: httpx.AsyncClient, url: str):
    """Open the MCP session if there is not one already."""
    if url in _mcp_sessions:
        return
    await _mcp_post(client, url, {
        "jsonrpc": "2.0", "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "enviroo", "version": "1.0"},
        },
    })
    headers = dict(_MCP_HEADERS)
    headers[REQUEST_ID_HEADER] = get_request_id()
    sid = _mcp_sessions.get(url)
    if sid:
        headers["Mcp-Session-Id"] = sid
    await client.post(url, json={
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    }, headers=headers)


async def _mcp_call(base_url: str, tool_name: str, args: dict):
    """Call one MCP tool over the Streamable HTTP transport.

    If the MCP server has been restarted, the remembered session is stale and
    the first call fails; it is retried once with a fresh session so that a
    server restart is invisible to the user.
    """
    stage = f"mcp.call.{tool_name}"
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with log_timing(log, stage, server=base_url, args=args, attempt=attempt):
                async with httpx.AsyncClient(timeout=60) as c:
                    await _mcp_ensure_init(c, base_url)
                    data = await _mcp_post(c, base_url, {
                        "jsonrpc": "2.0", "id": next(_rpc_id),
                        "method": "tools/call",
                        "params": {"name": tool_name, "arguments": args},
                    })
            content = data.get("result", {}).get("content", [])
            if content and content[0].get("type") == "text":
                try:
                    return json.loads(content[0]["text"])
                except Exception:
                    return content[0]["text"]
            return data.get("result", {})
        except Exception as e:
            last_error = e
            _mcp_sessions.pop(base_url, None)
            if attempt == 0:
                log.warning("MCP call %s/%s: stale session or error (%s), retrying",
                            base_url, tool_name, e)
    log.error("MCP call %s/%s failed: %s", base_url, tool_name, last_error)
    return {"error": str(last_error).strip() or repr(last_error)}


# ── Dynamic tool discovery (tools/list) ───────────────────────────────────

TOOL_ROUTES: dict[str, str] = {}     # tool name -> MCP server URL
OPENAI_TOOLS: list[dict] = []        # definitions in OpenAI function-calling form
DISCOVERY_REPORT: dict[str, dict] = {}


def _mcp_tool_to_openai(tool: dict) -> dict:
    """Convert an MCP tool schema into the OpenAI function-calling format.

    Both use JSON Schema for parameters, so the conversion is direct. The
    description is capped at 1024 characters (the OpenAI limit) — worth
    watching here, since the descriptions are the apparatus.
    """
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": (tool.get("description") or "")[:1024],
            "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}},
        },
    }


async def _probe_server(url: str) -> list[str]:
    """Query one MCP server; return the names of its tools.

    Raises if the server does not answer — used when adding a server from the
    UI, so a wrong URL is rejected before it is persisted.
    """
    async with httpx.AsyncClient(timeout=10) as c:
        _mcp_sessions.pop(url, None)
        await _mcp_ensure_init(c, url)
        data = await _mcp_post(c, url, {
            "jsonrpc": "2.0", "id": next(_rpc_id),
            "method": "tools/list", "params": {},
        })
    tools = data.get("result", {}).get("tools", [])
    return [t["name"] for t in tools if t.get("name")]


# What the servers say they publish, and what the harness measured on arrival.
# Refreshed at every discovery; recorded with every benchmark row.
DESCRIPTION_REGIME: dict = {"regim": "rich", "chars": 0,
                            "guidance_runtime": "unknown"}


async def _descriptions_of(url: str) -> dict:
    """The description fingerprint a server declares, from its /health endpoint."""
    health = url.rstrip("/")
    health = health[:-len("/mcp")] + "/health" if health.endswith("/mcp") else health + "/health"
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            data = (await c.get(health)).json()
        return (data.get("descriptions") or {})
    except Exception as e:
        log.warning("could not read the description regime from %s: %s", health, e)
        return {}


async def _regime_of(url: str) -> str | None:
    """The description regime a server declares, from its /health endpoint."""
    return (await _descriptions_of(url)).get("regime")


async def discover_tools() -> dict:
    """Ask every registered server for its tools and rebuild the registry."""
    routes: dict[str, str] = {}
    defs: list[dict] = []
    report: dict[str, dict] = {}

    for url in _server_urls():
        try:
            with log_timing(log, "mcp.discover", server=url):
                async with httpx.AsyncClient(timeout=10) as c:
                    _mcp_sessions.pop(url, None)   # fresh session on re-discovery
                    await _mcp_ensure_init(c, url)
                    data = await _mcp_post(c, url, {
                        "jsonrpc": "2.0", "id": next(_rpc_id),
                        "method": "tools/list", "params": {},
                    })
            tools = data.get("result", {}).get("tools", [])
        except Exception as e:
            log.error("discovery failed for %s: %s", url, e)
            report[url] = {"status": f"error: {e}", "tools": []}
            continue

        report_regime = await _regime_of(url)
        names = []
        for t in tools:
            name = t.get("name")
            if not name:
                continue
            if name in routes:
                # Both description arms publish the same nine names on purpose:
                # the name is part of the apparatus. Registering both servers
                # therefore yields ONE arm and this warning, not two - and the
                # regime recorded with the run becomes `mixed`, marking the rows
                # for discard rather than analysis.
                log.error("duplicate tool '%s' on %s (kept the one on %s). If these are "
                          "the two description arms, register only one server at a time.",
                          name, url, routes[name])
                continue
            routes[name] = url
            defs.append(_mcp_tool_to_openai(t))
            names.append(name)
        report[url] = {"status": "ok", "tools": names, "regime": report_regime}
        log.info("discovered %d tools on %s: %s", len(names), url, names,
                 extra={"metric": {"stage": "mcp.discover.result", "server": url,
                                   "tools": names}})

    TOOL_ROUTES.clear();      TOOL_ROUTES.update(routes)
    OPENAI_TOOLS.clear();     OPENAI_TOOLS.extend(defs)
    DISCOVERY_REPORT.clear(); DISCOVERY_REPORT.update(report)

    # Measured from the schemas that arrived, not from configuration: this is
    # the size of the surface the models are actually shown.
    chars = sum(len(d["function"].get("description") or "") for d in defs)
    chars += sum(len(p.get("description", ""))
                 for d in defs
                 for p in (d["function"].get("parameters") or {}).get("properties", {}).values())
    fingerprints = [await _descriptions_of(u) for u in _server_urls()]
    declared = [f.get("regime") for f in fingerprints if f.get("regime")]
    regim = declared[0] if len(set(declared)) == 1 else ("mixed" if declared else "unknown")
    # Same rule as the regime: one value across the active servers, or `mixed`.
    # A mixed guidance layer invalidates a comparison exactly as a mixed
    # description regime does, so it has to be visible in the data.
    guidance = {str(f["guidance_runtime"]).lower()
                for f in fingerprints if "guidance_runtime" in f}
    guidance_runtime = (guidance.pop() if len(guidance) == 1
                        else ("mixed" if guidance else "unknown"))
    DESCRIPTION_REGIME.update(regim=regim, chars=chars,
                              guidance_runtime=guidance_runtime)
    log.info("description regime: %s (%d tools, %d chars of description)",
             regim, len(defs), chars,
             extra={"metric": {"stage": "mcp.discover", "event": "regime",
                               "regim": regim, "chars": chars, "tools": len(defs)}})
    log.info("tool registry: %d tools from %d servers", len(routes), len(MCP_SERVERS))
    return report


async def dispatch(tool_name: str, args: dict):
    """Route a call to whichever MCP server published that tool."""
    url = TOOL_ROUTES.get(tool_name)
    if not url:
        return {"error": f"Unknown tool: {tool_name}. Available: {sorted(TOOL_ROUTES)}"}
    return await _mcp_call(url, tool_name, args)


# Geometry is excluded from the grounding projection on purpose. A polygon
# carries hundreds of coordinates, and almost any figure in an answer would
# find a near match among them, so including it would make every claim look
# grounded and the check would certify nothing.
_GEOMETRY_KEYS = {"polygon", "service_area", "coordinates", "geometry", "bbox"}
_SHELTER_ID_RE = re.compile(r"^S\d{4,6}$", re.IGNORECASE)
# A street number reaches the answer through the address string, not as a
# number of its own: `Str. Regele Ferdinand 22-26` is a figure the tools
# supplied, and an answer that repeats it is quoting, not inventing. The
# snapshot id is excluded because its digits are a date, not a measurement.
_TEXT_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_OPAQUE_STRING_KEYS = {"snapshot_id", "shelter_id", "run_id"}


def _grounding_of(result) -> dict:
    """The scalars a later claim can be checked against.

    The tool responses are what an answer is entitled to assert. Keeping a
    compact projection of each one - its numbers, its shelter identifiers and
    the status of each shelter - is what lets the grader ask whether a figure
    in the answer was measured or invented. The full payload is deliberately
    not kept: it would put a polygon in every row of the results file.
    """
    numbers: set[float] = set()
    ids: set[str] = set()
    statuses: dict[str, str] = {}

    def walk(node, shelter):
        if isinstance(node, dict):
            own = node.get("shelter_id")
            own = own if isinstance(own, str) else shelter
            if isinstance(own, str):
                ids.add(own.upper())
            status = node.get("status")
            if isinstance(own, str) and isinstance(status, str):
                statuses[own.upper()] = status
            for key, value in node.items():
                if key in _GEOMETRY_KEYS:
                    continue
                if isinstance(value, str) and key not in _OPAQUE_STRING_KEYS:
                    numbers.update(float(m) for m in _TEXT_NUMBER_RE.findall(value))
                walk(value, own)
        elif isinstance(node, list):
            for value in node:
                walk(value, shelter)
        elif isinstance(node, bool):
            pass
        elif isinstance(node, (int, float)):
            numbers.add(float(node))
        elif isinstance(node, str) and _SHELTER_ID_RE.match(node.strip()):
            ids.add(node.strip().upper())

    walk(result, None)
    return {"numbers": sorted(numbers), "ids": sorted(ids), "statuses": statuses}


def _result_summary(result) -> dict:
    """Compact summary of a tool result, for the logs.

    The full content is never logged — a polygon would fill the file — only
    its size and the signal fields from which the run can be reconstructed.
    """
    if not isinstance(result, dict):
        s = str(result)
        return {"kind": "text", "chars": len(s), "preview": s[:150]}
    out: dict = {"kind": "json", "chars": len(json.dumps(result, ensure_ascii=False))}
    for key in ("error", "error_code", "detail", "note"):
        if result.get(key):
            out[key] = str(result[key])[:200]
    for key in ("count", "population", "walk_time_min", "distance_m", "area_km2",
                "demand_ratio", "population_served", "total_capacity_nominal",
                "total_capacity_effective", "snapshot_id", "match_level"):
        if key in result:
            out[key] = result[key]
    polygon = result.get("polygon") or result.get("service_area")
    if isinstance(polygon, dict):
        try:
            out["vertices"] = sum(len(ring) for ring in polygon.get("coordinates", []))
        except TypeError:
            pass
    return out


# ── FastAPI app ───────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    await discover_tools()
    yield


app = FastAPI(title="ENVIROO", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RequestIdMiddleware, logger_name="enviroo_web.http")


# ── The agentic loop (shared by chat and benchmark) ───────────────────────

# Reasoning models may return their text in a field outside the OpenAI schema.
# Reading it is diagnostic by default; USING it changes what is measured, so
# that is a switch and not a silent fallback.
REASONING_FALLBACK = os.getenv("ENVIROO_REASONING_FALLBACK", "").lower() in (
    "1", "true", "yes", "on")
_REASONING_FIELDS = ("reasoning_content", "reasoning")


def _reasoning_of(message) -> str:
    """The reasoning text a provider returned outside `content`, or "".

    Pydantic keeps unknown fields in `model_extra`; some SDK versions also
    surface them as attributes. Both are checked, in the order the providers
    use them.
    """
    extra = getattr(message, "model_extra", None) or {}
    for field in _REASONING_FIELDS:
        value = getattr(message, field, None) or extra.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


async def run_turn(cfg: dict, model_id: str, history: list[dict],
                   tools: list | None, emit=None, turn_no: int = 1) -> dict:
    """Run one complete turn and return everything that happened in it.

    `history` must end with the user's message; the assistant's reply is
    appended to it. `emit` receives the events destined for the UI; the
    benchmark passes None and reads the structured result instead.

    One function serves both the chat and the evaluation, deliberately: a
    separate runner would drift from the behaviour users actually get, and
    the study would end up measuring something nobody runs.
    """
    async def _emit(payload: dict):
        if emit is not None:
            await emit(payload)

    turn_start = time.perf_counter()
    # What the gateway actually served. `model` in models.json is what we ask
    # for; a provider resolves it to a dated snapshot, and OpenRouter routes an
    # open-weights model to one of several upstreams whose quantisation and
    # tool-call parsing differ. Asking for `gpt-5` and recording `gpt-5` says
    # nothing about which weights answered, so both are read off the response.
    provenance: dict = {"model_resolved": None, "provider_upstream": None}

    def _note_provenance(response) -> None:
        if provenance["model_resolved"] is None:
            provenance["model_resolved"] = getattr(response, "model", None)
        if provenance["provider_upstream"] is None:
            extra = getattr(response, "model_extra", None) or {}
            provenance["provider_upstream"] = (extra.get("provider")
                                               or getattr(response, "provider", None))

    tools_called: list[str] = []
    calls: list[dict] = []           # structured record, for the metrics
    rounds_detail: list[dict] = []
    total_tokens = prompt_tokens = completion_tokens = 0
    reasoning_chars = 0               # text returned outside `content`, if any
    overhead_context = None          # prompt_tokens at round 0 = fixed cost of the tools
    # The prompt follows the surface: a documented server is paired with the
    # documented prompt, a bare one with a bare prompt. Mixing them would make
    # the measured difference smaller than the difference under test.
    messages = [{"role": "system",
                 "content": _system_for(DESCRIPTION_REGIME.get("regim") or "rich")}] + history
    llm = client_for(cfg)
    rounds_used = 0
    forced = False

    try:
        for round_i in range(MAX_ROUNDS):
            rounds_used = round_i + 1
            round_detail: dict = {"round": round_i, "tools": []}
            rounds_detail.append(round_detail)
            with log_timing(log, "llm.completion", round=round_i, model=model_id,
                            messages_sent=len(messages), tools_offered=len(tools or [])):
                kwargs: dict = {
                    "model": cfg["model"],
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto" if tools else None,
                }
                # Anti-runaway ceiling: without it, a model that copies
                # geometry into its arguments generates tens of thousands of
                # tokens per round. The parameter name and the temperature
                # differ between families (the GPT-5 family requires
                # max_completion_tokens and rejects temperature), so both are
                # configurable per model in models.json.
                kwargs[cfg.get("token_param", "max_tokens")] = cfg.get("max_output_tokens", 4096)
                temperature = cfg.get("temperature", 0)
                if temperature is not None:
                    kwargs["temperature"] = temperature
                resp = await llm.chat.completions.create(**kwargs)
            _note_provenance(resp)
            if resp.usage:
                total_tokens += resp.usage.total_tokens
                prompt_tokens += resp.usage.prompt_tokens
                completion_tokens += resp.usage.completion_tokens
                if overhead_context is None:
                    overhead_context = resp.usage.prompt_tokens
                round_detail.update(
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                )
                log.info("║  round %d │ LLM: prompt=%d completion=%d total=%d tokens", round_i,
                         resp.usage.prompt_tokens, resp.usage.completion_tokens,
                         resp.usage.total_tokens,
                         extra={"metric": {
                             "stage": "llm.usage", "round": round_i, "model": model_id,
                             "prompt_tokens": resp.usage.prompt_tokens,
                             "completion_tokens": resp.usage.completion_tokens,
                             "total_tokens": resp.usage.total_tokens,
                         }})

            choice = resp.choices[0]
            asst = choice.message
            messages.append(asst.model_dump(exclude_none=True))

            reasoning = _reasoning_of(asst)
            if reasoning:
                reasoning_chars += len(reasoning)
                log.info("║  round %d │ reasoning outside content (%d chars): %s",
                         round_i, len(reasoning), reasoning[:200].replace("\n", " "),
                         extra={"metric": {"stage": "llm.reasoning", "round": round_i,
                                           "model": model_id, "chars": len(reasoning),
                                           "excerpt": reasoning[:1000]}})

            if asst.tool_calls:
                # Arguments come from the LLM and can be invalid JSON (a huge
                # geometry copied inline and cut off at the token limit). One
                # broken call does not kill the turn: it becomes an error
                # result the model can recover from in the next round.
                parsed_calls = []
                for tc in asst.tool_calls:
                    try:
                        parsed_calls.append((tc, json.loads(tc.function.arguments), None))
                    except json.JSONDecodeError as je:
                        log.warning("invalid arguments for %s: %s", tc.function.name, je)
                        parsed_calls.append((tc, None, {
                            "error": "The tool arguments are not valid JSON (most likely "
                                     "truncated). Do not copy geometry or long coordinate "
                                     "lists into arguments - geometric results are drawn "
                                     "on the map automatically.",
                        }))
                chosen = [{"tool": tc.function.name, "args": args if err is None else "<invalid>"}
                          for tc, args, err in parsed_calls]
                tools_called.extend(c["tool"] for c in chosen)
                round_detail["tools"] = [c["tool"] for c in chosen]
                for c in chosen:
                    log.info("║  round %d │ TOOL called: %s(%s)", round_i, c["tool"],
                             json.dumps(c["args"], ensure_ascii=False)[:400]
                             if c["args"] != "<invalid>" else "<invalid>",
                             extra={"metric": {"stage": "llm.tool_selection", "round": round_i,
                                               "model": model_id, "tool": c["tool"],
                                               "args": c["args"]}})

                results = await asyncio.gather(*[
                    dispatch(tc.function.name, args) if err is None else asyncio.sleep(0, result=err)
                    for tc, args, err in parsed_calls
                ])
                for (tc, args, err), result in zip(parsed_calls, results):
                    summary = _result_summary(result)
                    ok = not any(k in summary for k in ("error", "error_code", "detail"))
                    status = "ok" if ok else "ERROR"
                    calls.append({
                        "round": round_i,
                        "tool": tc.function.name,
                        "args": args if err is None else None,
                        "ok": ok,
                        "error": (summary.get("error") or summary.get("error_code")
                                  or summary.get("detail")) if not ok else None,
                        # What this call entitles the answer to claim.
                        "grounding": _grounding_of(result),
                    })
                    log.info("║  round %d │ TOOL result: %s -> %s %s", round_i,
                             tc.function.name, status,
                             json.dumps({k: v for k, v in summary.items() if k != "kind"},
                                        ensure_ascii=False)[:300],
                             extra={"metric": {"stage": "tool.result", "round": round_i,
                                               "tool": tc.function.name, **summary}})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    await _emit({"type": "tool_result", "tool": tc.function.name, "data": result})
            else:
                final = asst.content or ""
                if not final.strip() and reasoning:
                    # The distinction the metrics could not make before: the
                    # model produced text, in a field the client never shows.
                    log.warning(
                        "║  round %d │ EMPTY content but %d chars of reasoning "
                        "(fallback=%s)", round_i, len(reasoning), REASONING_FALLBACK,
                        extra={"metric": {"stage": "turn.answer", "turn": turn_no,
                                          "round": round_i, "answer_empty": True,
                                          "reasoning_chars": len(reasoning),
                                          "reasoning_used": REASONING_FALLBACK,
                                          "reasoning": reasoning[:2000]}})
                    if REASONING_FALLBACK:
                        final = reasoning
                history.append({"role": "assistant", "content": final})
                log.info("║  round %d │ FINAL ANSWER (%d chars): %s", round_i, len(final), final[:250],
                         extra={"metric": {"stage": "turn.answer", "turn": turn_no,
                                           "round": round_i, "answer": final[:1000]}})
                await _emit({"type": "message", "content": final})
                break
        else:
            # The round budget ran out mid tool loop. Without the step below
            # the turn would end with NO message at all; a synthesis without
            # tools is forced so that an answer - even a partial one - always
            # exists.
            forced = True
            messages.append({
                "role": "system",
                "content": "The tool-call budget is exhausted. Answer the user now from "
                           "what you have obtained so far; if the answer is incomplete, "
                           "say explicitly what you were unable to establish.",
            })
            kwargs = {"model": cfg["model"], "messages": messages}
            kwargs[cfg.get("token_param", "max_tokens")] = cfg.get("max_output_tokens", 4096)
            temperature = cfg.get("temperature", 0)
            if temperature is not None:
                kwargs["temperature"] = temperature
            with log_timing(log, "llm.completion", round=MAX_ROUNDS, model=model_id,
                            messages_sent=len(messages), tools_offered=0):
                resp = await llm.chat.completions.create(**kwargs)
            _note_provenance(resp)
            if resp.usage:
                total_tokens += resp.usage.total_tokens
                prompt_tokens += resp.usage.prompt_tokens
                completion_tokens += resp.usage.completion_tokens
                rounds_detail.append({"round": MAX_ROUNDS, "tools": ["(forced synthesis)"],
                                      "prompt_tokens": resp.usage.prompt_tokens,
                                      "completion_tokens": resp.usage.completion_tokens})
            final = resp.choices[0].message.content or (
                "I could not reach a complete answer within the available call budget."
            )
            history.append({"role": "assistant", "content": final})
            log.info("║  FORCED SYNTHESIS (budget of %d rounds exhausted, %d chars): %s",
                     MAX_ROUNDS, len(final), final[:250],
                     extra={"metric": {"stage": "turn.answer", "turn": turn_no,
                                       "round": MAX_ROUNDS, "forced": True,
                                       "answer": final[:1000]}})
            await _emit({"type": "message", "content": final})
    except Exception as e:
        log.error("╚ TURN %d │ FAILED (model=%s): %s", turn_no, model_id, e,
                  extra={"metric": {"stage": "turn", "event": "failed", "turn": turn_no,
                                    "model": model_id, "error": str(e)[:300],
                                    "rounds_detail": rounds_detail}})
        log.info("═" * 66)
        return {
            "answer": "", "calls": calls, "rounds": rounds_used, "forced": forced,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": total_tokens, "overhead_context": overhead_context,
            "duration_ms": round((time.perf_counter() - turn_start) * 1000, 1),
            "tools_called": tools_called, "error": str(e),
            "reasoning_chars": reasoning_chars, **provenance,
        }

    turn_duration_ms = round((time.perf_counter() - turn_start) * 1000, 1)
    breakdown = " │ ".join(
        f"r{d['round']}: {d.get('prompt_tokens', '?')}+{d.get('completion_tokens', '?')}tk"
        + (f" [{', '.join(d['tools'])}]" if d.get("tools") else " [answer]")
        for d in rounds_detail
    )
    log.info(
        "╚ TURN %d │ END: model=%s rounds=%d tokens_total=%d duration=%sms ║ %s",
        turn_no, model_id, rounds_used, total_tokens, turn_duration_ms, breakdown,
        extra={"metric": {
            "stage": "turn", "event": "end", "turn": turn_no,
            "duration_ms": turn_duration_ms,
            "model": model_id, "rounds": rounds_used, "tools_called": tools_called,
            "total_tokens": total_tokens, "rounds_detail": rounds_detail,
        }},
    )
    log.info("═" * 66)
    return {
        "answer": history[-1]["content"] if history and history[-1]["role"] == "assistant" else "",
        "calls": calls, "rounds": rounds_used, "forced": forced,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "total_tokens": total_tokens, "overhead_context": overhead_context,
        "duration_ms": turn_duration_ms, "tools_called": tools_called, "error": None,
        "reasoning_chars": reasoning_chars, **provenance,
    }


def _tools_for(enabled: list[str] | None) -> list | None:
    """No key => every tool (a server that appeared meanwhile stays active by
    default). An empty list => no tools at all (the model answers directly),
    which is how the no-tool control condition is run."""
    if enabled is None:
        return OPENAI_TOOLS or None
    allowed = set(enabled)
    return [t for t in OPENAI_TOOLS if t["function"]["name"] in allowed] or None


# ── WebSocket chat ────────────────────────────────────────────────────────

@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    history: list[dict] = []
    turn_no = 0
    log.info("WS connected")
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            text = msg.get("message", "").strip()
            if not text:
                continue

            model_id = msg.get("model") or default_model_id()
            cfg = MODELS.get(model_id)
            if not cfg or not cfg["_api_key"]:
                await ws.send_text(json.dumps({
                    "type": "error",
                    "content": f"Model '{model_id}' is not available (missing API key).",
                }))
                continue

            new_request_id()   # new request_id, to correlate this turn's logs
            turn_no += 1

            # An area drawn on the map becomes context for the model: a
            # rectangle is five coordinates, so inlining it is cheap and
            # avoids any pass-by-reference machinery.
            selection = msg.get("map_selection")
            user_content = text
            if isinstance(selection, dict):
                selection_json = json.dumps(selection, ensure_ascii=False)
                if len(selection_json) <= 4000:
                    user_content += (
                        "\n\n[Context: the user has an area selected on the map. "
                        f"GeoJSON WGS84: {selection_json}. If the question refers to "
                        "\"the selected area\", use this geometry as the polygon "
                        "argument of the area tools.]"
                    )
                else:
                    log.warning("map_selection too large (%d chars), ignored", len(selection_json))

            history.append({"role": "user", "content": user_content})
            del history[:-MAX_HISTORY]
            await ws.send_text(json.dumps({"type": "typing"}))

            tools = _tools_for(msg.get("enabled_tools"))
            log.info("═" * 66)
            log.info(
                "╔ TURN %d │ USER: %s", turn_no, text,
                extra={"metric": {
                    "stage": "turn", "event": "start", "turn": turn_no,
                    "user_message": text, "model": model_id,
                    "history_messages": len(history), "tools_active": len(tools or []),
                    "map_selection": bool(selection),
                }},
            )

            async def emit(payload: dict):
                await ws.send_text(json.dumps(payload))

            result = await run_turn(cfg, model_id, history, tools, emit=emit, turn_no=turn_no)
            if result["error"]:
                # An LLM provider error (invalid key, model unavailable...):
                # tell the user, but keep the connection and the history.
                await ws.send_text(json.dumps({"type": "error", "content": result["error"]}))

    except WebSocketDisconnect:
        log.info("WS disconnected")
    except Exception as e:
        log.error("WS error: %s", e)
        try:
            await ws.send_text(json.dumps({"type": "error", "content": str(e)}))
        except Exception:
            pass


# ── API ───────────────────────────────────────────────────────────────────

@app.get("/api/models")
async def api_models():
    """The models declared in models.json; `available` = its key is in .env."""
    return {
        "default": default_model_id(),
        "models": [
            {"id": mid, "label": cfg.get("label", mid), "available": bool(cfg["_api_key"])}
            for mid, cfg in MODELS.items()
        ],
    }


@app.get("/api/tools")
async def api_tools():
    """The current registry, grouped by MCP server, with tool descriptions.

    Feeds the UI panel where the researcher ticks which tools are exposed to
    the model — the mechanism behind ablation runs.
    """
    known = {s["url"]: s for s in MCP_SERVERS}
    descs = {t["function"]["name"]: t["function"]["description"] for t in OPENAI_TOOLS}
    servers = []
    seen = set()
    for url, info in DISCOVERY_REPORT.items():
        seen.add(url)
        srv = known.get(url, {})
        servers.append({
            "url": url,
            "label": srv.get("label", url),
            "enabled": bool(srv.get("enabled", True)),
            "regime": info.get("regime"),
            "status": info.get("status", "unknown"),
            "tools": [{"name": n, "description": descs.get(n, "")} for n in info.get("tools", [])],
        })
    # A disabled server was never queried, so discovery says nothing about it.
    # It still belongs in the panel: that is the point of keeping it.
    for url, srv in known.items():
        if url in seen:
            continue
        servers.append({
            "url": url, "label": srv.get("label", url), "enabled": False,
            "regime": None, "status": "disabled", "tools": [],
        })
    return {"servers": servers, "total_tools": len(TOOL_ROUTES),
            "descriptions": DESCRIPTION_REGIME}


@app.post("/api/mcp/toggle")
async def api_mcp_toggle(payload: dict):
    """Switch a registered server on or off, then re-discover.

    Kept separate from add/remove because the two arms of the description
    experiment are swapped often and losing a URL to a stray click would be
    a needless annoyance.
    """
    url = (payload.get("url") or "").strip()
    enabled = bool(payload.get("enabled"))
    for srv in MCP_SERVERS:
        if srv["url"] == url:
            srv["enabled"] = enabled
            save_servers()
            report = await discover_tools()
            log.info("server %s %s", url, "enabled" if enabled else "disabled")
            return {"servers": report, "registry": MCP_SERVERS,
                    "total_tools": len(TOOL_ROUTES),
                    "descriptions": DESCRIPTION_REGIME}
    raise HTTPException(404, f"Not a registered server: {url}")


@app.get("/api/mcp/servers")
async def api_mcp_list():
    """Registered MCP servers (from mcp_servers.json)."""
    return {"servers": MCP_SERVERS, "report": DISCOVERY_REPORT, "total_tools": len(TOOL_ROUTES)}


@app.post("/api/mcp/servers")
async def api_mcp_add(payload: dict):
    """Add an MCP server from the UI: validate, probe, persist, rescan.

    The server is accepted only if it answers `tools/list`, so a wrong URL
    never reaches the registry.
    """
    url = (payload.get("url") or "").strip()
    label = (payload.get("label") or "").strip()
    try:
        _validate_server_url(url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if any(s["url"] == url for s in MCP_SERVERS):
        raise HTTPException(status_code=409, detail="Server already registered")
    try:
        names = await _probe_server(url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"The MCP server does not answer tools/list: {e}")
    if not names:
        raise HTTPException(status_code=502, detail="The server exposes no tools")
    MCP_SERVERS.append({"url": url, "label": label or _derive_label(url)})
    save_servers()
    report = await discover_tools()
    log.info("MCP server added: %s (%d tools)", url, len(names))
    return {"added": url, "tools": names, "servers": report, "total_tools": len(TOOL_ROUTES)}


@app.delete("/api/mcp/servers")
async def api_mcp_remove(url: str):
    """Remove a registered MCP server and its tools from the registry."""
    before = len(MCP_SERVERS)
    MCP_SERVERS[:] = [s for s in MCP_SERVERS if s["url"] != url]
    if len(MCP_SERVERS) == before:
        raise HTTPException(status_code=404, detail=f"Server not found: {url}")
    _mcp_sessions.pop(url, None)
    save_servers()
    report = await discover_tools()
    log.info("MCP server removed: %s", url)
    return {"removed": url, "servers": report, "total_tools": len(TOOL_ROUTES)}


@app.post("/api/mcp/refresh")
async def api_mcp_refresh():
    """Re-run discovery on the registered servers, without a restart."""
    report = await discover_tools()
    return {"servers": report, "total_tools": len(TOOL_ROUTES)}


# ── Benchmark ─────────────────────────────────────────────────────────────

GOLDEN_FILE = os.path.join(_ROOT, "benchmark", "golden_chains.json")


def load_tasks() -> dict:
    """Tasks from golden_chains.json, reloaded on every request so that
    editing the file does not require restarting the server."""
    try:
        with open(GOLDEN_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        log.error("golden_chains.json is invalid: %s", e)
        return {}
    return {t["id"]: t for t in data.get("tasks", [])}


@app.get("/api/bench/tasks")
async def api_bench_tasks():
    """The task list for the selector in the UI."""
    tasks = load_tasks()
    return {"tasks": [{
        "id": t["id"],
        "difficulty": t.get("difficulty"),
        "title": t.get("title"),
        "prompt": t.get("prompt"),
        "prompt_ro": t.get("prompt_ro"),
        "servers": t.get("servers", []),
        "golden_steps": t.get("golden_steps"),
        "requires": t.get("requires", []),
        "pitfalls": t.get("pitfalls", []),
        # The ideal chain and the answer it should produce. Shown in the panel
        # before a run, so the researcher reads the reference first and the
        # model's behaviour second - not the other way round.
        "golden_chain": [{"step": c.get("step"), "tool": c.get("tool"),
                          "purpose": c.get("purpose")}
                         for c in t.get("golden_chain", [])],
        "expected_answer": t.get("expected_answer"),
        "capabilities": t.get("capabilities", []),
        "notes": t.get("notes"),
    } for t in tasks.values()]}


@app.post("/api/bench/run")
async def api_bench_run(payload: dict):
    """Run one task with one model, score it, append a line to metrics.csv."""
    from app.benchmark import scorer, store

    task_id = (payload.get("task_id") or "").strip()
    model_id = (payload.get("model") or default_model_id() or "").strip()
    tasks = load_tasks()
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, f"Unknown task: {task_id}")
    regim = DESCRIPTION_REGIME.get("regim") or "rich"
    cfg = MODELS.get(model_id)
    if not cfg or not cfg["_api_key"]:
        raise HTTPException(400, f"Model '{model_id}' is not available (missing API key).")

    repetition = payload.get("repetition") or store.next_repetition(task_id, model_id, regim)
    # The regime is in the identifier as well as in its own column: a file full
    # of run ids is the first thing anyone reads, and two arms that look alike
    # there would be mixed by hand long before they were mixed by code.
    tag = "" if regim == "rich" else f"__{regim[:3]}"
    run_id = f"{task_id}__{model_id}{tag}__r{repetition}"
    set_request_id(run_id)   # ties the logs of EVERY server to this run

    # Empty history: every run starts clean, or repetitions would contaminate
    # each other and stop measuring the same thing.
    history: list[dict] = [{"role": "user", "content": task["prompt"]}]
    tools = _tools_for(payload.get("enabled_tools"))

    log.info("═" * 66)
    log.info("╔ BENCH %s │ model=%s rep=%d", task_id, model_id, repetition,
             extra={"metric": {"stage": "bench", "event": "start", "task": task_id,
                               "model": model_id, "repetition": repetition,
                               "tools_active": len(tools or [])}})

    run = await run_turn(cfg, model_id, history, tools, emit=None, turn_no=1)
    metrics = scorer.score_run(task, run,
                               price_in=cfg.get("price_in"), price_out=cfg.get("price_out"))

    row = {
        "ts": store.now_iso(), "run_id": run_id, "task_id": task_id,
        "difficulty": task.get("difficulty"), "title": task.get("title"),
        "servers": ",".join(task.get("servers", [])),
        "model_id": model_id, "model_label": cfg.get("label", model_id),
        "repetition": repetition, "manual_verdict": "",
        "descriptions_regime": regim,
        "descriptions_chars": DESCRIPTION_REGIME.get("chars") or 0,
        "prompt_regime": config.SYSTEM_PROMPT_REGIME,
        "prompt_sha": prompt_fingerprint(regim),
        # Read off the response, not off the configuration: see run_turn.
        "model_route": cfg.get("_route") or config.MODEL_ROUTE,
        "model_resolved": run.get("model_resolved") or "",
        "provider_upstream": run.get("provider_upstream") or "",
        # `not_sent` is a finding, not a gap: the parameter is deliberately
        # omitted, so the run keeps the provider's own default sampling. The
        # GPT-5 family rejects `temperature` outright.
        "temperature": ("not_sent" if cfg.get("temperature") is None
                        else cfg["temperature"]),
        "top_p": "not_sent" if cfg.get("top_p") is None else cfg["top_p"],
        "max_output_tokens": cfg.get("max_output_tokens", 4096),
        "token_param": cfg.get("token_param", "max_tokens"),
        "tool_choice": "auto" if tools else "none",
        "max_rounds": MAX_ROUNDS,
        "guidance_runtime": DESCRIPTION_REGIME.get("guidance_runtime") or "unknown",
        "run_error": run.get("error") or "",
        "answer": (run.get("answer") or "")[:2000],
        "note": "forced synthesis" if run.get("forced") else "",
        **{k: v for k, v in metrics.items() if k != "assertion_details"},
    }
    # Serialised last, over the spread: the scorer hands this back as a dict
    # and the CSV needs the claims themselves, not "{'numbers_total': ...}".
    row["grounding_details"] = json.dumps(metrics.get("grounding_details") or {},
                                          ensure_ascii=False)[:2000]
    csv_path = store.append_result(row)

    # `%.2f` on a metric that is legitimately None raises inside the logging
    # handler, which loses the whole structured record for the run - the CSV
    # row survives but the log line does not, and the traceback drowns the
    # console for the rest of a sweep. M1 and M3 are undefined when a turn
    # made no relevant calls, and M4 when nothing was checkable, so the
    # formatting has to expect it.
    def _num(value: float | None) -> str:
        return "—" if value is None else f"{value:.2f}"

    log.info("╚ BENCH %s │ success=%s f1_tool=%s param=%s tokens=%d",
             task_id, metrics["strict_success"], _num(metrics["m3_tool_f1"]),
             _num(metrics["m4_param_accuracy"]), metrics["m12_total_tokens"] or 0,
             extra={"metric": {"stage": "bench", "event": "end", "task": task_id,
                               "model": model_id, "repetition": repetition, **{
                                   k: v for k, v in metrics.items()
                                   if k not in ("assertion_details", "param_errors")}}})
    log.info("═" * 66)

    return {
        "run_id": run_id, "task": {"id": task_id, "title": task.get("title"),
                                   "difficulty": task.get("difficulty"),
                                   "prompt": task.get("prompt"),
                                   "golden_chain": [{"step": s["step"], "tool": s["tool"],
                                                     "server": s.get("server")}
                                                    for s in task.get("golden_chain", [])]},
        "model": {"id": model_id, "label": cfg.get("label", model_id)},
        "repetition": repetition, "metrics": metrics,
        "answer": run.get("answer"), "calls": run.get("calls"),
        "error": run.get("error"), "csv": os.path.relpath(csv_path, _ROOT),
    }


@app.get("/api/bench/results")
async def api_bench_results(limit: int = 25):
    """The latest runs plus the per-model aggregate, for the UI table."""
    from app.benchmark import store
    return {"rulari": store.read_results(limit), "sumar": store.summary_by_model()}


@app.get("/api/bench/csv")
async def api_bench_csv():
    """Download metrics.csv exactly as it is on disk."""
    from app.benchmark import store
    if not os.path.exists(store.RESULTS_CSV):
        raise HTTPException(404, "There are no runs yet.")
    return FileResponse(store.RESULTS_CSV, media_type="text/csv", filename="metrics.csv")


@app.post("/api/bench/save")
async def api_bench_save(payload: dict):
    """Save an annotated observation (verdict + note) into observatii.csv.

    Also updates the line in metrics.csv, so the raw log and the annotated set
    cannot contradict each other.
    """
    from app.benchmark import store
    run_id = (payload.get("run_id") or "").strip()
    verdict = payload.get("verdict", "") or ""
    note = payload.get("note", "") or ""
    # A second reader annotates from their own browser without touching the
    # environment: the identity travels with the observation.
    evaluator = (payload.get("evaluator") or config.EVALUATOR).strip()
    if not run_id:
        raise HTTPException(400, "run_id missing")

    path = store.append_observation(run_id, verdict, note, evaluator)
    if path is None:
        raise HTTPException(404, f"Run not found in metrics.csv: {run_id}")

    # Reported, never assumed. The verdict lands in two files, and the second
    # write can fail on its own - a run deleted from metrics.csv after it was
    # observed, for instance. Saying "saved" when only one of the two happened
    # is how an annotation goes missing without anyone noticing.
    metrics_updated = store.set_manual_verdict(run_id, verdict, note, evaluator)

    log.info("observation saved: %s verdict=%s note=%d chars metrics=%s",
             run_id, verdict or "-", len(note), metrics_updated,
             extra={"metric": {"stage": "bench", "event": "observation",
                               "run_id": run_id, "verdict": verdict,
                               "metrics_updated": metrics_updated}})
    if not metrics_updated:
        log.warning("verdict for %s was NOT written into metrics.csv "
                    "(run absent from the file)", run_id)
    return {"run_id": run_id, "verdict": verdict,
            "fisier": os.path.relpath(path, _ROOT),
            "metrics_updated": metrics_updated,
            "total_observatii": store.count_observations()}


@app.get("/api/bench/observations")
async def api_bench_observations():
    """Download observatii.csv — the manually annotated set."""
    from app.benchmark import store
    if not os.path.exists(store.OBSERVATIONS_CSV):
        raise HTTPException(404, "No observation has been saved yet.")
    return FileResponse(store.OBSERVATIONS_CSV, media_type="text/csv",
                        filename="observatii.csv")


@app.get("/api/map/shelters")
async def api_map_shelters(ids: str = ""):
    """Proxy the map geometry endpoint of the shelter MCP servers.

    The frontend cannot reach an MCP server directly (nor should it), and the
    tool responses carry no coordinates by design, so the map asks here and
    the web server asks whichever registered server can answer.
    """
    features: list[dict] = []
    for url in _server_urls():
        base = url.rsplit("/mcp", 1)[0]
        try:
            async with httpx.AsyncClient(timeout=8) as c:
                r = await c.get(f"{base}/map/shelters", params={"ids": ids})
            if r.status_code == 200:
                features.extend((r.json() or {}).get("features", []))
        except Exception:
            continue
    return {"type": "FeatureCollection", "features": features}


@app.get("/api/health")
async def health():
    status = {}
    for url in _server_urls():
        try:
            health_url = url.rsplit("/mcp", 1)[0] + "/health"
            async with httpx.AsyncClient(timeout=4) as c:
                r = await c.get(health_url)
            status[url] = r.json() if r.status_code == 200 else f"http_{r.status_code}"
        except Exception as e:
            status[url] = f"error: {e}"
    return {"status": "ok", "servers": status, "tools_discovered": len(TOOL_ROUTES)}


# ── Static files + index ──────────────────────────────────────────────────

_frontend = os.path.join(_ROOT, "frontend")
if os.path.isdir(_frontend):
    app.mount("/static", StaticFiles(directory=_frontend), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(_frontend, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.web.web_server:app",
        host="0.0.0.0",
        port=int(os.getenv("ENVIROO_WEB_PORT", "8090")),
        reload=False,
    )
