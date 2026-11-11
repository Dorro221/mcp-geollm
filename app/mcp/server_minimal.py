"""The minimal-description arm, as its own server.

Same nine tools, same implementations, same database — only the published
text differs. It is a separate PROCESS rather than separate code: two copies
of the tool logic would drift the first time a query was fixed in one and not
the other, and the comparison would then be measuring the drift.

The regime is set here, before `app.mcp.server` is imported, because that
module applies the description set at import time. Setting the environment
variable would not work: `config` calls `load_dotenv(override=True)`, so the
file beats the process environment — which is the right default for secrets
and the wrong one for a run mode. Assigning the attribute is explicit and
cannot be silently overridden by whatever happens to be in `.env`.

Run:  python -m app.mcp.server_minimal        (default port 9101)

IMPORTANT — the two servers publish the SAME nine tool names, because the
name is part of the apparatus and renaming it would change what is measured.
The harness registry is keyed by name and keeps the first server that claims
one, so registering both at once does not give you both arms: it gives you
one arm and a warning. Register exactly one in `mcp_servers.json` at a time.
Runs made while both are registered are recorded with regime `mixed`, which
is a signal to discard them, not a third condition.
"""
from __future__ import annotations

from app import config

# Both assignments happen before the server module is imported, and that
# ordering is the whole mechanism - do not move them below the import.
config.TOOL_DESCRIPTIONS = "minimal"
config.MCP_PORT = config.MCP_PORT_MINIMAL

from app.mcp.server import app  # noqa: E402  (import order is load-bearing)

__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.mcp.server_minimal:app",
        host=config.MCP_HOST,
        port=config.MCP_PORT_MINIMAL,
        reload=False,
    )
