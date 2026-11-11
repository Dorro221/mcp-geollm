"""The bare-description arm, as its own server.

Same nine tools, same implementations, same database as the other two arms —
only the published text differs, and here it is as thin as a callable surface
can be: the tool's own name restated as a sentence, and the JSON type of each
parameter. See app/mcp/descriptions_bare.py for where the line is drawn and
why the enum values survive.

As in server_minimal, the regime is set BEFORE `app.mcp.server` is imported,
because that module applies the description set at import time. Setting the
environment variable would not work: `config` calls `load_dotenv(override=True)`,
so the file beats the process environment.

Run:  python -m app.mcp.server_bare        (default port 9102)

The three arms publish the SAME nine tool names, because the name is part of
the apparatus. The harness registry is keyed by name and keeps the first
server that claims one, so exactly one arm may be enabled at a time — the
panel has a switch per server for precisely this.
"""
from __future__ import annotations

from app import config

# Both assignments happen before the server module is imported, and that
# ordering is the whole mechanism - do not move them below the import.
config.TOOL_DESCRIPTIONS = "bare"
config.MCP_PORT = config.MCP_PORT_BARE

from app.mcp.server import app  # noqa: E402  (import order is load-bearing)

__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.mcp.server_bare:app",
        host=config.MCP_HOST,
        port=config.MCP_PORT_BARE,
        reload=False,
    )
