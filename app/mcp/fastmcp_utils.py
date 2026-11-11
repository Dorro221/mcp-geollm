"""Shared helpers for the native FastMCP servers."""
import inspect

from mcp.server.fastmcp import FastMCP


def dedent_tool_descriptions(mcp: FastMCP) -> None:
    """Normalize the indentation of tool descriptions published by FastMCP.

    FastMCP publishes ``fn.__doc__`` verbatim, so continuation lines of the
    docstrings keep their source-code indentation. ``cleandoc`` strips it
    while preserving the text exactly as written.
    """
    for tool in mcp._tool_manager.list_tools():
        if tool.description:
            tool.description = inspect.cleandoc(tool.description)


def apply_descriptions(mcp: FastMCP, spec: dict) -> int:
    """Overwrite the published descriptions with `spec`, in place.

    Rewrites both the tool description and the per-parameter descriptions in
    the JSON Schema, because the parameter text carries as much of the
    apparatus as the tool text does - the mapping from "an elderly person" to
    `reduced_mobility` lives there.

    Fails loudly on any drift between `spec` and the published surface: a
    silently unmatched tool would leave one description rich while the rest
    went minimal, and the arm would be measuring a mixture.
    """
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    missing = set(tools) - set(spec)
    extra = set(spec) - set(tools)
    if missing or extra:
        raise RuntimeError(
            f"description set does not match the published surface: "
            f"no entry for {sorted(missing)}, unknown entry for {sorted(extra)}")

    for name, entry in spec.items():
        tool = tools[name]
        tool.description = entry["description"]
        props = (tool.parameters or {}).get("properties", {})
        for param, text in entry.get("params", {}).items():
            if param not in props:
                raise RuntimeError(f"{name}: no parameter '{param}' to describe")
            props[param]["description"] = text
        # An empty string is a description: it is what the `bare` set publishes.
        # What must never happen is a parameter the set forgot, which would
        # leave the previous regime's text in place and mix the arms.
        forgotten = [p for p in props if p not in entry.get("params", {})]
        if forgotten:
            raise RuntimeError(f"{name}: parameters missing from the set: {forgotten}")
    return len(spec)
