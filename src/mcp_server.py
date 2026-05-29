"""FastMCP server exposing the dataset tools from src/tools.py.

The server reuses the existing tool functions directly and is independent of
the LangGraph agent (it never calls run_agent).

Running `python -m src.mcp_server` starts the server over stdio. In stdio mode
the process blocks silently waiting for an MCP client to connect over
stdin/stdout; this is expected and not a hang. Connect with an MCP client (see
examples/mcp_client_example.py) to call the tools.
"""

from fastmcp import FastMCP

from src.tools import (
    count_rows,
    get_categories,
    get_examples,
    get_intent_distribution,
)

mcp = FastMCP("support-data-analyst-tools")

mcp.tool(get_categories)
mcp.tool(count_rows)
mcp.tool(get_examples)
mcp.tool(get_intent_distribution)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
