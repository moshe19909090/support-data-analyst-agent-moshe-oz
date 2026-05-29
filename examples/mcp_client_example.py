"""Minimal FastMCP client that talks to the local support-data-analyst-tools server.

Run from the project root:

    python examples/mcp_client_example.py

It launches `python -m src.mcp_server` as a stdio subprocess, lists the
available tools, and calls `get_categories`.
"""

import asyncio
import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def main() -> None:
    # Spawn the server with the same interpreter so it uses this environment.
    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "src.mcp_server"],
        cwd=str(PROJECT_ROOT),
    )

    async with Client(transport) as client:
        tools = await client.list_tools()
        print("Available tools:", [tool.name for tool in tools])

        result = await client.call_tool("get_categories", {})
        print("Categories:", result.data)


if __name__ == "__main__":
    asyncio.run(main())
