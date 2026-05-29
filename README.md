# Support Data Analyst Agent

An AI-powered customer support analytics agent for exploring, summarizing, and querying a customer service dataset using tool-based reasoning.

## Requirements

- Python 3.10+ (required for the MCP server via [FastMCP](https://gofastmcp.com))
- Python 3.9+ for the interactive agent (`main.py`)

## Install dependencies

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in a `.env` file when running the LangGraph agent.

## Run the interactive agent

```bash
python main.py
```

Optional session id for persisted memory:

```bash
python main.py --session my-session
```

## MCP server

The dataset tools in `src/tools.py` are also exposed as an MCP server named **support-data-analyst-tools** (via [FastMCP](https://gofastmcp.com)). The server reuses the existing tool functions directly (`get_categories`, `count_rows`, `get_examples`, `get_intent_distribution`); it does not duplicate dataset logic and is independent of the LangGraph agent.

### Requirements

FastMCP requires **Python 3.10+**. The project's main `.venv` may be on Python 3.9, so the MCP server uses a separate environment. Python 3.12 is recommended.

### Set up a dedicated environment

From the project root, create `.venv-mcp` with Python 3.12 (installed e.g. via Homebrew: `brew install python@3.12`):

```bash
python3.12 -m venv .venv-mcp
source .venv-mcp/bin/activate
pip install -r requirements.txt
```

### Start the server

From the project root (stdio transport, default):

```bash
python -m src.mcp_server
```

In stdio mode the process blocks silently, waiting for an MCP client to connect over stdin/stdout. This is expected and not a hang. Stop it with `Ctrl+C`, or connect a client (below).

### Run the client example

The example launches the server as a stdio subprocess, lists the tools, and calls `get_categories`:

```bash
python examples/mcp_client_example.py
```

### Connect a client and call a tool

Minimal client that spawns the local server over stdio (run from the project root):

```python
import asyncio
import sys

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


async def main() -> None:
    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "src.mcp_server"],
        cwd=".",
    )
    async with Client(transport) as client:
        result = await client.call_tool("get_categories", {})
        print(result.data)


asyncio.run(main())
```

This returns the category list from the Bitext support dataset, e.g. `['ACCOUNT', 'CANCEL', 'CONTACT', ...]`.

### Cursor / Claude Desktop config (stdio)

```json
{
  "mcpServers": {
    "support-data-analyst-tools": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/path/to/support-data-analyst-agent-moshe-oz"
    }
  }
}
```

Replace `cwd` with your local clone path.
