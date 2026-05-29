# Support Data Analyst Agent

An AI-powered customer support analytics agent for exploring, summarizing, and querying the Bitext customer support dataset using LangGraph, tool-based reasoning, persistent memory, and MCP tools.

The project was built for the **From AI Model to AI Product — Advanced Agents** assignment. It includes:

- **Task 1:** A LangGraph ReAct agent with a query router, structured tools, multi-step tool use, CLI reasoning output, and max-iteration fallback.
- **Task 2:** Persistent conversation memory using LangGraph SQLite checkpoints, session IDs, follow-up support, and a persistent user profile memory file.
- **Task 3:** A FastMCP server exposing selected dataset tools to external MCP clients.

---

## Architecture overview

```text
User CLI
  ↓
main.py
  ↓
router.py ── classifies query as structured / unstructured / memory / out_of_scope
  ↓
agent.py ── LangGraph ReAct agent + SQLite checkpointer
  ↓
tools.py ── dataset tools using Pydantic schemas
  ↓
data_loader.py ── loads Bitext dataset from Hugging Face
```

For MCP:

```text
MCP Client
  ↓
src/mcp_server.py
  ↓
selected tools from src/tools.py
  ↓
Bitext customer support dataset
```

The MCP server is independent from the LangGraph agent. It exposes direct dataset tools and does not call `run_agent`.

---

## Model choice

The agent uses an OpenAI-compatible chat model through `langchain-openai`.

In this implementation, the project is configured to work with **Nebius AI Studio** using:

```text
meta-llama/Llama-3.3-70B-Instruct
```

I chose this model because it is a strong instruction-following model, supports tool-calling style workflows through the OpenAI-compatible API, and is capable enough for routing, summarization, and multi-step reasoning over dataset tools.

The code reads model configuration from environment variables, so the same project can be adapted to another OpenAI-compatible provider.

---

## Tools defined

The agent tools are defined in `src/tools.py` using clear names, descriptions, and Pydantic input schemas.

Main tools:

- `get_categories` — returns all unique dataset categories.
- `get_intents` — returns all unique intents, optionally filtered by category.
- `find_intents_by_keyword` — finds intents matching a keyword such as `refund`.
- `count_rows` — counts dataset rows, optionally filtered by category and/or intent.
- `get_examples` — returns customer instruction / agent response examples.
- `search_instructions` — searches customer instructions by keyword or phrase.
- `get_intent_distribution` — returns intent counts within a category.
- `summarize_category_data` — returns representative rows for category summarization.

The MCP server exposes at least these tools:

- `get_categories`
- `count_rows`
- `get_examples`
- `get_intent_distribution`

---

## Requirements

- Python 3.12 is recommended for the full project, including the CLI agent and FastMCP server.
- Python 3.10+ is required by FastMCP, but the project was tested end-to-end with Python 3.12.
- Network access is needed on the first run to download the Bitext dataset from Hugging Face.
- An OpenAI-compatible API key is required for the LangGraph agent.

---

## Setup

Clone the repository and enter the project folder:

```bash
git clone <your-repo-url>
cd support-data-analyst-agent-moshe-oz
```

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The same `.venv` environment is used for both the interactive LangGraph CLI agent and the FastMCP server.

Create a `.env` file in the project root.

Example:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=your_openai_compatible_base_url_here
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
```

If your local `src/llm.py` uses slightly different variable names, keep the same values but match the names expected by that file.

---

## Run the interactive CLI agent

From the project root:

```bash
python main.py
```

The app loads the dataset and starts an interactive loop:

```text
Customer Support Dataset Agent
Type 'exit' or 'quit' to stop.

You:
```

Example questions:

```text
What categories exist in the dataset?
How many refund requests did we get?
Show me 5 examples of the SHIPPING category.
Summarize how agents respond to complaint intents.
Show me examples of people wanting their money back.
What is the distribution of intents in the ACCOUNT category?
What's the best CRM software for handling complaints?
Who is the president of France?
```

The CLI prints:

- the router decision,
- tool calls,
- tool observations,
- and the final answer.

---

## Conversation memory and sessions

The agent supports persistent conversation memory using LangGraph SQLite checkpoints.

Run with a session ID:

```bash
python main.py --session memory_test
```

Example memory flow:

```text
You: Show me 3 examples from the REFUND category
You: Show me 3 more
You: How many complaints did we get?
You: What about refunds?
You: What is the total count of the last two?
```

Restart the app with the same session ID:

```bash
python main.py --session memory_test
```

Then ask:

```text
What did we discuss in this session?
Show me 3 more
```

The same session ID restores the previous conversation context.

Runtime memory files are stored locally under `memory/` and are intentionally ignored by Git.

---

## User profile memory

The project also includes persistent user profile memory in `src/profile_memory.py`.

This is separate from conversation history. It stores distilled profile facts such as the user's name, preferences, and frequent topics.

Example:

```text
You: My name is Moshe.
You: What do you remember about me?
```

The profile is stored locally under `memory/profiles/` and is ignored by Git.

---

## Out-of-scope behavior

The router declines questions that are not answerable from the dataset or memory.

Examples:

```text
What's the best CRM software for handling complaints?
Who is the president of France?
Write me a poem about customer service.
```

The agent should politely refuse instead of answering from general model knowledge.

---

## MCP server

The dataset tools in `src/tools.py` are also exposed as an MCP server named **support-data-analyst-tools** using [FastMCP](https://gofastmcp.com).

The server reuses existing tool functions directly. It does not duplicate dataset logic and does not call the LangGraph agent.

### MCP setup

FastMCP requires Python 3.10+. This project was tested end-to-end with Python 3.12 using the same `.venv` environment created in the setup section.

If using Homebrew on macOS and Python 3.12 is not installed yet:

```bash
brew install python@3.12
```

Then use the existing project environment:

```bash
source .venv/bin/activate
python --version
pip install -r requirements.txt
```

### Start the MCP server

From the project root:

```bash
python -m src.mcp_server
```

The server uses stdio transport by default. In stdio mode, the process may appear to block silently while waiting for an MCP client. This is expected. Stop it with `Ctrl+C`.

### Run the MCP client example

In another terminal, or after stopping the server, run:

```bash
source .venv/bin/activate
python examples/mcp_client_example.py
```

The example launches the MCP server as a stdio subprocess, lists available tools, and calls `get_categories`.

Expected output includes something like:

```text
Available tools: ['get_categories', 'count_rows', 'get_examples', 'get_intent_distribution']
Categories: ['ACCOUNT', 'CANCEL', 'CONTACT', 'DELIVERY', 'FEEDBACK', 'INVOICE', 'ORDER', 'PAYMENT', 'REFUND', 'SHIPPING', 'SUBSCRIPTION']
```

A warning about Hugging Face authentication or a Python resource tracker warning may appear locally, but the important part is that the client lists tools and receives category data.

### Minimal MCP client example

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

### Cursor / Claude Desktop MCP config

Example stdio config:

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

Replace `cwd` with your local clone path. If your MCP client does not automatically use the project environment, set `command` to the full path of the Python executable inside `.venv/bin/python`.

---

## Suggested grading checklist

After setup, a grader can verify the project quickly with:

```bash
python main.py --session test_session
```

Try:

```text
What categories exist in the dataset?
How many refund requests did we get?
Show me 3 examples from the REFUND category
Show me 3 more
My name is Moshe.
What do you remember about me?
Who is the president of France?
```

Then verify MCP:

```bash
source .venv/bin/activate
python examples/mcp_client_example.py
```

---

## Notes

- The dataset is loaded from Hugging Face on demand.
- Local runtime files are ignored by Git:
  - `.venv/`
  - `.env`
  - `memory/`
  - `__pycache__/`
- The first dataset call may take a few seconds because it downloads or caches the dataset.
