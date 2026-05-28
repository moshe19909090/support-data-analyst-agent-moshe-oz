from typing import Any, TypedDict

from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.llm import get_llm
from src.router import route_query
from src.tools import build_tools


MAX_ITERATIONS = 12


class AgentResult(TypedDict):
    query_type: str
    final_answer: str


SYSTEM_PROMPT = """
You are a customer support dataset analyst.

You must answer questions ONLY using the available dataset tools.
Do not answer from general knowledge.

The dataset contains customer requests, agent responses, categories, and intents.

Critical behavior rules:
- For every structured or unstructured dataset question, you MUST call at least one tool before answering.
- Do NOT describe which tool you would call.
- Do NOT say "this function call will..." or "I would use...".
- Actually call the relevant tool, inspect the result, and then answer.
- Never invent dataset facts that were not returned by tools.
- If the query type is out_of_scope, do not answer it.
- Tool arguments must be plain JSON values only: strings, numbers, booleans, lists, or null.
- Never pass another tool call, function name, or nested object as an argument to a tool.
- If you need the result of one tool before calling another tool, call the first tool separately,
  wait for its observation, and only then call the next tool with the observed value.
- For example, do NOT call count_rows(intent={"function_name": "..."}).
- Correct flow: call find_intents_by_keyword(keyword="refund"), observe the matching intents,
  then call count_rows(intent="get_refund").

Structured query rules:
- For category questions, use the category/listing tools.
- For count questions, use filtering/counting tools.
- For example questions, use example retrieval tools.
- For distribution questions, use distribution tools.

Unstructured query rules:
- First retrieve representative examples from the dataset.
- Then summarize only what appears in those examples.
- If the user asks about complaints, cancellation, refund, money back, feedback, or agent responses,
  search or filter the dataset first and summarize based only on the returned rows.

Refund / money-back rules:
- Treat "refund", "money back", "return my money", and similar phrases as refund-related dataset queries.
- Use refund-related intents or text search before answering.

Final answer style:
- Be concise.
- Mention the dataset evidence briefly.
- If the tools return no matching data, say that no matching rows were found.
"""


def build_agent():
    llm = get_llm(temperature=0.0)
    tools = build_tools()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )


def _shorten(value: Any, max_chars: int = 700) -> str:
    """
    Keep CLI observations readable.
    Tool outputs can be long, so we print a shortened version.
    """
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def _print_reasoning_step(message: Any) -> None:
    """
    Print tool calls and tool observations from LangGraph messages.
    This satisfies the CLI requirement to show reasoning/tool steps.
    """
    if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
        for tool_call in message.tool_calls:
            tool_name = tool_call.get("name", "unknown_tool")
            tool_args = tool_call.get("args", {})

            print("\nTool call:")
            print(f"- name: {tool_name}")
            print(f"- args: {tool_args}")

    if isinstance(message, ToolMessage):
        print("\nObservation:")
        print(_shorten(message.content))


def run_agent(query: str) -> AgentResult:
    route_decision = route_query(query)
    query_type = route_decision.route

    if query_type == "out_of_scope":
        return {
            "query_type": query_type,
            "final_answer": (
                "Sorry, this question is outside the scope of the customer support dataset. "
                "I can only answer questions about the dataset's categories, intents, examples, and responses."
            ),
        }

    agent = build_agent()

    messages = [
        HumanMessage(
            content=(
                f"Query type: {query_type}\n"
                f"Router reason: {route_decision.reason}\n"
                f"User question: {query}\n\n"
                "You MUST use at least one dataset tool before answering. "
                "Do not merely describe a tool call. Actually call the tool. "
                "After receiving the tool observation, answer only from the returned dataset evidence.\n\n"
                "If the question asks for examples, call an example/filter/search tool. "
                "If the question asks for a count, call a filtering tool and then a count/distribution tool. "
                "If the question asks for a summary, first retrieve representative rows, then summarize those rows. "
                "Do not answer from general knowledge."
            )
        )
    ]

    final_answer = ""

    try:
        for chunk in agent.stream(
            {"messages": messages},
            config={"recursion_limit": MAX_ITERATIONS},
            stream_mode="values",
        ):
            current_messages = chunk.get("messages", [])
            if not current_messages:
                continue

            latest_message = current_messages[-1]
            _print_reasoning_step(latest_message)

            if isinstance(latest_message, AIMessage) and latest_message.content:
                final_answer = latest_message.content

        if not final_answer:
            final_answer = (
                "I could not produce a final answer from the dataset. "
                "Please try asking a more specific dataset question."
            )

        return {
            "query_type": query_type,
            "final_answer": final_answer,
        }

    except (GraphRecursionError, RecursionError):
        return {
            "query_type": query_type,
            "final_answer": (
                "I reached the maximum number of reasoning steps before producing a final answer. "
                "Please try asking a narrower question about the dataset."
            ),
        }
