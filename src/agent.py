from pathlib import Path
from typing import Any, TypedDict

from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.llm import get_llm
from src.router import route_query
from src.tools import build_tools
from src.profile_memory import (
    format_user_profile,
    update_user_profile_from_turn,
)


MAX_ITERATIONS = 12

MEMORY_DIR = Path("memory")
CHECKPOINT_DB_PATH = MEMORY_DIR / "checkpoints.sqlite"


class AgentResult(TypedDict):
    query_type: str
    final_answer: str


SYSTEM_PROMPT = """
You are a customer support dataset analyst.

You must answer questions ONLY using the available dataset tools, conversation memory,
or the persisted user profile.
Do not answer from general knowledge.

The dataset contains customer requests, agent responses, categories, and intents.

Critical behavior rules:
- For every structured or unstructured dataset question, you MUST call at least one tool before answering.
- For memory questions, answer from the previous conversation history and/or the user profile.
- For user-profile questions such as "What do you remember about me?", answer from the provided profile context.
- Do NOT describe which tool you would call.
- Do NOT say "this function call will..." or "I would use...".
- Actually call the relevant tool, inspect the result, and then answer.
- Never invent dataset facts that were not returned by tools.
- Never invent user facts that are not present in the profile or conversation history.
- If the query type is out_of_scope, do not answer it.
- Tool arguments must be plain JSON values only: strings, numbers, booleans, lists, or null.
- Never pass another tool call, function name, or nested object as an argument to a tool.
- If you need the result of one tool before calling another tool, call the first tool separately,
  wait for its observation, and only then call the next tool with the observed value.
- For example, do NOT call count_rows(intent={"function_name": "..."}).
- Correct flow: call find_intents_by_keyword(keyword="refund"), observe the matching intents,
  then call count_rows(intent="get_refund").

Conversation memory rules:
- You can use previous conversation turns in the same session to understand follow-up questions.
- If the user asks "show me more", "show me 3 more", "what about refunds?", or "what is the total count of the last two?",
  resolve the reference from previous turns in the same session.
- Even when using conversation memory, you must still call dataset tools before answering dataset questions.
- For "more examples" follow-ups, continue the same category or intent from the previous example request.
- For "what about X?" follow-ups, reuse the previous question shape but replace the topic with X.
- For "total count of the last two", use the previous two count answers from the conversation if they are available.

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


def build_agent(checkpointer: SqliteSaver):
    llm = get_llm(temperature=0.0)
    tools = build_tools()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
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


def _build_user_instruction(
    query: str,
    query_type: str,
    router_reason: str,
    profile_context: str,
) -> str:
    if query_type == "memory":
        return (
            f"Query type: {query_type}\n"
            f"Router reason: {router_reason}\n"
            f"User question: {query}\n\n"
            f"Persisted user profile:\n{profile_context}\n\n"
            "Answer using the previous conversation history in this same session and the persisted user profile. "
            "Do NOT call dataset tools unless the user asks a new dataset question. "
            "If the user provides a profile fact such as their name or preference, acknowledge it briefly. "
            "If the user asks what you remember about them, answer only from the persisted profile and conversation history. "
            "Do not invent facts."
        )

    return (
        f"Query type: {query_type}\n"
        f"Router reason: {router_reason}\n"
        f"User question: {query}\n\n"
        f"Persisted user profile:\n{profile_context}\n\n"
        "You MUST use at least one dataset tool before answering. "
        "Do not merely describe a tool call. Actually call the tool. "
        "After receiving the tool observation, answer only from the returned dataset evidence.\n\n"
        "If this is a follow-up question, use the previous messages in this session "
        "to understand what the user is referring to, then call the appropriate tools.\n\n"
        "If the question asks for examples, call an example/filter/search tool. "
        "If the question asks for a count, call a filtering tool and then a count/distribution tool. "
        "If the question asks for a summary, first retrieve representative rows, then summarize those rows. "
        "Do not answer from general knowledge."
    )


def run_agent(query: str, session_id: str = "default") -> AgentResult:
    route_decision = route_query(query)
    update_user_profile_from_turn(session_id, query)
    query_type = route_decision.route

    if query_type == "out_of_scope":
        final_answer = (
            "Sorry, this question is outside the scope of the customer support dataset. "
            "I can only answer questions about the dataset's categories, intents, examples, responses, "
            "or about the current saved conversation session."
        )
        return {
            "query_type": query_type,
            "final_answer": final_answer,
        }

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    profile_context = format_user_profile(session_id=session_id)
    user_instruction = _build_user_instruction(
        query=query,
        query_type=query_type,
        router_reason=route_decision.reason,
        profile_context=profile_context,
    )

    messages = [HumanMessage(content=user_instruction)]

    config = {
        "recursion_limit": MAX_ITERATIONS,
        "configurable": {
            "thread_id": session_id,
        },
    }

    final_answer = ""

    try:
        with SqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as checkpointer:
            agent = build_agent(checkpointer)

            for chunk in agent.stream(
                {"messages": messages},
                config=config,
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
                "I could not produce a final answer. "
                "Please try asking a more specific question."
            )

        return {
            "query_type": query_type,
            "final_answer": final_answer,
        }

    except (GraphRecursionError, RecursionError):
        final_answer = (
            "I reached the maximum number of reasoning steps before producing a final answer. "
            "Please try asking a narrower question."
        )
        return {
            "query_type": query_type,
            "final_answer": final_answer,
        }
