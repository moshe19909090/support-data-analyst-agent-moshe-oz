import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.llm import get_llm


QueryType = Literal["structured", "unstructured", "memory", "out_of_scope"]


class RouteDecision(BaseModel):
    route: QueryType = Field(
        description="The query type: structured, unstructured, memory, or out_of_scope."
    )
    reason: str = Field(description="Short explanation for the routing decision.")


ROUTER_SYSTEM_PROMPT = """
You are a strict router for a customer support dataset analysis agent.

Classify the user's query into exactly one route:
- structured
- unstructured
- memory
- out_of_scope

Rules:

structured:
Use this when the user asks for concrete dataset facts, counts, examples, categories,
intents, distributions, filtering, rows from the dataset, or follow-up questions that continue
a previous dataset query.

Examples:
- What categories exist in the dataset?
- How many refund requests did we get?
- Show me 5 examples of the SHIPPING category.
- What is the distribution of intents in the ACCOUNT category?
- Show me 3 more.
- What about refunds?
- What is the total count of the last two?

unstructured:
Use this when the user asks to summarize, analyze, or explain patterns INSIDE the dataset.
The question must still be about the dataset/customer support interactions.

Examples:
- Summarize the FEEDBACK category.
- How do agents respond to cancellation requests?
- Show me examples of people wanting their money back.
- Summarize how agents respond to complaint intents.

memory:
Use this when the user asks about the current conversation, previous turns,
session history, what the assistant remembers about the user, or when the user shares
stable profile information such as their name, preferences, or recurring interests.

Examples:
- My name is Moshe.
- Call me Moshe.
- I prefer concise answers.
- I like working step by step.
- What did we discuss in this session?
- What do you remember about me?
- What was my previous question?
- What were the last two things we counted?
- Remind me what we discussed earlier.
- What did I ask you before?

out_of_scope:
Use this when the user asks for general knowledge, recommendations, poems, advice,
software recommendations, current events, or anything not answerable from the dataset
or conversation/profile memory.

Important:
- Questions about "best CRM software", "president", "Champions League", poems,
  business advice, or general customer service strategy are out_of_scope.
- Do not classify a query as unstructured just because it mentions customer service.
  It must ask about the dataset itself.
- Follow-up dataset questions like "show me 3 more", "what about refunds?", or
  "what is the total count of the last two?" are structured because they require dataset work plus conversation history.
- Questions about the conversation itself or the persistent user profile are memory, not out_of_scope.
- Explicit user profile statements like "my name is...", "call me...", "I prefer...", or "I like..." are memory.

Return ONLY valid JSON in this exact format:
{
  "route": "structured | unstructured | memory | out_of_scope",
  "reason": "short reason"
}
"""


def route_query(query: str) -> RouteDecision:
    llm = get_llm(temperature=0.0)

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    response = llm.invoke(messages)
    raw_content = response.content.strip()

    try:
        data = json.loads(raw_content)
        return RouteDecision(**data)
    except Exception:
        return fallback_route(query)


def fallback_route(query: str) -> RouteDecision:
    lowered = query.lower().strip()

    profile_statement_keywords = [
        "my name is",
        "call me",
        "i am ",
        "i'm ",
        "i prefer",
        "i like",
        "i want",
    ]

    memory_keywords = [
        "what did we discuss",
        "what do you remember",
        "remember about me",
        "previous question",
        "earlier",
        "this session",
        "conversation",
        "last question",
        "what did i ask",
        "remind me",
    ]

    out_of_scope_keywords = [
        "crm software",
        "president",
        "champions league",
        "poem",
        "best software",
        "recommend",
        "who won",
        "current events",
    ]

    structured_keywords = [
        "how many",
        "count",
        "categories",
        "category",
        "intents",
        "intent",
        "distribution",
        "show me",
        "examples",
        "rows",
        "what about",
        "more",
        "total count",
        "last two",
    ]

    unstructured_keywords = [
        "summarize",
        "summary",
        "how do",
        "typically respond",
        "patterns",
        "analyze",
        "explain",
        "people wanting",
        "money back",
        "complaint",
    ]

    if any(keyword in lowered for keyword in profile_statement_keywords):
        return RouteDecision(
            route="memory",
            reason="The query shares stable user profile information that should be remembered.",
        )

    if any(keyword in lowered for keyword in memory_keywords):
        return RouteDecision(
            route="memory",
            reason="The query asks about the current conversation or remembered session/profile context.",
        )

    if any(keyword in lowered for keyword in out_of_scope_keywords):
        return RouteDecision(
            route="out_of_scope",
            reason="The query asks for general knowledge or recommendations outside the dataset.",
        )

    if any(keyword in lowered for keyword in structured_keywords):
        return RouteDecision(
            route="structured",
            reason="The query asks for concrete dataset facts, counts, examples, distributions, or a dataset follow-up.",
        )

    if any(keyword in lowered for keyword in unstructured_keywords):
        return RouteDecision(
            route="unstructured",
            reason="The query asks for summarization or pattern analysis inside the dataset.",
        )

    return RouteDecision(
        route="out_of_scope",
        reason="The query could not be clearly answered from the customer support dataset or conversation/profile memory.",
    )
