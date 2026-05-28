import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.llm import get_llm


QueryType = Literal["structured", "unstructured", "out_of_scope"]


class RouteDecision(BaseModel):
    route: QueryType = Field(
        description="The query type: structured, unstructured, or out_of_scope."
    )
    reason: str = Field(description="Short explanation for the routing decision.")


ROUTER_SYSTEM_PROMPT = """
You are a strict router for a customer support dataset analysis agent.

Classify the user's query into exactly one route:
- structured
- unstructured
- out_of_scope

Rules:

structured:
Use this when the user asks for concrete dataset facts, counts, examples, categories,
intents, distributions, filtering, or rows from the dataset.

Examples:
- What categories exist in the dataset?
- How many refund requests did we get?
- Show me 5 examples of the SHIPPING category.
- What is the distribution of intents in the ACCOUNT category?

unstructured:
Use this when the user asks to summarize, analyze, or explain patterns INSIDE the dataset.
The question must still be about the dataset/customer support interactions.

Examples:
- Summarize the FEEDBACK category.
- How do agents respond to cancellation requests?
- Show me examples of people wanting their money back.
- Summarize how agents respond to complaint intents.

out_of_scope:
Use this when the user asks for general knowledge, recommendations, poems, advice,
software recommendations, current events, or anything not answerable from the dataset.

Important:
- Questions about "best CRM software", "president", "Champions League", poems,
  business advice, or general customer service strategy are out_of_scope.
- Do not classify a query as unstructured just because it mentions customer service.
  It must ask about the dataset itself.

Return ONLY valid JSON in this exact format:
{
  "route": "structured | unstructured | out_of_scope",
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
    lowered = query.lower()

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

    if any(keyword in lowered for keyword in out_of_scope_keywords):
        return RouteDecision(
            route="out_of_scope",
            reason="The query asks for general knowledge or recommendations outside the dataset.",
        )

    if any(keyword in lowered for keyword in structured_keywords):
        return RouteDecision(
            route="structured",
            reason="The query asks for concrete dataset facts, counts, examples, or distributions.",
        )

    if any(keyword in lowered for keyword in unstructured_keywords):
        return RouteDecision(
            route="unstructured",
            reason="The query asks for summarization or pattern analysis inside the dataset.",
        )

    return RouteDecision(
        route="out_of_scope",
        reason="The query could not be clearly answered from the customer support dataset.",
    )
