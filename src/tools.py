from typing import Dict, List, Optional

import re

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.data_loader import load_support_dataset


# -----------------------------
# Input Schemas
# -----------------------------


class EmptyInput(BaseModel):
    """No input required."""


class CategoryInput(BaseModel):
    category: str = Field(
        description=(
            "Customer support category name. Examples: ACCOUNT, SHIPPING, "
            "FEEDBACK, CANCEL, REFUND. Use CANCEL for cancellation-related questions."
        )
    )


class IntentsInput(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description=(
            "Optional category name. If provided, return intents only for this category. "
            "Examples: ACCOUNT, SHIPPING, REFUND, CANCEL."
        ),
    )


class IntentInput(BaseModel):
    intent: str = Field(
        description=(
            "Exact customer support intent name. Examples: get_refund, cancel_order, "
            "track_order, recover_password."
        )
    )


class IntentSearchInput(BaseModel):
    keyword: str = Field(
        description=(
            "Keyword used to find matching intent names. Examples: refund, cancel, "
            "shipping, complaint, password."
        )
    )


class CountRowsInput(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description="Optional exact category name to filter by, for example SHIPPING or ACCOUNT.",
    )
    intent: Optional[str] = Field(
        default=None,
        description="Optional exact intent name to filter by, for example get_refund.",
    )


class ExampleInput(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description="Optional category to filter by, for example SHIPPING, ACCOUNT, REFUND, CANCEL.",
    )
    intent: Optional[str] = Field(
        default=None,
        description="Optional exact intent to filter by, for example get_refund.",
    )
    limit: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of examples to return. Must be between 1 and 10.",
    )


class SearchInput(BaseModel):
    query: str = Field(
        description=(
            "Plain-English phrase or keywords to search in customer instructions and responses. "
            "Examples: 'money back', 'refund', 'cancel order', 'complaint', 'shipping address'."
        )
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of matching examples to return. Must be between 1 and 10.",
    )


class SummarizeCategoryInput(BaseModel):
    category: str = Field(
        description="Category to summarize, for example FEEDBACK, ACCOUNT, SHIPPING, CANCEL, REFUND."
    )
    limit: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Number of representative examples to retrieve for summarization.",
    )


# -----------------------------
# Helpers
# -----------------------------


CATEGORY_ALIASES = {
    "cancellation": "CANCEL",
    "cancellations": "CANCEL",
    "cancel": "CANCEL",
    "shipping": "SHIPPING",
    "shipment": "SHIPPING",
    "delivery": "DELIVERY",
    "refund": "REFUND",
    "refunds": "REFUND",
    "account": "ACCOUNT",
    "feedback": "FEEDBACK",
    "payment": "PAYMENT",
    "invoice": "INVOICE",
    "order": "ORDER",
    "subscription": "SUBSCRIPTION",
    "contact": "CONTACT",
}


def normalize_category(category: Optional[str]) -> Optional[str]:
    if not category:
        return None

    cleaned = category.strip().lower()
    return CATEGORY_ALIASES.get(cleaned, category.strip().upper())


def normalize_intent(intent: Optional[str]) -> Optional[str]:
    if not intent:
        return None

    return intent.strip().lower()


def rows_to_examples(df, limit: int) -> List[Dict[str, str]]:
    rows = df.head(limit)

    return [
        {
            "category": str(row["category"]),
            "intent": str(row["intent"]),
            "instruction": str(row["instruction"]),
            "response": str(row["response"]),
        }
        for _, row in rows.iterrows()
    ]


def apply_filters(df, category: Optional[str] = None, intent: Optional[str] = None):
    normalized_category = normalize_category(category)
    normalized_intent = normalize_intent(intent)

    if normalized_category:
        df = df[df["category"].str.lower() == normalized_category.lower()]

    if normalized_intent:
        df = df[df["intent"].str.lower() == normalized_intent.lower()]

    return df


# -----------------------------
# Tools
# -----------------------------


def get_categories() -> List[str]:
    """
    Return all unique customer support categories in the dataset.
    Use this tool when the user asks what categories exist in the dataset.
    """
    df = load_support_dataset()
    return sorted(df["category"].dropna().unique().tolist())


def get_intents(category: Optional[str] = None) -> List[str]:
    """
    Return unique customer support intents.
    Use this when the user asks what intents exist, optionally inside a category.
    """
    df = load_support_dataset()
    df = apply_filters(df, category=category)

    return sorted(df["intent"].dropna().unique().tolist())


def find_intents_by_keyword(keyword: str) -> List[str]:
    """
    Find intent names that contain a keyword.
    Use this before counting or filtering when the user uses a natural phrase
    like 'refund requests', 'money back', 'cancel requests', or 'shipping issues'
    and you need to discover the exact intent name first.
    """
    df = load_support_dataset()
    cleaned_keyword = keyword.strip().lower()

    matches = (
        df[df["intent"].str.lower().str.contains(cleaned_keyword, regex=False)][
            "intent"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    return sorted(matches)


def count_rows(category: Optional[str] = None, intent: Optional[str] = None) -> int:
    """
    Count rows in the dataset, optionally filtered by category and/or exact intent.
    Use this after discovering the exact category or intent.
    Example multi-step use: find_intents_by_keyword('refund') -> count_rows(intent='get_refund').
    """
    df = load_support_dataset()
    df = apply_filters(df, category=category, intent=intent)

    return int(len(df))


def get_examples(
    category: Optional[str] = None,
    intent: Optional[str] = None,
    limit: int = 3,
) -> List[Dict[str, str]]:
    """
    Return example customer instructions and support responses.
    Use this when the user asks for examples from a category or exact intent.
    For example: 'Show me 5 examples of the SHIPPING category.'
    """
    df = load_support_dataset()
    df = apply_filters(df, category=category, intent=intent)

    return rows_to_examples(df, limit)


def search_instructions(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Search customer instructions and support responses using flexible keyword matching.
    Use this for natural-language searches where the user does not provide an exact category
    or intent, such as 'people wanting their money back' or 'complaint intents'.
    """
    df = load_support_dataset()

    raw_query = query.lower().strip()

    # Small synonym expansion for common assignment examples.
    synonyms = {
        "money back": ["money back", "refund", "reimbursement", "return my money"],
        "wanting their money back": [
            "money back",
            "refund",
            "reimbursement",
            "return my money",
        ],
        "complaint": [
            "complaint",
            "complain",
            "dissatisfied",
            "unhappy",
            "bad experience",
        ],
        "complaints": [
            "complaint",
            "complain",
            "dissatisfied",
            "unhappy",
            "bad experience",
        ],
        "cancellation": ["cancel", "cancellation"],
        "cancel": ["cancel", "cancellation"],
        "shipping": ["shipping", "shipment", "delivery"],
    }

    terms = [raw_query]

    for key, values in synonyms.items():
        if key in raw_query:
            terms.extend(values)

    # Also split the query into meaningful words.
    words = [word for word in re.findall(r"[a-zA-Z_]+", raw_query) if len(word) >= 4]
    terms.extend(words)

    # Remove duplicates while preserving order.
    seen = set()
    unique_terms = []
    for term in terms:
        term = term.strip().lower()
        if term and term not in seen:
            seen.add(term)
            unique_terms.append(term)

    searchable_text = (
        df["instruction"].fillna("").str.lower()
        + " "
        + df["response"].fillna("").str.lower()
        + " "
        + df["intent"].fillna("").str.lower()
        + " "
        + df["category"].fillna("").str.lower()
    )

    mask = False
    for term in unique_terms:
        mask = mask | searchable_text.str.contains(term, regex=False)

    results = df[mask].head(limit)

    return rows_to_examples(results, limit)


def get_intent_distribution(category: str) -> Dict[str, int]:
    """
    Return the distribution of intents inside a specific support category.
    Use this when the user asks for intent distribution in a category.
    Example: 'What is the distribution of intents in the ACCOUNT category?'
    """
    df = load_support_dataset()
    normalized_category = normalize_category(category)
    filtered = df[df["category"].str.lower() == normalized_category.lower()]

    return {
        str(intent): int(count)
        for intent, count in filtered["intent"].value_counts().to_dict().items()
    }


def summarize_category_data(category: str, limit: int = 20) -> List[Dict[str, str]]:
    """
    Return representative examples from a category for summarization.
    Use this before summarizing how customers behave or how agents respond in a category.
    """
    df = load_support_dataset()
    normalized_category = normalize_category(category)
    filtered = df[df["category"].str.lower() == normalized_category.lower()].head(limit)

    return [
        {
            "category": str(row["category"]),
            "intent": str(row["intent"]),
            "instruction": str(row["instruction"]),
            "response": str(row["response"]),
        }
        for _, row in filtered.iterrows()
    ]


# -----------------------------
# Tool registration
# -----------------------------


def build_tools():
    return [
        StructuredTool.from_function(
            func=get_categories,
            name="get_categories",
            description=(
                "Return all unique customer support categories in the dataset. "
                "Use when the user asks what categories exist."
            ),
            args_schema=EmptyInput,
        ),
        StructuredTool.from_function(
            func=get_intents,
            name="get_intents",
            description=(
                "Return all unique customer support intents, optionally filtered by category. "
                "Use when the user asks what intents exist."
            ),
            args_schema=IntentsInput,
        ),
        StructuredTool.from_function(
            func=find_intents_by_keyword,
            name="find_intents_by_keyword",
            description=(
                "Find exact intent names by keyword. Use this before counting/filtering when "
                "the user uses natural language like 'refund requests', 'money back', "
                "'cancel requests', or 'shipping issues'."
            ),
            args_schema=IntentSearchInput,
        ),
        StructuredTool.from_function(
            func=count_rows,
            name="count_rows",
            description=(
                "Count rows in the customer support dataset, optionally filtered by exact "
                "category and/or exact intent. Use after finding the correct intent/category."
            ),
            args_schema=CountRowsInput,
        ),
        StructuredTool.from_function(
            func=get_examples,
            name="get_examples",
            description=(
                "Return example customer instructions and support responses filtered by "
                "category or exact intent. Use when the user asks to show examples."
            ),
            args_schema=ExampleInput,
        ),
        StructuredTool.from_function(
            func=search_instructions,
            name="search_instructions",
            description=(
                "Search customer instructions, responses, intents, and categories using flexible "
                "keyword matching. Use for natural-language dataset searches like 'people wanting "
                "their money back' or 'complaint intents'."
            ),
            args_schema=SearchInput,
        ),
        StructuredTool.from_function(
            func=get_intent_distribution,
            name="get_intent_distribution",
            description=(
                "Return a count distribution of intents inside a specific support category. "
                "Use for questions like 'What is the distribution of intents in ACCOUNT?'"
            ),
            args_schema=CategoryInput,
        ),
        StructuredTool.from_function(
            func=summarize_category_data,
            name="summarize_category_data",
            description=(
                "Return representative rows from a category so the agent can summarize customer "
                "issues and support agent responses based only on dataset examples."
            ),
            args_schema=SummarizeCategoryInput,
        ),
    ]
