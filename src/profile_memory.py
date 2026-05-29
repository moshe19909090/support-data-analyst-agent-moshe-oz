import json
import re
from pathlib import Path
from typing import Any, Dict, List


MEMORY_DIR = Path("memory")
PROFILE_DIR = MEMORY_DIR / "profiles"


def _profile_path(session_id: str) -> Path:
    safe_session_id = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)
    return PROFILE_DIR / f"{safe_session_id}.json"


def load_user_profile(session_id: str) -> Dict[str, Any]:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(session_id)

    if not path.exists():
        return {
            "name": None,
            "frequent_topics": [],
            "preferences": [],
            "notes": [],
        }

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "name": None,
            "frequent_topics": [],
            "preferences": [],
            "notes": [],
        }


def save_user_profile(session_id: str, profile: Dict[str, Any]) -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(session_id)
    path.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _append_unique(items: List[str], value: str) -> List[str]:
    normalized_value = value.strip()
    if not normalized_value:
        return items

    existing = {item.lower().strip() for item in items}
    if normalized_value.lower() not in existing:
        items.append(normalized_value)

    return items


def update_user_profile_from_turn(
    session_id: str,
    user_query: str = "",
    final_answer: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Maintains a small distilled user profile.

    This is intentionally not a replay of the conversation.
    It stores stable facts/preferences/topics that may help future interactions.

    Accepts both positional arguments:
        update_user_profile_from_turn(session_id, user_query, final_answer)

    and keyword arguments used by the agent:
        update_user_profile_from_turn(
            session_id=session_id,
            user_message=query,
            assistant_message=final_answer,
        )
    """
    user_query = user_query or kwargs.get("user_message", "") or kwargs.get("query", "")
    final_answer = final_answer or kwargs.get("assistant_message", "") or kwargs.get("answer", "")

    profile = load_user_profile(session_id)
    lowered = user_query.lower()

    # Capture simple explicit name statements.
    name_patterns = [
        r"\bmy name is ([a-zA-Z][a-zA-Z\s'-]{1,40})",
        r"\bi am ([a-zA-Z][a-zA-Z\s'-]{1,40})",
        r"\bi'm ([a-zA-Z][a-zA-Z\s'-]{1,40})",
        r"\bcall me ([a-zA-Z][a-zA-Z\s'-]{1,40})",
    ]

    for pattern in name_patterns:
        match = re.search(pattern, user_query, flags=re.IGNORECASE)
        if match:
            profile["name"] = match.group(1).strip().rstrip(".")
            break

    # Track frequent dataset topics.
    topic_keywords = {
        "refunds": ["refund", "refunds", "money back", "reimbursement"],
        "shipping": ["shipping", "delivery address", "address"],
        "accounts": ["account", "password", "registration"],
        "complaints": ["complaint", "complaints"],
        "cancellations": ["cancel", "cancellation"],
        "intent distributions": ["distribution", "intent distribution"],
        "dataset examples": ["example", "examples", "show me"],
    }

    for topic, keywords in topic_keywords.items():
        if any(keyword in lowered for keyword in keywords):
            profile["frequent_topics"] = _append_unique(
                profile.get("frequent_topics", []),
                topic,
            )

    # Capture simple preference statements.
    preference_patterns = [
        r"\bi prefer ([^.]+)",
        r"\bi like ([^.]+)",
        r"\bi want ([^.]+)",
    ]

    for pattern in preference_patterns:
        match = re.search(pattern, user_query, flags=re.IGNORECASE)
        if match:
            preference = match.group(0).strip().rstrip(".")
            profile["preferences"] = _append_unique(
                profile.get("preferences", []),
                preference,
            )

    save_user_profile(session_id, profile)
    return profile


def format_user_profile(session_id: str) -> str:
    profile = load_user_profile(session_id)

    lines = ["Here is what I remember about you from the persistent user profile:"]

    if profile.get("name"):
        lines.append(f"- Name: {profile['name']}")

    frequent_topics = profile.get("frequent_topics", [])
    if frequent_topics:
        lines.append(f"- Frequent topics: {', '.join(frequent_topics)}")

    preferences = profile.get("preferences", [])
    if preferences:
        lines.append("- Preferences:")
        for preference in preferences:
            lines.append(f"  - {preference}")

    notes = profile.get("notes", [])
    if notes:
        lines.append("- Notes:")
        for note in notes:
            lines.append(f"  - {note}")

    if len(lines) == 1:
        lines.append("- I do not have any saved profile facts yet.")

    return "\n".join(lines)
