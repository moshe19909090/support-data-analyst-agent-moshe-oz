import argparse

from src.agent import run_agent
from src.data_loader import load_support_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Customer support dataset analyst agent"
    )
    parser.add_argument(
        "--session",
        default="default",
        help="Session ID used to persist and restore conversation memory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_id = args.session

    print("Loading customer support dataset from Hugging Face...")
    df = load_support_dataset()
    print(f"Dataset loaded successfully: {len(df)} rows")
    print("Customer Support Dataset Agent")
    print(f"Session: {session_id}")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not query:
            continue

        result = run_agent(query=query, session_id=session_id)

        print(f"\nRouter: {result['query_type']}")
        print(f"Agent: {result['final_answer']}\n")


if __name__ == "__main__":
    main()
