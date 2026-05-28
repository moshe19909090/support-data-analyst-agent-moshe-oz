from src.agent import run_agent
from src.data_loader import load_support_dataset


def main() -> None:
    print("Loading customer support dataset from Hugging Face...")
    df = load_support_dataset()
    print(f"Dataset loaded successfully: {len(df)} rows")
    print("Customer Support Dataset Agent")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not query:
            continue

        result = run_agent(query)

        print(f"\nRouter: {result['query_type']}")
        print(f"Agent: {result['final_answer']}\n")


if __name__ == "__main__":
    main()
