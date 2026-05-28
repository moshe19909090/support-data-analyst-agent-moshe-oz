import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """
    Create an OpenAI-compatible chat model client using Nebius Token Factory.
    """
    api_key = os.getenv("NEBIUS_API_KEY")
    base_url = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
    model = os.getenv("NEBIUS_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")

    if not api_key:
        raise ValueError(
            "Missing NEBIUS_API_KEY. Add it to your .env file before running the app."
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )