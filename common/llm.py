"""LLM factory.

Supports two modes:
1) OpenAI direct (preferred when OPENAI_API_KEY is set)
2) OpenRouter fallback (when OPENROUTER_API_KEY is set)
"""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        return ChatOpenAI(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            api_key=openai_api_key,
            temperature=temperature,
        )

    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_api_key:
        return ChatOpenAI(
            model=os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
            base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=openrouter_api_key,
            temperature=temperature,
        )

    raise RuntimeError(
        "Set OPENAI_API_KEY (preferred) or OPENROUTER_API_KEY in .env before running."
    )

