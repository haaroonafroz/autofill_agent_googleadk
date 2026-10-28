"""
Factories for the chat and embedding models used by the agent.
"""

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import AGENT_MODEL, AGENT_REASONING_EFFORT, EMBEDDING_MODEL, require_api_key


def get_chat_model(model: str = AGENT_MODEL, reasoning_effort: str = AGENT_REASONING_EFFORT) -> ChatOpenAI:
    """Returns a LangChain chat model that talks to OpenAI via the Responses API."""
    return ChatOpenAI(
        model=model,
        reasoning={"effort": reasoning_effort},
        use_responses_api=True,
        api_key=require_api_key(),
    )


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=require_api_key())
