import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from config import BASE_URL, TEXT_MODEL, VISION_MODEL

load_dotenv()


def _api_key() -> str:
    cle = os.getenv("FREELLMAPI_API_KEY")
    if not cle:
        raise RuntimeError(
            "FREELLMAPI_API_KEY absente du .env (cle unifiee du dashboard FreeLLMAPI)."
        )
    return cle


def get_llm(
    model: str | None = None,
    temperature: float = 0,
    vision: bool = False,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or (VISION_MODEL if vision else TEXT_MODEL),
        base_url=BASE_URL,
        api_key=_api_key(),
        temperature=temperature,
    )
