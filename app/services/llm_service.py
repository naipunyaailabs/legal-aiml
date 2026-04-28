"""LLM service wrapping Groq API with OpenAI-compatible interface."""
import httpx
import json
import logging
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


async def call_llm(
    messages: list[dict],
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    response_format: dict | None = None,
) -> str:
    """Call Groq LLM API. Falls back to smaller model on rate limit."""
    model = model or settings.GROQ_MODEL
    max_tokens = max_tokens or settings.GROQ_MAX_TOKENS
    temperature = temperature if temperature is not None else settings.GROQ_TEMPERATURE

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(GROQ_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Rate limited — try fallback model
                logger.warning(f"Rate limited on {model}, trying fallback {settings.GROQ_FALLBACK_MODEL}")
                payload["model"] = settings.GROQ_FALLBACK_MODEL
                resp = await client.post(GROQ_API_URL, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            raise
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
