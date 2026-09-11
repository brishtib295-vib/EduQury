"""Mistral client utilities for EduQuery.

Presentation-ready version:
- Fails fast on invalid credentials (401/403) instead of retrying.
- Uses a short, web-safe retry policy for transient 429/5xx errors.
- Honors Retry-After but caps waiting time so Gunicorn workers do not time out.
- Caches embeddings in-process to reduce repeated API calls during a demo.
- Provides local YouTube search recommendations without an extra LLM call.
"""
from __future__ import annotations

import hashlib
import os
import random
import re
import time
import urllib.parse
from typing import Iterable, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("❌ MISTRAL_API_KEY not found in environment / .env file!")

MISTRAL_CHAT_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_EMBED_URL = "https://api.mistral.ai/v1/embeddings"
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-small-latest")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mistral-embed")

# Safe defaults for a synchronous Flask/Gunicorn demo.
# One retry is enough to survive a transient 429 without creating a 30–60s request.
MAX_RETRIES = int(os.getenv("MISTRAL_MAX_RETRIES", "1"))
INITIAL_BACKOFF = float(os.getenv("MISTRAL_INITIAL_BACKOFF", "2.0"))
MAX_BACKOFF = float(os.getenv("MISTRAL_MAX_BACKOFF", "6.0"))
DEFAULT_TIMEOUT = int(os.getenv("MISTRAL_TIMEOUT", "45"))

_session: Optional[requests.Session] = None
_embedding_cache: dict[str, List[float]] = {}
_EMBED_CACHE_LIMIT = 800


class MistralRateLimitError(RuntimeError):
    """Raised when Mistral returns HTTP 429 after the allowed retry(s)."""

    def __init__(self, message: str = "Mistral API rate limit reached.", retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class MistralAPIError(RuntimeError):
    """Raised for non-transient Mistral API failures."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def _auth_headers() -> dict:
    return {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }


def _safe_response_message(response: requests.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            err = data.get("message") or data.get("detail") or data.get("error")
            if isinstance(err, dict):
                err = err.get("message") or str(err)
            if err:
                return str(err)
    except Exception:
        pass
    return response.text[:500].strip() or response.reason or "Unknown API error"


def _retry_delay(response: Optional[requests.Response], attempt: int) -> float:
    """Use Retry-After when present, but cap it for synchronous web requests."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(MAX_BACKOFF, max(0.5, float(retry_after)))
            except (TypeError, ValueError):
                pass

    delay = min(MAX_BACKOFF, INITIAL_BACKOFF * (2 ** attempt))
    return min(MAX_BACKOFF, delay + random.uniform(0.0, 0.5))


def _post_with_retries(
    url: str,
    payload: dict,
    timeout: int,
    operation: str,
) -> requests.Response:
    """POST to Mistral with web-safe transient retry handling."""
    sess = _get_session()

    for attempt in range(MAX_RETRIES + 1):
        response: Optional[requests.Response] = None

        try:
            response = sess.post(
                url,
                json=payload,
                headers=_auth_headers(),
                timeout=timeout,
            )
        except requests.RequestException as exc:
            if attempt >= MAX_RETRIES:
                raise MistralAPIError(
                    503,
                    f"Mistral {operation} network error. Please try again in a moment."
                ) from exc

            wait = _retry_delay(None, attempt)
            print(
                f"⚠️ Network error during Mistral {operation}. "
                f"Retrying in {wait:.1f}s ({attempt + 1}/{MAX_RETRIES})..."
            )
            time.sleep(wait)
            continue

        if response.ok:
            return response

        status = response.status_code

        # Authentication/permission/bad-request errors must NOT be retried.
        if status in (400, 401, 403, 404, 405, 406, 409, 413, 422):
            message = _safe_response_message(response)
            if status in (401, 403):
                message = "Mistral API key was rejected. Check MISTRAL_API_KEY in Railway."
            raise MistralAPIError(status, f"Mistral {operation} failed ({status}): {message}")

        # Retry only transient rate/server errors.
        if status == 429 or 500 <= status < 600:
            if attempt >= MAX_RETRIES:
                message = _safe_response_message(response)
                if status == 429:
                    raise MistralRateLimitError(
                        "Mistral API rate limit reached. Please wait and try again.",
                        retry_after=None,
                    )
                raise MistralAPIError(
                    status,
                    f"Mistral {operation} temporarily failed ({status}): {message}",
                )

            wait = _retry_delay(response, attempt)
            print(
                f"⚠️ Mistral {status} during {operation}. "
                f"Retrying in {wait:.1f}s ({attempt + 1}/{MAX_RETRIES})..."
            )
            time.sleep(wait)
            continue

        # Any other status is not safe to retry.
        message = _safe_response_message(response)
        raise MistralAPIError(status, f"Mistral {operation} failed ({status}): {message}")

    raise MistralAPIError(503, f"Mistral {operation} failed after retries.")


def chat(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Send a prompt to Mistral Chat and return response text."""
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = _post_with_retries(
        MISTRAL_CHAT_URL, payload, timeout, "chat"
    )

    try:
        return response.json()["choices"][0]["message"]["content"]
    except (KeyError, TypeError, ValueError) as exc:
        raise MistralAPIError(
            502,
            f"Unexpected Mistral chat response: {response.text[:500]}",
        ) from exc


def _chunk_iterable(iterable: Iterable, size: int):
    """Yield successive chunks of size from an iterable."""
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(size):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk


def _embedding_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_embeddings(
    texts: List[str], batch_size: int = 32
) -> List[List[float]]:
    """Return Mistral embeddings, reusing an in-process cache where possible."""
    if not texts:
        return []

    result: List[Optional[List[float]]] = [None] * len(texts)
    missing_texts: List[str] = []
    missing_positions: List[int] = []

    for i, text in enumerate(texts):
        key = _embedding_key(text)
        cached = _embedding_cache.get(key)
        if cached is not None:
            result[i] = cached
        else:
            missing_texts.append(text)
            missing_positions.append(i)

    if missing_texts:
        for batch_start in range(0, len(missing_texts), batch_size):
            batch = missing_texts[batch_start:batch_start + batch_size]
            payload = {"model": EMBED_MODEL, "input": batch}
            response = _post_with_retries(
                MISTRAL_EMBED_URL, payload, 60, "embeddings"
            )

            try:
                data = response.json()
                batch_embs = [
                    item["embedding"]
                    for item in sorted(data["data"], key=lambda x: x["index"])
                ]
            except (KeyError, TypeError, ValueError) as exc:
                raise MistralAPIError(
                    502,
                    f"Unexpected Mistral embedding response: {response.text[:500]}",
                ) from exc

            for local_idx, emb in enumerate(batch_embs):
                global_missing_idx = batch_start + local_idx
                if global_missing_idx >= len(missing_positions):
                    break
                original_idx = missing_positions[global_missing_idx]
                result[original_idx] = emb

                key = _embedding_key(texts[original_idx])
                if len(_embedding_cache) >= _EMBED_CACHE_LIMIT:
                    _embedding_cache.pop(next(iter(_embedding_cache)))
                _embedding_cache[key] = emb

    return [emb for emb in result if emb is not None]


def embed_query(text: str) -> List[float]:
    """Return an embedding for one query, using the same cache."""
    if not text:
        return []
    result = generate_embeddings([text])
    return result[0] if result else []


_STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been",
    "before", "being", "between", "both", "but", "can", "could", "does",
    "doing", "down", "each", "for", "from", "have", "having", "how",
    "into", "its", "just", "more", "most", "other", "our", "over", "same",
    "should", "some", "such", "than", "that", "their", "there", "these",
    "they", "this", "through", "using", "very", "was", "were", "what",
    "when", "where", "which", "while", "with", "would", "your", "you",
    "the", "then", "them", "will", "under", "has", "had", "not", "only",
    "too", "his", "her", "out", "use", "used", "learn", "learning", "topic",
    "topics", "lecture", "video", "notes", "question", "questions", "answer",
    "answers", "welcome", "thank", "thanks",
}


def _extract_topic_phrases(text: str, limit: int = 3) -> List[str]:
    """Extract useful topic phrases locally without an API call."""
    if not text or not text.strip():
        return []

    clean = re.sub(r"https?://\S+", " ", text)
    clean = re.sub(r"[^A-Za-z0-9+#.\- ]+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    words = clean.split()
    candidates: List[str] = []

    for n in (4, 3, 2):
        for i in range(len(words) - n + 1):
            phrase_words = words[i:i + n]
            normalized = [w.strip(".-+#").lower() for w in phrase_words]
            if any(not w or len(w) < 3 or w in _STOPWORDS for w in normalized):
                continue
            phrase = " ".join(phrase_words).strip(" .,-")
            lower = phrase.lower()
            if not (8 <= len(phrase) <= 80):
                continue
            if any(bad in lower for bad in (
                "http", "www", "copyright", "subscribe", "click here"
            )):
                continue
            if lower not in {x.lower() for x in candidates}:
                candidates.append(phrase)
            if len(candidates) >= limit:
                return candidates

    for word in words:
        w = word.strip(".,-+#()[]{}").lower()
        if len(w) >= 5 and w not in _STOPWORDS and w not in [x.lower() for x in candidates]:
            candidates.append(w.title())
            if len(candidates) >= limit:
                break

    return candidates[:limit]


def get_youtube_recommendations(
    source_text: str, limit: int = 3
) -> List[dict]:
    """Return YouTube search recommendations without an extra API/LLM call."""
    topics = _extract_topic_phrases(source_text, limit=limit)
    return [
        {
            "topic": topic,
            "url": "https://www.youtube.com/results?search_query="
            + urllib.parse.quote_plus(topic + " tutorial"),
        }
        for topic in topics
    ]


if __name__ == "__main__":
    print("Mistral client ready.")

