"""Mistral client utilities for EduQuery.

Includes robust 429/5xx retry handling and lightweight local YouTube
recommendations that do not make an extra Mistral API call.
"""
from __future__ import annotations

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

MAX_RETRIES = 4
INITIAL_BACKOFF = 3.0
MAX_BACKOFF = 30.0

_session: Optional[requests.Session] = None


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


def _retry_delay(response: Optional[requests.Response], attempt: int) -> float:
    """Use Retry-After when available; otherwise exponential backoff + jitter."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(MAX_BACKOFF, max(1.0, float(retry_after)))
            except (TypeError, ValueError):
                pass

    delay = min(MAX_BACKOFF, INITIAL_BACKOFF * (2 ** attempt))
    return min(MAX_BACKOFF, delay + random.uniform(0.0, 0.75))


def _post_with_retries(
    url: str,
    payload: dict,
    timeout: int,
    operation: str,
) -> requests.Response:
    """POST to Mistral; retry only transient rate/server/network failures."""
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

            if response.ok:
                return response

            status = response.status_code

            # 429 = rate limit; 5xx = transient server problem.
            if status == 429 or 500 <= status < 600:
                if attempt >= MAX_RETRIES:
                    response.raise_for_status()

                wait = _retry_delay(response, attempt)
                if status == 429:
                    print(
                        f"⚠️ Mistral rate limit (429) during {operation}. "
                        f"Retrying in {wait:.1f}s "
                        f"(retry {attempt + 1}/{MAX_RETRIES})..."
                    )
                else:
                    print(
                        f"⚠️ Mistral server error ({status}) during {operation}. "
                        f"Retrying in {wait:.1f}s "
                        f"(retry {attempt + 1}/{MAX_RETRIES})..."
                    )
                time.sleep(wait)
                continue

            # 400/401/403/etc. are not fixed by retrying.
            response.raise_for_status()

        except requests.RequestException as exc:
            if attempt >= MAX_RETRIES:
                raise RuntimeError(
                    f"Mistral {operation} failed after {MAX_RETRIES + 1} attempts: {exc}"
                ) from exc

            wait = _retry_delay(response, attempt)
            print(
                f"⚠️ Network error during Mistral {operation}. "
                f"Retrying in {wait:.1f}s "
                f"(retry {attempt + 1}/{MAX_RETRIES})..."
            )
            time.sleep(wait)

    raise RuntimeError(f"Mistral {operation} failed after retries.")


def chat(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    timeout: int = 120,
) -> str:
    """Send a prompt to Mistral Chat and return the response text."""
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
        raise RuntimeError(
            f"Unexpected Mistral chat response: {response.text[:500]}"
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


def generate_embeddings(
    texts: List[str], batch_size: int = 32
) -> List[List[float]]:
    """Return embeddings using Mistral's embedding API."""
    if not texts:
        return []

    all_embs: List[List[float]] = []

    for batch in _chunk_iterable(texts, batch_size):
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
            all_embs.extend(batch_embs)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Unexpected Mistral embedding response: {response.text[:500]}"
            ) from exc

    return all_embs


def embed_query(text: str) -> List[float]:
    """Return an embedding for a single query string."""
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

    # Fallback for short source text.
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
    """Return YouTube search recommendations without making an API/LLM call."""
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
    print("Testing Mistral chat...")
    print(chat("Hello! Explain RAG in simple words."))

    print("\nTesting Mistral embeddings...")
    vec = embed_query("machine learning")
    print(f"Embedding dimensions: {len(vec)}")
    print(f"First 5 values: {vec[:5]}")

    print("\nTesting local YouTube recommendations...")
    print(get_youtube_recommendations(
        "Python exception handling try except finally functions",
        limit=3,
    ))
