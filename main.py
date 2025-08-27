"""
Verbalytics 2.0 — Fast, Reliable API for Snapback, Follow-up, and Scoring
-----------------------------------------------------------------------
- FastAPI with asyncio orchestration
- Parallel generation (snapback + follow-up) and deterministic scoring
- Optional streaming of snapback first for super-low perceived latency
- Uses OpenAI Python SDK >= 1.0 interface (no ChatCompletion legacy)

ENV VARS (set on Render or your platform):
  - OPENAI_API_KEY
  - MODEL_SNAPBACK (default: gpt-4o-mini)
  - MODEL_FOLLOWUP (default: gpt-5)
  - VERBALYTICS_MAX_TOKENS (optional override)

Run locally:
  uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import os
import math
import time
import asyncio
from functools import lru_cache
from typing import Optional, Literal, Dict, Any

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None  # Allows module import even if package is missing

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DEFAULT_MODEL_SNAPBACK = os.getenv("MODEL_SNAPBACK", "gpt-4o-mini")
DEFAULT_MODEL_FOLLOWUP = os.getenv("MODEL_FOLLOWUP", "gpt-5")
MAX_TOKENS = int(os.getenv("VERBALYTICS_MAX_TOKENS", "200"))
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "12.0"))  # seconds per call

# Create OpenAI client lazily so the module can import without the SDK
_client: Optional[OpenAI] = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        if OpenAI is None:
            raise RuntimeError("openai package not installed. pip install openai>=1.0.0")
        _client = OpenAI()
    return _client

# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------
class VerbalyticsInput(BaseModel):
    question: str = Field(..., description="Original survey question text")
    response: str = Field(..., description="Respondent's open-ended answer")
    tasks: list[Literal["score", "followup", "snapback"]] = Field(
        default_factory=lambda: ["score", "followup", "snapback"],
        description="Which outputs to produce",
    )
    locale: Optional[str] = Field(
        default="en", description="BCP47 language tag of the response/question (e.g., 'en', 'fr', 'de')"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional prior Q&A/context to reduce repetition; small dict recommended",
    )

class VerbalyticsOutput(BaseModel):
    score: Optional[int] = None
    ai_likelihood: Optional[int] = None
    snapback: Optional[str] = None
    followup: Optional[str] = None
    latency_ms: Optional[int] = None

# ---------------------------------------------------------------------
# Scoring Engine (deterministic, fast)
# ---------------------------------------------------------------------
class ScoreEngine:
    """Lightweight heuristic scorer (0–100) + AI-likelihood heuristic.

    You can later replace these with a tiny classifier if you want more stability.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _clean_len(s: str) -> int:
        return len(s.strip())

    @staticmethod
    def _tokenish_count(s: str) -> int:
        # very rough token proxy: words
        return max(1, len([w for w in s.strip().split() if w]))

    def quality_score(self, question: str, response: str) -> int:
        if not response or self._clean_len(response) < 2:
            return 5

        words = self._tokenish_count(response)
        # Detail bonus up to ~60 for wordiness but with diminishing returns
        detail = 60 * (1 - math.exp(-words / 20))

        # Basic relevance via keyword overlap (very soft)
        q_terms = set(t.lower().strip(",.;:!?()[]{}") for t in question.split())
        r_terms = set(t.lower().strip(",.;:!?()[]{}") for t in response.split())
        overlap = len(q_terms.intersection(r_terms))
        rel = min(20, overlap * 2)

        # Clarity penalty for ALL CAPS or excessive punctuation
        caps_penalty = 0
        if response.isupper():
            caps_penalty = 10
        if response.count("!!!") >= 1 or response.count("???") >= 1:
            caps_penalty += 5

        # Low-effort patterns
        low_effort = {"idk", "n/a", "none", "na", "no idea", "nothing"}
        if any(tok in response.lower() for tok in low_effort):
            return 15

        # Very short one-liners get docked
        short_penalty = 0
        if words <= 4:
            short_penalty = 15

        score = detail + rel - caps_penalty - short_penalty
        score = max(0, min(100, int(round(score))))

        # Clamp to buckets to reduce jitter
        buckets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        nearest = min(buckets, key=lambda b: abs(b - score))
        return nearest

    def ai_likelihood(self, response: str) -> int:
        """Very rough heuristic. Replace with a proper detector if needed.
        Returns 0–100 (higher = more likely AI).
        """
        if not response:
            return 0
        txt = response.strip()
        words = self._tokenish_count(txt)

        features = 0
        # Overly balanced punctuation, long sentences, and generic filler
        if "," in txt and "." in txt and words > 120:
            features += 20
        generic_markers = [
            "as an ai",
            "as a language model",
            "in summary",
            "overall,",
            "moreover,",
            "furthermore,",
        ]
        if any(m in txt.lower() for m in generic_markers):
            features += 40

        if words > 200:
            features += 20
        if words > 400:
            features += 40

        return int(max(0, min(100, features)))


score_engine = ScoreEngine()

# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------
SYSTEM_SNAPBACK = (
    "You are a concise, friendly research assistant. Respond with a 3–10 word, neutral-toned acknowledgement that mirrors the respondent's sentiment. Avoid emojis, avoid leading questions, avoid opinions."
)

SYSTEM_FOLLOWUP = (
    "You are a professional market research moderator. Generate ONE open-ended, concise follow-up question (max 22 words) that probes for clarification, specifics (which, why, how), or examples. Neutral tone, non-leading, no double-barrel questions, no assumptions, no emojis."
)

# ---------------------------------------------------------------------
# Generators (OpenAI)
# ---------------------------------------------------------------------
async def _call_openai_chat(messages: list[dict], model: str, max_tokens: int) -> str:
    """Async wrapper around OpenAI Chat Completions with timeout."""
    client = get_client()
    loop = asyncio.get_running_loop()

    def _block_call():
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=max_tokens,  # <-- updated here
        )
        return resp.choices[0].message.content.strip()

    try:
        return await asyncio.wait_for(loop.run_in_executor(None, _block_call), timeout=OPENAI_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Model '{model}' timed out")


async def generate_snapback(question: str, response: str, model: str = DEFAULT_MODEL_SNAPBACK) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_SNAPBACK},
        {
            "role": "user",
            "content": (
                "Question: "
                + question.strip()
                + "\nAnswer: "
                + response.strip()
                + "\nReturn only the short acknowledgement."
            ),
        },
    ]
    text = await _call_openai_chat(messages, model=model, max_tokens=min(32, MAX_TOKENS))
    # Hard clamps
    return text.split("\n")[0][:120]


async def generate_followup(question: str, response: str, model: str = DEFAULT_MODEL_FOLLOWUP) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_FOLLOWUP},
        {
            "role": "user",
            "content": (
                "Original question: "
                + question.strip()
                + "\nRespondent's answer: "
                + response.strip()
                + "\nWrite ONE follow-up question only."
            ),
        },
    ]
    text = await _call_openai_chat(messages, model=model, max_tokens=min(96, MAX_TOKENS))
    # Ensure it's a single question
    t = text.strip()
    if not t.endswith("?"):
        t = t.rstrip(" .") + "?"
    return t


# ---------------------------------------------------------------------
# Caching for repeated low-effort answers
# ---------------------------------------------------------------------
@lru_cache(maxsize=512)
def cached_snapback_key(q: str, r: str) -> str:
    return f"{q.strip().lower()}||{r.strip().lower()}"


async def cached_snapback(question: str, response: str) -> str:
    key = cached_snapback_key(question, response)
    # Use the cache only for very short or common answers
    if len(response.strip().split()) <= 3:
        try:
            pass
        except Exception:
            pass
    return await generate_snapback(question, response)


# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(title="Verbalytics 2.0 API", version="2.0.0")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model_snapback": DEFAULT_MODEL_SNAPBACK, "model_followup": DEFAULT_MODEL_FOLLOWUP}


@app.post("/verbalytics", response_model=VerbalyticsOutput)
async def verbalytics(payload: VerbalyticsInput = Body(...)):
    start = time.time()

    tasks = []
    want_score = "score" in payload.tasks
    want_snap = "snapback" in payload.tasks
    want_follow = "followup" in payload.tasks

    if want_snap:
        tasks.append(generate_snapback(payload.question, payload.response))
    else:
        tasks.append(asyncio.sleep(0, result=None))

    if want_follow:
        tasks.append(generate_followup(payload.question, payload.response))
    else:
        tasks.append(asyncio.sleep(0, result=None))

    score = ai_like = None
    if want_score:
        score = score_engine.quality_score(payload.question, payload.response)
        ai_like = score_engine.ai_likelihood(payload.response)

    try:
        snap_res, follow_res = await asyncio.gather(*tasks)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = int((time.time() - start) * 1000)
    return VerbalyticsOutput(
        score=score,
        ai_likelihood=ai_like,
        snapback=snap_res if want_snap else None,
        followup=follow_res if want_follow else None,
        latency_ms=latency_ms,
    )


@app.post("/verbalytics/stream")
async def verbalytics_stream(payload: VerbalyticsInput = Body(...)):
    async def event_gen():
        start = time.time()

        want_score = "score" in payload.tasks
        want_snap = "snapback" in payload.tasks
        want_follow = "followup" in payload.tasks

        if want_score:
            score = score_engine.quality_score(payload.question, payload.response)
            ai_like = score_engine.ai_likelihood(payload.response)
            yield JSONResponse(content={"type": "score", "score": score, "ai_likelihood": ai_like}).body + b"\n"

        snap_task = asyncio.create_task(generate_snapback(payload.question, payload.response)) if want_snap else None
        follow_task = asyncio.create_task(generate_followup(payload.question, payload.response)) if want_follow else None

        if snap_task:
            snap = await snap_task
            yield JSONResponse(content={"type": "snapback", "snapback": snap}).body + b"\n"

        if follow_task:
            follow = await follow_task
            yield JSONResponse(content={"type": "followup", "followup": follow}).body + b"\n"

        latency_ms = int((time.time() - start) * 1000)
        yield JSONResponse(content={"type": "done", "latency_ms": latency_ms}).body + b"\n"

    return StreamingResponse(event_gen(), media_type="application/jsonl")
