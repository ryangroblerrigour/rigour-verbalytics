"""
Verbalytics 2.2 — Fast, Reliable API for Acknowledgement, Snapback, Follow-up, and Scoring
-----------------------------------------------------------------------
- FastAPI with asyncio orchestration
- Parallel generation (ack + snapback + follow-up) and deterministic scoring
- Streaming JSONL endpoint for low perceived latency
- Uses OpenAI Python SDK >= 1.0 (client.chat.completions.create)

ENV VARS:
  - OPENAI_API_KEY
  - MODEL_ACK (default: gpt-4o-mini)
  - MODEL_SNAPBACK (default: gpt-4o-mini)
  - MODEL_FOLLOWUP (default: gpt-5)
  - VERBALYTICS_MAX_TOKENS (default: 200)
  - GOOGLE_BLOCKLIST_SHEET (publish-to-web CSV url)
  - VERBALYTICS_BLOCKLIST_REFRESH_SECONDS (default: 300)
  - VERBALYTICS_BLOCK_INPUT_MODE (off | flag | reject)  # default: flag

Run locally:
  uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import os, math, time, asyncio, csv, requests
from typing import Optional, Literal, Dict, Any
from time import monotonic
from threading import RLock

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DEFAULT_MODEL_ACK = os.getenv("MODEL_ACK", "gpt-4o-mini")
DEFAULT_MODEL_SNAPBACK = os.getenv("MODEL_SNAPBACK", "gpt-4o-mini")
DEFAULT_MODEL_FOLLOWUP = os.getenv("MODEL_FOLLOWUP", "gpt-5")
MAX_TOKENS = int(os.getenv("VERBALYTICS_MAX_TOKENS", "200"))
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "12.0"))

BLOCKLIST_CSV_URL = os.getenv("GOOGLE_BLOCKLIST_SHEET", "")
BLOCKLIST_REFRESH_SECONDS = int(os.getenv("VERBALYTICS_BLOCKLIST_REFRESH_SECONDS", "300"))
INPUT_BLOCK_MODE = os.getenv("VERBALYTICS_BLOCK_INPUT_MODE", "flag").lower()  # off | flag | reject

# ---------------------------------------------------------------------
# OpenAI Client
# ---------------------------------------------------------------------
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
    question: str
    response: str
    tasks: list[Literal["score","ack","followup"]] = Field(default_factory=lambda: ["score","ack","followup"])
    locale: Optional[str] = "en"
    context: Optional[Dict[str, Any]] = None
    project_id: Optional[str] = None

class VerbalyticsOutput(BaseModel):
    subscores: Optional[Dict[str, int]] = None
    score: Optional[int] = None
    ai_likelihood: Optional[int] = None
    ack: Optional[str] = None
    snapback: Optional[str] = None
    followup: Optional[str] = None
    input_blocked_phrases: Optional[list[str]] = None
    latency_ms: Optional[int] = None

# ---------------------------------------------------------------------
# Blocklist Loader (global + project)
# ---------------------------------------------------------------------
_block_lock = RLock()
_block_global: set[str] = set()
_block_projects: dict[str, set[str]] = {}
_block_loaded_at: float = 0.0
_block_version: int = 0

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def _load_blocklist_csv(url: str) -> tuple[set[str], dict[str, set[str]]]:
    g, p = set(), {}
    if not url:
        return g, p
    try:
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        reader = csv.DictReader(resp.content.decode("utf-8", errors="ignore").splitlines())
        for row in reader:
            scope = _norm(row.get("scope", ""))
            proj = _norm(row.get("project", ""))
            phrase = _norm(row.get("phrase", ""))
            if not phrase:
                continue
            if scope == "global" or scope == "":
                g.add(phrase)
            elif scope == "project" and proj:
                p.setdefault(proj, set()).add(phrase)
    except Exception:
        # keep previous cache on error
        pass
    return g, p

def _refresh_blocklist(force: bool = False) -> None:
    global _block_global, _block_projects, _block_loaded_at, _block_version
    now = monotonic()
    with _block_lock:
        if not force and (now - _block_loaded_at) < BLOCKLIST_REFRESH_SECONDS:
            return
        g, p = _load_blocklist_csv(BLOCKLIST_CSV_URL)
        if g or p or force:
            _block_global = g
            _block_projects = p
            _block_loaded_at = now
            _block_version += 1

def get_block_phrases(project_id: Optional[str]) -> set[str]:
    _refresh_blocklist(False)
    with _block_lock:
        phrases = set(_block_global)
        if project_id:
            phrases |= _block_projects.get(_norm(project_id), set())
        return phrases

def contains_blocked_phrase(text: str, project_id: Optional[str]) -> bool:
    if not text:
        return False
    lt = text.lower()
    return any(p in lt for p in get_block_phrases(project_id))

def find_blocked_phrases(text: str, project_id: Optional[str]) -> list[str]:
    if not text:
        return []
    lt = text.lower()
    phrases = get_block_phrases(project_id)
    hits = {p for p in phrases if p and p in lt}
    return sorted(list(hits))

# ---------------------------------------------------------------------
# Scoring Engine (Subscores + improved AI-likeness)
# ---------------------------------------------------------------------
class ScoreEngine:
    # --- helpers ---
    def _tokens(self, s: str) -> list[str]:
        return [t.lower().strip(",.;:!?()[]{}\"'") for t in (s or "").split() if t.strip()]

    def _word_count(self, s: str) -> int:
        return len(self._tokens(s))

    def _jaccard(self, a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    # --- subscores ---
    def subscores(self, question: str, response: str) -> dict:
        q = (question or "").strip()
        r = (response or "").strip()

        if not r or r.lower() in {"", "na", "n/a", "none", "no idea", "idk", "?"}:
            return {"specificity": 0, "concreteness": 0, "relevance": 0, "clarity": 0}

        q_toks = set(self._tokens(q))
        r_toks = self._tokens(r)
        r_set  = set(r_toks)
        wc     = len(r_toks)

        # Specificity: reasons, locators, unique mentions
        has_reason = any(k in r_set for k in {"because","since","so that","due"})
        has_locators = any(k in r_set for k in {"when","where","which","that"})
        unique_terms = len([t for t in r_set if t not in q_toks and len(t) > 3])
        spec_base = 60 * (1 - (2.71828 ** (-wc / 18)))  # diminishing returns
        spec_bonus = 10 * has_reason + 6 * has_locators + min(24, unique_terms)
        specificity = max(0, min(100, int(spec_base + spec_bonus)))

        # Concreteness (UPDATED): first-hand usage + context + feature/sensory + examples + light proper-noun weight
        first_hand_markers = {
            "i use", "i used", "i tried", "i've used", "i have used", "i bought", "we bought",
            "my kids use", "my kid uses", "my children", "my family", "we use", "i saw", "i heard", "i've seen"
        }
        usage_context = {
            "at home", "at work", "for work", "on the train", "on the commute", "in the car",
            "at night", "in the morning", "on weekends", "after school", "during football", "during ads",
            "when cooking", "when cleaning", "while driving", "before bed"
        }
        feature_sensory = {
            "logo","pack","packaging","design","music","jingle","voiceover","scene","actor","character",
            "taste","smell","texture","colour","color","price","offer","discount","durable","battery","speed",
            "instructions","interface","app","sound","volume","quality","resolution","camera","label","slogan"
        }
        example_markers = {"for example", "such as", "like when", "like the time", "e.g."}

        rt = " " + " ".join(r_toks) + " "
        def has_any_phrase(hay: str, phrases: set[str]) -> bool:
            return any((" " + p + " ") in hay for p in phrases)

        first_hand = has_any_phrase(rt, first_hand_markers)
        context_hit = has_any_phrase(rt, usage_context)
        example_hit = has_any_phrase(rt, example_markers)
        feature_hits = sum(1 for f in feature_sensory if f in r_set)
        properish = sum(1 for w in (response.split()) if w[:1].isupper() and len(w) > 3)
        length_credit = min(30, wc // 5)

        conc_score = (
            (22 if first_hand else 0) +
            (14 if context_hit else 0) +
            (12 if example_hit else 0) +
            min(28, feature_hits * 6) +
            min(12, properish * 2) +
            length_credit
        )
        concreteness = max(0, min(100, int(conc_score)))

        # Relevance: lexical overlap; penalize off-topic markers
        rel_overl_
