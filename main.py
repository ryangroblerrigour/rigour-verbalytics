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
    OpenAI = None  # service will error at call-time if not installed

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
    # helpers
    def _tokens(self, s: str) -> list[str]:
        return [t.lower().strip(",.;:!?()[]{}\"'") for t in (s or "").split() if t.strip()]

    def _jaccard(self, a: set[str], b: set[str]) -> float:
        if not a or not b: return 0.0
        inter = len(a & b); union = len(a | b)
        return inter / union if union else 0.0

    # subscores
    def subscores(self, question: str, response: str) -> dict:
        q = (question or "").strip()
        r = (response or "").strip()

        if not r or r.lower() in {"", "na", "n/a", "none", "no idea", "idk", "?"}:
            return {"specificity": 0, "concreteness": 0, "relevance": 0, "clarity": 0}

        q_toks = set(self._tokens(q))
        r_toks = self._tokens(r)
        r_set  = set(r_toks)
        wc     = len(r_toks)

        # ---------- Specificity ----------
        has_reason = any(k in r_set for k in {"because","since","so that","due"})
        has_emphasis = "especially" in r_set or "particularly" in r_set
        unique_terms = len([t for t in r_set if t not in q_toks and len(t) > 3])
        spec_base = 55 * (1 - (2.71828 ** (-wc / 16)))
        spec_bonus = (12 if has_reason else 0) + (8 if has_emphasis else 0) + min(22, unique_terms)
        specificity = max(0, min(100, int(spec_base + spec_bonus)))

        # ---------- Concreteness ----------
        first_hand_markers = {
            "i use","i used","i tried","i've used","i have used","i bought","we bought",
            "my kids use","my kid uses","my children","my family","we use","i saw","i heard","i've seen"
        }
        usage_context = {
            "at home","at work","for work","on the train","on the commute","in the car",
            "at night","in the morning","on weekends","after school","during football","during ads",
            "when cooking","when cleaning","while driving","before bed"
        }
        feature_sensory = {
            "logo","pack","packaging","design","music","jingle","voiceover","scene","actor","character","dog",
            "taste","smell","texture","colour","color","price","offer","discount","durable","battery","speed",
            "instructions","interface","app","sound","volume","quality","resolution","camera","label","slogan"
        }
        example_markers = {"for example","such as","like when","like the time","e.g.","especially","particularly"}

        rt = " " + " ".join(r_toks) + " "
        def has_any_phrase(hay: str, phrases: set[str]) -> bool:
            return any((" " + p + " ") in hay for p in phrases)

        first_hand = has_any_phrase(rt, first_hand_markers)
        context_hit = has_any_phrase(rt, usage_context)
        example_hit = has_any_phrase(rt, example_markers)
        feature_hits = sum(1 for f in feature_sensory if f in r_set)

        properish = sum(1 for w in (response.split()) if (w[:1].isupper() and len(w) > 3))
        length_credit = min(28, wc // 5)

        conc_score = (
            (22 if first_hand else 0) +
            (14 if context_hit else 0) +
            (12 if example_hit else 0) +
            min(30, feature_hits * 7) +
            min(10, properish * 2) +
            length_credit
        )
        concreteness = max(0, min(100, int(conc_score)))

        # ---------- Relevance ----------
        rel_overlap = 100 * self._jaccard(q_toks, r_set)
        opinion_q = any(k in q_toks for k in {"think","opinion","feel","like","dislike","impression","favourite","favorite","rate"})
        subjective_resp = any(k in r_set for k in {
            "good","bad","great","love","hate","nice","funny","boring","memorable","confusing","clear","useful","annoying",
            "enjoy","enjoyed","liked","disliked","amazing","awful","meh"
        })
        refers_to_ad = any(k in r_set for k in {"ad","advert","advertisement","commercial","spot","it","this"})

        rel_heur = 0
        if opinion_q and subjective_resp:
            rel_heur += 65
        if refers_to_ad:
            rel_heur += 15
        relevance = max(0, min(100, int(max(rel_overlap, rel_heur))))

        # ---------- Clarity ----------
        sentences = [s for s in r.replace("!", ".").split(".") if s.strip()]
        avg_len = (sum(len(self._tokens(s)) for s in sentences)/len(sentences)) if sentences else wc
        too_long = avg_len > 28
        too_short = avg_len < 3
        fillers = {"like","basically","sort of","kind of","you know","stuff","things"}
        hedge = any(f in " ".join(r_toks) for f in fillers)
        clarity_base = 82
        clarity_pen = (14 if too_long else 0) + (14 if too_short else 0) + (8 if hedge else 0)
        clarity = max(0, min(100, int(clarity_base - clarity_pen)))

        return {
            "specificity": specificity,
            "concreteness": concreteness,
            "relevance": relevance,
            "clarity": clarity,
        }

    def quality_score(self, q: str, r: str) -> int:
        ss = self.subscores(q, r)
        avg = (ss["specificity"] + ss["concreteness"] + ss["relevance"] + ss["clarity"]) / 4
        return max(0, min(100, int(round(avg / 10) * 10)))

    def ai_likelihood(self, r: str) -> int:
        if not r: return 0
        toks = self._tokens(r); wc = len(toks)

        base = 0
        if wc >= 8: base = 5
        if wc >= 20: base = 10

        templ = [
            "as an ai","as a language model","overall,","moreover,","furthermore,",
            "in summary","in conclusion","additionally,","importantly,"
        ]
        templ_hits = sum(1 for t in templ if t in r.lower())

        sents = [s.strip() for s in r.replace("!", ".").split(".") if s.strip()]
        lengths = [len(self._tokens(s)) for s in sents] or [wc]
        var = (max(lengths) - min(lengths))
        low_var = 12 if (len(lengths) >= 3 and var <= 4) else 0

        uniq_ratio = len(set(toks)) / (wc or 1)
        rep = 14 if uniq_ratio < 0.48 else 0

        verbose = 16 if wc > 180 else 0
        very_verbose = 18 if wc > 350 else 0

        score = base + templ_hits*22 + low_var + rep + verbose + very_verbose
        return int(max(0, min(100, score)))


score_engine = ScoreEngine()

# ---------------------------------------------------------------------
# Magic-moments coverage helpers (NEW)
# ---------------------------------------------------------------------
def _covers_what_happened(text: str) -> bool:
    if not text: return False
    t = " " + text.lower() + " "
    action_markers = {
        "i went","she gave","they helped","staff","associate","cashier","manager",
        "showed me","found","brought","fixed","gift-wrapped","opened","closed",
        "in aisle","at checkout","in store","queue","line","counter","fitting room",
        "then","after","before","first","next","finally"
    }
    return any((" " + m + " ") in t for m in action_markers)

def _covers_why_magic(text: str) -> bool:
    if not text: return False
    t = text.lower()
    reason_markers = {
        "because","so that","which meant","made me feel","felt","surprised","unexpected",
        "above and beyond","special","appreciated","seen","understood","saved time",
        "solved","fixed","resolved","exceeded","delighted"
    }
    return any(m in t for m in reason_markers)

# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------
SYSTEM_ACK = (
    "You are a concise, friendly research assistant. Respond with a 3–10 word acknowledgement that mirrors the respondent's sentiment."
)
SYSTEM_SNAPBACK = (
    "You are a professional market research moderator. If the respondent's answer is unclear, nonsense, or irrelevant, "
    "politely re-ask the original question, making clear the first answer did not address it. Keep neutral tone."
)
SYSTEM_FOLLOWUP = (
    "You are a professional market research moderator. Generate ONE open-ended follow-up question (max 22 words). "
    "It must be a single sentence, >=6 words, end with '?' and be neutral."
)
# NEW: focused follow-up for retail “magic moments”
SYSTEM_FOLLOWUP_MAGIC = (
    "You are a professional research moderator for retail 'magic moments'. "
    "Ask ONE open-ended follow-up (6–22 words, single sentence, ends with '?'). "
    "Target ONLY the missing piece:\n"
    "• If the story lacks WHAT HAPPENED: ask for concrete sequence (who did what, where, when).\n"
    "• If the story lacks WHY IT WAS MAGIC: ask for the reason it felt special (emotion, expectation vs reality, problem solved).\n"
    "Do not ask about anything else. No pleasantries. No multiple questions."
)

# ---------------------------------------------------------------------
# Generators (OpenAI)
# ---------------------------------------------------------------------
async def _call_openai_chat(messages: list[dict], model: str, max_tokens: int) -> str:
    client = get_client()
    loop = asyncio.get_running_loop()

    def _block_call():
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
            stop=["\n"]
        )
        return resp.choices[0].message.content.strip()

    try:
        return await asyncio.wait_for(loop.run_in_executor(None, _block_call), timeout=OPENAI_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Model '{model}' timed out")

async def generate_ack(q: str, r: str, project_id: Optional[str] = None, model: str = DEFAULT_MODEL_ACK) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_ACK},
        {"role": "user", "content": f"Q: {q.strip()} A: {r.strip()} Return only acknowledgement."},
    ]
    text = await _call_openai_chat(messages, model, max_tokens=min(24, MAX_TOKENS))
    ack = text.split("\n")[0][:120]
    if contains_blocked_phrase(ack, project_id):
        avoid = ", ".join(sorted(get_block_phrases(project_id)))
        messages[0]["content"] += f" Avoid: {avoid}"
        ack2 = await _call_openai_chat(messages, model, max_tokens=min(24, MAX_TOKENS))
        if not contains_blocked_phrase(ack2, project_id):
            return ack2
    return ack

async def generate_snapback(q: str, r: str, project_id: Optional[str] = None, model: str = DEFAULT_MODEL_SNAPBACK) -> Optional[str]:
    # Only trigger if poor/irrelevant response (very short or empty)
    if not r or len(r.split()) <= 3:
        messages = [
            {"role": "system", "content": SYSTEM_SNAPBACK},
            {"role": "user", "content": f"Original question: {q.strip()} Respondent's answer: {r.strip()} Please re-ask the question clearly."},
        ]
        text = await _call_openai_chat(messages, model, max_tokens=min(48, MAX_TOKENS))
        snap = text.split("\n")[0][:200]
        if not contains_blocked_phrase(snap, project_id):
            return snap
    return None

def _fallback_followup(q: str, r: str) -> str:
    rlow = (r or "").lower()
    if any(k in rlow for k in ["funny","humor","humour","memorable","like","love","enjoy"]): return "What specifically made it stand out for you?"
    if any(k in rlow for k in ["confusing","unclear","didn't get","dont get","don't get","did not get"]): return "Which part felt confusing, and why?"
    if any(k in rlow for k in ["boring","slow","long"]): return "What made it feel boring, and how could it be improved?"
    if len(rlow.split()) <= 3: return "Could you share a bit more detail about that?"
    return "Can you give a specific example of what you mean?"

# UPDATED: adds magic-mode branching and takes context
async def generate_followup(
    question: str,
    response: str,
    project_id: Optional[str] = None,
    model: str = DEFAULT_MODEL_FOLLOWUP,
    context: Optional[dict] = None
) -> str:
magic_mode = True

    if magic_mode:
        has_what = _covers_what_happened(response)
        has_why  = _covers_why_magic(response)

        if not has_what and not has_why:
            target = "WHAT HAPPENED"
            instruction = "Ask specifically for the concrete sequence of events: who did what, where, and when."
        elif not has_what:
            target = "WHAT HAPPENED"
            instruction = "Ask for the concrete sequence of events: who did what, where, and when."
        elif not has_why:
            target = "WHY IT WAS MAGIC"
            instruction = "Ask for the reason it felt special—emotion, expectation vs reality, or problem solved."
        else:
            target = "WHY IT WAS MAGIC"
            instruction = "Ask what exactly made the peak moment feel magical in customer terms."

        messages = [
            {"role": "system", "content": SYSTEM_FOLLOWUP_MAGIC},
            {"role": "user", "content": (
                f"Original question: {question.strip()}\n"
                f"Respondent's answer: {response.strip()}\n"
                f"Target: {target}. {instruction}\n"
                "Return ONE follow-up question only."
            )},
        ]
        text = await _call_openai_chat(messages, model=model, max_tokens=min(96, MAX_TOKENS))
        t = (text or "").strip().replace("\n", " ")
        if not t or t == "?" or len(t.split()) < 6:
            t = ("Could you walk me through exactly what happened, step by step?"
                 if target == "WHAT HAPPENED"
                 else "What made that moment feel special compared with your usual expectations?")
        if not t.endswith("?"):
            t = t.rstrip(" .!") + "?"
        if contains_blocked_phrase(t, project_id):
            avoid = ", ".join(sorted(list(get_block_phrases(project_id))))
            messages[0]["content"] = SYSTEM_FOLLOWUP_MAGIC + (f" Avoid: {avoid}" if avoid else "")
            text2 = await _call_openai_chat(messages, model=model, max_tokens=min(96, MAX_TOKENS))
            t2 = (text2 or "").strip().replace("\n", " ")
            if t2 and t2 != "?" and len(t2.split()) >= 6 and t2.endswith("?") and not contains_blocked_phrase(t2, project_id):
                t = t2
        return t

    # --- default path (non-magic projects) ---
    messages = [
        {"role": "system", "content": SYSTEM_FOLLOWUP},
        {"role": "user", "content": f"Original question: {question.strip()} Respondent's answer: {response.strip()} Write ONE follow-up question only."},
    ]
    text = await _call_openai_chat(messages, model=model, max_tokens=min(96, MAX_TOKENS))
    t = (text or "").strip().replace("\n", " ")
    if not t or t == "?" or len(t.split()) < 6:
        t = _fallback_followup(question, response)
    if not t.endswith("?"):
        t = t.rstrip(" .!") + "?"
    if contains_blocked_phrase(t, project_id):
        avoid = ", ".join(sorted(list(get_block_phrases(project_id))))
        messages2 = [
            {"role": "system", "content": SYSTEM_FOLLOWUP + (" Avoid any of these terms: " + avoid if avoid else "")},
            {"role": "user", "content": f"Original question: {question.strip()} Respondent's answer: {response.strip()} Write ONE follow-up question only."},
        ]
        text2 = await _call_openai_chat(messages2, model=model, max_tokens=min(96, MAX_TOKENS))
        t2 = (text2 or "").strip().replace("\n", " ")
        if t2 and t2 != "?" and len(t2.split()) >= 6 and t2.endswith("?") and not contains_blocked_phrase(t2, project_id):
            t = t2
    return t

# ---------------------------------------------------------------------
# FastAPI app (TOP-LEVEL — this fixes the "app not found")
# ---------------------------------------------------------------------
app = FastAPI(title="Verbalytics 2.2 API", version="2.2.0")

@app.on_event("startup")
async def startup_blocklist():
    _refresh_blocklist(True)

@app.get("/health")
def health() -> dict:
    try:
        _refresh_blocklist(False)
    except Exception:
        pass
    with _block_lock:
        global_count = len(_block_global)
        project_count = len(_block_projects)
    return {
        "ok": True,
        "model_ack": DEFAULT_MODEL_ACK,
        "model_snapback": DEFAULT_MODEL_SNAPBACK,
        "model_followup": DEFAULT_MODEL_FOLLOWUP,
        "blocklist_version": _block_version,
        "blocklist_global_count": global_count,
        "blocklist_projects": project_count,
    }

@app.post("/verbalytics", response_model=VerbalyticsOutput)
async def verbalytics(payload: VerbalyticsInput = Body(...)):
    start = time.time()

    # Input blocklist (question + response)
    input_hits = sorted(set(
        find_blocked_phrases(payload.question, payload.project_id) +
        find_blocked_phrases(payload.response, payload.project_id)
    ))
    if input_hits and INPUT_BLOCK_MODE == "reject":
        raise HTTPException(status_code=422, detail={"message": "Input contains blocked phrases", "phrases": input_hits})

    want_score = "score" in payload.tasks
    want_ack = "ack" in payload.tasks
    want_follow = "followup" in payload.tasks

    # Always attempt snapback automatically (returns None if not needed)
    snap_task = asyncio.create_task(generate_snapback(payload.question, payload.response, payload.project_id))
    ack_task = asyncio.create_task(generate_ack(payload.question, payload.response, payload.project_id)) if want_ack else None
    # UPDATED: pass context
    follow_task = asyncio.create_task(
        generate_followup(payload.question, payload.response, payload.project_id, context=payload.context)
    ) if want_follow else None

    subs = None
    score = ai_like = None
    if want_score:
        subs = score_engine.subscores(payload.question, payload.response)
        score = score_engine.quality_score(payload.question, payload.response)
        ai_like = score_engine.ai_likelihood(payload.response)

    ack_res = follow_res = snap_res = None
    try:
        snap_res = await snap_task
        if ack_task: ack_res = await ack_task
        if follow_task: follow_res = await follow_task
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = int((time.time() - start) * 1000)
    return VerbalyticsOutput(
        subscores=subs,
        score=score,
        ai_likelihood=ai_like,
        ack=ack_res if want_ack else None,
        snapback=snap_res,
        followup=follow_res if want_follow else None,
        input_blocked_phrases=input_hits if input_hits else None,
        latency_ms=latency_ms,
    )

@app.post("/verbalytics/stream")
async def verbalytics_stream(payload: VerbalyticsInput = Body(...)):
    async def event_gen():
        start = time.time()

        want_score = "score" in payload.tasks
        want_ack = "ack" in payload.tasks
        want_follow = "followup" in payload.tasks

        # Input blocklist (flag-only in stream; enforce reject in sync endpoint)
        input_hits = sorted(set(
            find_blocked_phrases(payload.question, payload.project_id) +
            find_blocked_phrases(payload.response, payload.project_id)
        ))

        if want_score:
            subs = score_engine.subscores(payload.question, payload.response)
            score = score_engine.quality_score(payload.question, payload.response)
            ai_like = score_engine.ai_likelihood(payload.response)
            yield JSONResponse(content={
                "type": "score",
                "subscores": subs,
                "score": score,
                "ai_likelihood": ai_like,
                "input_blocked_phrases": input_hits or None
            }).body + b"\n"

        snap_task = asyncio.create_task(generate_snapback(payload.question, payload.response, payload.project_id))
        ack_task = asyncio.create_task(generate_ack(payload.question, payload.response, payload.project_id)) if want_ack else None
        # UPDATED: pass context
        follow_task = asyncio.create_task(
            generate_followup(payload.question, payload.response, payload.project_id, context=payload.context)
        ) if want_follow else None

        snap = await snap_task
        if snap:
            yield JSONResponse(content={"type": "snapback", "snapback": snap}).body + b"\n"

        if ack_task:
            ack = await ack_task
            yield JSONResponse(content={"type": "ack", "ack": ack}).body + b"\n"

        if follow_task:
            follow = await follow_task
            yield JSONResponse(content={"type": "followup", "followup": follow}).body + b"\n"

        latency_ms = int((time.time() - start) * 1000)
        yield JSONResponse(content={"type": "done", "latency_ms": latency_ms}).body + b"\n"

    return StreamingResponse(event_gen(), media_type="application/jsonl")

# ---------------- Admin: Blocklist Introspection & Refresh ----------------
@app.get("/admin/blocklist")
async def admin_blocklist(project: Optional[str] = None):
    phrases = get_block_phrases(project)
    with _block_lock:
        return {
            "version": _block_version,
            "global_count": len(_block_global),
            "projects_map_count": len(_block_projects),
            "project": _norm(project) if project else None,
            "project_specific_count": len(_block_projects.get(_norm(project), set())) if project else None,
            "merged_count": len(phrases),
            "phrases": sorted(list(phrases)),
        }

@app.post("/admin/blocklist/refresh")
async def admin_blocklist_refresh():
    _refresh_blocklist(True)
    with _block_lock:
        return {
            "ok": True,
            "version": _block_version,
            "global_count": len(_block_global),
            "projects_map_count": len(_block_projects),
        }
