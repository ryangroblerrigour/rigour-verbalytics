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
  - GOOGLE_BLOCKLIST_SHEET (CSV of global+project-specific blocklist)
  - VERBALYTICS_BLOCKLIST_REFRESH_SECONDS (default 300)

Run locally:
  uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import os, math, time, asyncio, csv, requests
from functools import lru_cache
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
DEFAULT_MODEL_SNAPBACK = os.getenv("MODEL_SNAPBACK", "gpt-4o-mini")
DEFAULT_MODEL_FOLLOWUP = os.getenv("MODEL_FOLLOWUP", "gpt-5")
MAX_TOKENS = int(os.getenv("VERBALYTICS_MAX_TOKENS", "200"))
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "12.0"))

BLOCKLIST_CSV_URL = os.getenv("GOOGLE_BLOCKLIST_SHEET", "")
BLOCKLIST_REFRESH_SECONDS = int(os.getenv("VERBALYTICS_BLOCKLIST_REFRESH_SECONDS", "300"))

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
    tasks: list[Literal["score","followup","snapback"]] = Field(default_factory=lambda: ["score","followup","snapback"])
    locale: Optional[str] = "en"
    context: Optional[Dict[str, Any]] = None
    project_id: Optional[str] = None

class VerbalyticsOutput(BaseModel):
    score: Optional[int] = None
    ai_likelihood: Optional[int] = None
    snapback: Optional[str] = None
    followup: Optional[str] = None
    latency_ms: Optional[int] = None

# ---------------------------------------------------------------------
# Blocklist Loader (global + project)
# ---------------------------------------------------------------------
_block_lock = RLock()
_block_global: set[str] = set()
_block_projects: dict[str,set[str]] = {}
_block_loaded_at: float = 0.0
_block_version: int = 0

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def _load_blocklist_csv(url: str) -> tuple[set[str], dict[str,set[str]]]:
    g, p = set(), {}
    if not url: return g, p
    try:
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        reader = csv.DictReader(resp.content.decode("utf-8",errors="ignore").splitlines())
        for row in reader:
            scope, proj, phrase = _norm(row.get("scope","")), _norm(row.get("project","")), _norm(row.get("phrase",""))
            if not phrase: continue
            if scope=="global": g.add(phrase)
            elif scope=="project" and proj: p.setdefault(proj,set()).add(phrase)
    except Exception:
        pass
    return g,p

def _refresh_blocklist(force: bool=False) -> None:
    global _block_global,_block_projects,_block_loaded_at,_block_version
    now=monotonic()
    with _block_lock:
        if not force and (now-_block_loaded_at)<BLOCKLIST_REFRESH_SECONDS: return
        g,p=_load_blocklist_csv(BLOCKLIST_CSV_URL)
        if g or p or force:
            _block_global, _block_projects, _block_loaded_at, _block_version = g,p,now,_block_version+1

def get_block_phrases(project_id: Optional[str]) -> set[str]:
    _refresh_blocklist(False)
    with _block_lock:
        phrases=set(_block_global)
        if project_id: phrases |= _block_projects.get(_norm(project_id),set())
        return phrases

def contains_blocked_phrase(text:str, project_id:Optional[str]) -> bool:
    if not text: return False
    lt=text.lower()
    return any(p in lt for p in get_block_phrases(project_id))

# ---------------------------------------------------------------------
# Scoring Engine
# ---------------------------------------------------------------------
class ScoreEngine:
    def _tokenish_count(self,s:str)->int: return max(1,len([w for w in s.strip().split() if w]))
    def quality_score(self,q:str,r:str)->int:
        if not r or len(r.strip())<2: return 5
        words=self._tokenish_count(r)
        detail=60*(1-math.exp(-words/20))
        q_terms=set(t.lower().strip(",.;:!?()[]{}") for t in q.split())
        r_terms=set(t.lower().strip(",.;:!?()[]{}") for t in r.split())
        rel=min(20,len(q_terms & r_terms)*2)
        score=detail+rel
        if words<=4: score-=15
        return max(0,min(100,int(round(score/10))*10))
    def ai_likelihood(self,r:str)->int:
        if not r: return 0
        words=self._tokenish_count(r)
        f=0
        if "," in r and "." in r and words>120: f+=20
        for m in ["as an ai","as a language model","in summary","overall,","moreover,","furthermore,"]:
            if m in r.lower(): f+=40
        if words>200: f+=20
        if words>400: f+=40
        return int(max(0,min(100,f)))

score_engine=ScoreEngine()

# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------
SYSTEM_SNAPBACK=("You are a concise, friendly research assistant. Respond with a 3–10 word acknowledgement that mirrors the respondent's sentiment.")
SYSTEM_FOLLOWUP=("You are a professional market research moderator. Generate ONE open-ended follow-up question (max 22 words). It must be a single sentence, >=6 words, end with '?' and be neutral.")

# ---------------------------------------------------------------------
# Generators (OpenAI)
# ---------------------------------------------------------------------
async def _call_openai_chat(messages:list[dict], model:str, max_tokens:int)->str:
    client=get_client(); loop=asyncio.get_running_loop()
    def _block_call():
        resp=client.chat.completions.create(model=model,messages=messages,max_completion_tokens=max_tokens)
        return resp.choices[0].message.content.strip()
    try:
        return await asyncio.wait_for(loop.run_in_executor(None,_block_call),timeout=OPENAI_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504,detail=f"Model '{model}' timed out")

async def generate_snapback(q:str,r:str,model:str=DEFAULT_MODEL_SNAPBACK)->str:
    messages=[{"role":"system","content":SYSTEM_SNAPBACK},{"role":"user","content":f"Q: {q.strip()} A: {r.strip()} Return only acknowledgement."}]
    text=await _call_openai_chat(messages,model,max_tokens=min(24,MAX_TOKENS))
    snap=text.split("\n")[0][:120]
    if contains_blocked_phrase(snap,None):
        avoid=", ".join(sorted(get_block_phrases(None)))
        messages[0]["content"]+=f" Avoid: {avoid}"
        snap2=await _call_openai_chat(messages,model,max_tokens=min(24,MAX_TOKENS))
        if not contains_blocked_phrase(snap2,None): return snap2
    return snap

def _fallback_followup(q:str,r:str)->str:
    r=(r or "").lower()
    if any(k in r for k in ["funny","humor","humour","memorable","like","love","enjoy"]): return "What specifically made it stand out for you?"
    if any(k in r for k in ["confusing","unclear","didn't get","dont get","don't get","did not get"]): return "Which part felt confusing, and why?"
    if any(k in r for k in ["boring","slow","long"]): return "What made it feel boring, and how could it be improved?"
    if len(r.split())<=3: return "Could you share a bit more detail about that?"
    return "Can you give a specific example of what you mean?"

async def generate_followup(question: str, response: str, project_id: Optional[str] = None, model: str = DEFAULT_MODEL_FOLLOWUP) -> str:
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
# Cache
# ---------------------------------------------------------------------
@lru_cache(maxsize=512)
def cached_snapback_key(q:str,r:str)->str: return f"{q.strip().lower()}||{r.strip().lower()}"

async def cached_snapback(q:str,r:str)->str:
    if len(r.strip().split())<=3: pass
    return await generate_snapback(q,r)

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app=FastAPI(title="Verbalytics 2.0 API",version="2.0.0")

@app.on_event("startup")
async def startup_blocklist(): _refresh_blocklist(True)

@app.get("/health")
def health() -> dict:
    try:
        _refresh_blocklist(False)
    except Exception:
        pass
    return {"ok": True, "model_snapback": DEFAULT_MODEL_SNAPBACK, "model_followup": DEFAULT_MODEL_FOLLOWUP, "blocklist_version": _block_version}

@app.get("/admin/blocklist")
async def admin_blocklist(project:Optional[str]=None):
    phrases=get_block_phrases(project)
    return {"version":_block_version,"global_count":len(_block_global),"project":project,"merged_count":len(phrases),"phrases":sorted(list(phrases))}

@app.post("/admin/blocklist/refresh")
async def admin_refresh():
    _refresh_blocklist(True)
    return {"ok":True,"version":_block_version,"global_count":len(_block_global),"projects":len(_block_projects)}

@app.post("/verbalytics", response_model=VerbalyticsOutput)
async def verbalytics(payload: VerbalyticsInput = Body(...)):
    start = time.time()

    tasks = []
    want_score = "score" in payload.tasks
    want_snap = "snapback" in payload.tasks
    want_follow = "followup" in payload.tasks

    if want_snap:
        tasks.append(generate_snapback(payload.question, payload.response, payload.project_id))
    else:
        tasks.append(asyncio.sleep(0, result=None))

    if want_follow:
        tasks.append(generate_followup(payload.question, payload.response, payload.project_id))
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

        snap_task = asyncio.create_task(generate_snapback(payload.question, payload.response, payload.project_id)) if want_snap else None
        follow_task = asyncio.create_task(generate_followup(payload.question, payload.response, payload.project_id)) if want_follow else None

        if snap_task:
            snap = await snap_task
            yield JSONResponse(content={"type": "snapback", "snapback": snap}).body + b"\n"

        if follow_task:
            follow = await follow_task
            yield JSONResponse(content={"type": "followup", "followup": follow}).body + b"\n"

        latency_ms = int((time.time() - start) * 1000)
        yield JSONResponse(content={"type": "done", "latency_ms": latency_ms}).body + b"\n"

    return StreamingResponse(event_gen(), media_type="application/jsonl")

# ------------------- Admin: blocklist introspection -------------------
@app.get("/admin/blocklist/full")
async def admin_get_blocklist(project: Optional[str] = None):
    phrases = get_block_phrases(project)
    with _block_lock:
        return {
            "version": _block_version,
            "loaded_seconds_ago": int(monotonic() - _block_loaded_at) if _block_loaded_at else None,
            "global_count": len(_block_global),
            "project": _norm(project) if project else None,
            "project_count": len(_block_projects.get(_norm(project), set())) if project else None,
            "merged_count": len(phrases),
            "phrases": sorted(list(phrases)),
        }

@app.post("/admin/blocklist/full/refresh")
async def admin_refresh_blocklist():
    _refresh_blocklist(True)
    with _block_lock:
        return {"ok": True, "version": _block_version, "global_count": len(_block_global), "projects": len(_block_projects)}
