# main.py

import os
import tempfile
from datetime import datetime
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional, List

import gspread
from google.oauth2.service_account import Credentials

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import openai

# -------------------------------
# 1) BLOCKLIST LOADING (Google Sheet)
# -------------------------------

GOOGLE_BLOCKLIST_SHEET = os.getenv("GOOGLE_BLOCKLIST_SHEET")  # e.g., "RigourVerbalytics Blocklist"
GOOGLE_LOGS_SHEET      = os.getenv("GOOGLE_LOGS_SHEET")      # e.g., "RigourVerbalytics Logs"
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")

def load_blocklist_from_sheet() -> List[tuple]:
    """
    Read every row in the Google Sheet “RigourVerbalytics Blocklist”
    and return a list of (project_id, phrase) pairs.
    If project_id is empty, that phrase is global.
    """
    entries = []
    try:
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmp:
            tmp.write(GOOGLE_CREDENTIALS_JSON)
            tmp.flush()
            creds = Credentials.from_service_account_file(tmp.name, scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ])
        client = gspread.authorize(creds)
        sheet = client.open(GOOGLE_BLOCKLIST_SHEET).sheet1
        rows = sheet.get_all_records()
        for row in rows:
            pid = (row.get("project_id") or "").strip()
            phrase = (row.get("phrase") or "").strip().lower()
            if phrase:
                entries.append((pid, phrase))
    except Exception as e:
        print(f"⚠️  Could not load blocklist from sheet: {e}")
    return entries

# Load once at startup
BLOCKLIST = load_blocklist_from_sheet()

def is_blocked(project_id: str, answer: str) -> bool:
    """
    Return True if 'answer' contains any phrase from BLOCKLIST.
    A phrase with empty project_id applies globally.
    """
    ans_low = answer.lower()
    for pid, phrase in BLOCKLIST:
        if phrase in ans_low and (pid == "" or pid.lower() == project_id.lower()):
            return True
    return False

# -------------------------------
# 2) LOGGING TO GOOGLE SHEETS
# -------------------------------

def get_sheets_client():
    """
    Return an authorized gspread client for both blocklist and logs.
    """
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmp:
        tmp.write(GOOGLE_CREDENTIALS_JSON)
        tmp.flush()
        creds = Credentials.from_service_account_file(tmp.name, scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ])
    return gspread.authorize(creds)

try:
    sheets_client = get_sheets_client()
    log_sheet     = sheets_client.open(GOOGLE_LOGS_SHEET).sheet1
except Exception as e:
    print(f"⚠️ Could not open logs sheet: {e}")
    log_sheet = None

def log_to_sheet(row_values: List):
    """
    Append a row to the “RigourVerbalytics Logs” sheet (if available).
    """
    if log_sheet:
        try:
            log_sheet.append_row(row_values)
        except Exception as e:
            print(f"⚠️  Failed to append log row: {e}")

# -------------------------------
# 3) SCORING MODELS SETUP
# -------------------------------

semantic_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
ai_tokenizer   = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
ai_model       = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")

# UK <-> US spelling maps
UK_TO_US = {
    "colour": "color",
    "favourite": "favorite",
    "organise": "organize",
    "centre": "center",
    "behaviour": "behavior",
    "analyse": "analyze",
    "theatre": "theater",
    "travelling": "traveling",
    "labour": "labor"
}
US_TO_UK = {us: uk for uk, us in UK_TO_US.items()}

def detect_locale_mismatch(requested_lang: str, answer: str) -> bool:
    """
    Return True if the answer's spelling suggests the OTHER locale.
    - If requested_lang == 'UK' but answer uses any US spelling, return True.
    - If requested_lang == 'USA' but answer uses any UK spelling, return True.
    """
    ans_low = answer.lower()
    if requested_lang.upper() == "UK":
        for us_spelling in US_TO_UK:
            if f" {us_spelling} " in f" {ans_low} ":
                return True
    if requested_lang.upper() == "USA":
        for uk_spelling in UK_TO_US:
            if f" {uk_spelling} " in f" {ans_low} ":
                return True
    return False

# -------------------------------
# 4) OPENAI SETUP FOR DYNAMIC PROBES
# -------------------------------

openai.api_key = os.getenv("OPENAI_API_KEY", "")

def make_snapback_text(question: str, answer: str) -> str:
    """
    Short snapback phrasing: ask them to elaborate on their poor answer.
    """
    return (
        f"Can you elaborate on what you mean by “{answer}” in the context of “{question}”?"
    )

def make_probe_text(question: str, answer: str) -> str:
    """
    Fallback probe if OpenAI key is missing or call fails.
    """
    snippet = answer.strip()
    return (
        f"You said “{snippet}” when asked “{question}.” "
        f"Could you tell me more about why/how that is important to you?"
    )

def get_dynamic_probe(question: str, answer: str) -> str:
    """
    Ask GPT to draft a customer-centric follow-up question
    based on the original question and their answer.
    """
    if not openai.api_key:
        return make_probe_text(question, answer)

    prompt = (
        "You are a market-research pro. Given this survey prompt:\n"
        f"  QUESTION: \"{question}\"\n"
        "and this respondent’s verbatim answer:\n"
        f"  ANSWER: \"{answer}\"\n"
        "Write a single, concise follow-up question that probes for "
        "more detail or emotion—something a human moderator would ask. "
        "Do not repeat the original question; instead, build on the "
        "respondent’s answer. Return only the follow-up sentence."
    )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=60
        )
        text = resp.choices[0].message.content.strip()
        return text
    except Exception:
        return make_probe_text(question, answer)

# -------------------------------
# 5) FASTAPI SETUP
# -------------------------------

API_KEY = os.getenv("API_KEY", "not-set")

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 6) Pydantic Schemas
# -------------------------------

class VerbalyticsRequest(BaseModel):
    respid: str
    project_id: str
    question_id: str
    question: str
    answer: str
    language: str               # "UK" or "USA"
    probe_required: Optional[str] = "no"   # "yes", "no", or "force"
    response_format: Optional[str] = "json"  # "json" or "csv"

class VerbalyticsResponse(BaseModel):
    respid: str
    project_id: str
    question_id: str
    quality_score: int
    ai_likelihood_score: int

    snapback_required: str   # "yes" or "no"
    snapback_text: Optional[str] = ""

    probe_required: str      # "yes", "no", or "force"
    probe_text: Optional[str] = ""

# -------------------------------
# 7) SCORING FUNCTIONS
# -------------------------------

def get_quality_score(question: str, answer: str, lang: str) -> int:
    """
    Compute a 1–100 quality score using semantic similarity + heuristics.
    Then apply a 20-point penalty if the answer’s spelling mismatches the requested lang.
    """
    try:
        clean_ans = answer.strip()
        if not clean_ans:
            return 1

        embeddings = semantic_model.encode([question, answer])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        if sim > 0.75:
            base_score = max(1, min(int(sim * 99) + 1, 100))
        else:
            q_words = set(question.lower().split())
            a_words = set(answer.lower().split())
            overlap = q_words & a_words
            keyword_score = min(len(overlap) / max(1, len(q_words)), 1.0)

            narrative_penalty = -0.15 if len(answer.split()) > 25 else 0.0
            length_bonus = 0.0
            if len(answer.split()) <= 3 and sim > 0.4:
                length_bonus = 0.1

            combined = (0.9 * sim) + (0.1 * keyword_score) + length_bonus + narrative_penalty
            base_score = int(min(max(combined, 0.0), 1.0) * 99) + 1

        # Apply locale penalty if mismatch detected
        if detect_locale_mismatch(lang, answer):
            base_score = max(1, base_score - 20)

        return base_score
    except Exception:
        return 200

def get_ai_likelihood_score(answer: str) -> int:
    """
    Compute an AI-likelihood score (0–100) using a RoBERTa-based classifier.
    """
    try:
        inputs = ai_tokenizer(answer, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = ai_model(**inputs).logits
        probs = torch.softmax(logits, dim=1).tolist()[0]
        return max(0, min(int(probs[1] * 100), 100))
    except Exception:
        return 200

# -------------------------------
# 8) API ENDPOINTS
# -------------------------------

@app.options("/check-verbalytics")
async def options_handler():
    return {}

@app.post("/check-verbalytics", response_model=VerbalyticsResponse)
async def check_verbalytics(
    payload: VerbalyticsRequest,
    x_api_key: Optional[str] = Header(None)
):
    # 1) Validate API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 2) Blocklist check
    if is_blocked(payload.project_id, payload.answer):
        response_data = {
            "respid":              payload.respid,
            "project_id":          payload.project_id,
            "question_id":         payload.question_id,
            "quality_score":       0,
            "ai_likelihood_score": 0,
            "snapback_required":   "yes",
            "snapback_text":       "Your response contains disallowed content. Please re-answer.",
            "probe_required":      "no",
            "probe_text":          ""
        }
        log_to_sheet([
            datetime.utcnow().isoformat(),
            payload.respid,
            payload.project_id,
            payload.question_id,
            payload.question,
            payload.answer,
            payload.language,
            response_data["quality_score"],
            response_data["ai_likelihood_score"],
            response_data["snapback_required"],
            response_data["probe_required"],
            response_data["snapback_text"],
            response_data["probe_text"]
        ])
        return JSONResponse(content=response_data)

    # 3) Compute scores
    q_score  = get_quality_score(payload.question, payload.answer, payload.language)
    ai_score = get_ai_likelihood_score(payload.answer)

    # 4) Determine snapback vs probe
    incoming_probe = (payload.probe_required or "no").lower()
    if incoming_probe not in ("yes", "no", "force"):
        incoming_probe = "no"

    if q_score <= 20 and incoming_probe != "force":
        # Snapback case
        snapback_required = "yes"
        snap_text = make_snapback_text(payload.question, payload.answer)

        probe_required = "no"
        probe_text = ""
    else:
        # Not a snapback (either q_score > 20 or forced probe)
        snapback_required = "no"
        snap_text = ""

        if incoming_probe == "force":
            probe_required = "yes"
            probe_text = get_dynamic_probe(payload.question, payload.answer)
        elif incoming_probe == "yes" and q_score > 20:
            probe_required = "yes"
            probe_text = get_dynamic_probe(payload.question, payload.answer)
        else:
            probe_required = "no"
            probe_text = ""

    # 5) Assemble response object
    response_data = {
        "respid":               payload.respid,
        "project_id":           payload.project_id,
        "question_id":          payload.question_id,
        "quality_score":        q_score,
        "ai_likelihood_score":  ai_score,
        "snapback_required":    snapback_required,
        "snapback_text":        snap_text,
        "probe_required":       probe_required,
        "probe_text":           probe_text
    }

    # 6) Log to Google Sheets
    log_to_sheet([
        datetime.utcnow().isoformat(),
        payload.respid,
        payload.project_id,
        payload.question_id,
        payload.question,
        payload.answer,
        payload.language,
        response_data["quality_score"],
        response_data["ai_likelihood_score"],
        response_data["snapback_required"],
        response_data["probe_required"],
        response_data["snapback_text"] or response_data["probe_text"]
    ])

    # 7) Return in requested format
    if payload.response_format.lower() == "csv":
        text_field = response_data["snapback_text"] or response_data["probe_text"]
        csv_line = ",".join([
            str(response_data["quality_score"]),
            str(response_data["ai_likelihood_score"]),
            response_data["snapback_required"],
            text_field.replace(",", ";")
        ])
        return PlainTextResponse(csv_line)
    else:
        return JSONResponse(content=response_data)

# -------------------------------
# 9) Local Uvicorn Runner
# -------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
